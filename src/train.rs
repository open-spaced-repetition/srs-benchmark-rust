//! Training infrastructure shared by Adam-optimized models: an Adam optimizer (matching
//! `torch.optim.Adam` defaults, no weight decay — `BaseModel.get_optimizer`), a
//! CosineAnnealingLR schedule, and a generic training driver mirroring
//! `script.py::Trainer.train` (BCE summed × weights per batch, best weights by eval loss).
//!
//! Bit-exactness is not required (rule #5 is a ±0.0005 tolerance), so batch order uses a
//! cheap deterministic RNG rather than reproducing torch's `randperm`.

/// Adam (torch defaults: betas (0.9, 0.999), eps 1e-8, no weight decay).
pub struct Adam {
    b1: f64,
    b2: f64,
    eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: f64,
}

impl Adam {
    pub fn new(n: usize, betas: (f64, f64)) -> Self {
        Adam {
            b1: betas.0,
            b2: betas.1,
            eps: 1e-8,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0.0,
        }
    }

    /// One Adam step with an explicit learning rate (set by the cosine schedule).
    pub fn step(&mut self, params: &mut [f64], grad: &[f64], lr: f64) {
        self.t += 1.0;
        let bc1 = 1.0 - self.b1.powf(self.t);
        let bc2 = 1.0 - self.b2.powf(self.t);
        for i in 0..params.len() {
            let g = grad[i];
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g;
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g;
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}

/// torch CosineAnnealingLR (eta_min=0) uses a *recurrent* update, not the closed form;
/// in floating point they differ slightly, which matters for chaotic models. This
/// advances lr from step k to k+1: `lr *= (1+cos(pi*(k+1)/T)) / (1+cos(pi*k/T))`.
fn cosine_advance(lr: f64, t_max: usize, k: usize) -> f64 {
    if t_max == 0 {
        return lr;
    }
    let pi = std::f64::consts::PI;
    let num = 1.0 + (pi * (k as f64 + 1.0) / t_max as f64).cos();
    let den = 1.0 + (pi * k as f64 / t_max as f64).cos();
    lr * num / den
}

/// ATen's MT19937 engine, reproduced exactly (verified against torch 2.10 for seed 2023).
/// Used so the batch-visitation order matches `BatchLoader`'s `torch.randperm`, which is
/// the only uncontrolled source of variance vs the Python trained models.
pub struct Mt19937 {
    s: [u32; 624],
    left: i32,
    next: usize,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_b0df;
    const UMASK: u32 = 0x8000_0000;
    const LMASK: u32 = 0x7fff_ffff;

    pub fn new(seed: u32) -> Self {
        let mut s = [0u32; 624];
        s[0] = seed;
        for j in 1..624 {
            s[j] = 1812433253u32
                .wrapping_mul(s[j - 1] ^ (s[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        Mt19937 { s, left: 1, next: 0 }
    }

    fn next_state(&mut self) {
        let s = &mut self.s;
        self.left = Self::N as i32;
        self.next = 0;
        for j in 0..(Self::N - Self::M) {
            let y = (s[j] & Self::UMASK) | (s[j + 1] & Self::LMASK);
            s[j] = s[j + Self::M] ^ (y >> 1) ^ if y & 1 != 0 { Self::MATRIX_A } else { 0 };
        }
        for j in (Self::N - Self::M)..(Self::N - 1) {
            let y = (s[j] & Self::UMASK) | (s[j + 1] & Self::LMASK);
            s[j] = s[j - (Self::N - Self::M)] ^ (y >> 1) ^ if y & 1 != 0 { Self::MATRIX_A } else { 0 };
        }
        let y = (s[Self::N - 1] & Self::UMASK) | (s[0] & Self::LMASK);
        s[Self::N - 1] =
            s[Self::M - 1] ^ (y >> 1) ^ if y & 1 != 0 { Self::MATRIX_A } else { 0 };
    }

    pub fn next_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left <= 0 {
            self.next_state();
        }
        let mut y = self.s[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `torch.randperm(n, generator)` for the CPU generator: forward Fisher–Yates with a
    /// 32-bit draw per step.
    pub fn randperm(&mut self, n: usize) -> Vec<usize> {
        let mut r: Vec<usize> = (0..n).collect();
        for i in 0..n.saturating_sub(1) {
            let z = (self.next_u32() as usize) % (n - i);
            r.swap(i, i + z);
        }
        r
    }
}

/// A model that can be trained via the shared Adam loop. Holds its own train-set features;
/// `predict`/`grad` index into them.
pub trait BatchModel {
    fn n_params(&self) -> usize;
    fn init_params(&self) -> Vec<f64>;
    /// Number of training rows.
    fn n_rows(&self) -> usize;
    /// Sequence length per row (drives sort-by-length batching). Constant for static models.
    fn seq_len(&self, row: usize) -> usize;
    /// label (0/1) and loss weight for a row.
    fn y(&self, row: usize) -> f64;
    fn weight(&self, row: usize) -> f64;
    /// Predicted recall probability for each row in `idx`.
    fn predict(&self, params: &[f64], idx: &[usize]) -> Vec<f64>;
    /// Gradient of `sum_i weight_i * BCE(p_i, y_i)` over `idx`, w.r.t. params.
    fn grad(&self, params: &[f64], idx: &[usize]) -> Vec<f64>;
    /// Apply the model's parameter clipper after an optimizer step (default: none).
    /// Mirrors `Trainer`'s `apply_parameter_clipper`.
    fn clip_params(&self, _params: &mut [f64]) {}
}

/// Hyperparameters for a training run (from `BaseModel` unless overridden).
pub struct TrainConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub n_epoch: usize,
    pub batch_size: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            lr: 4e-2,
            betas: (0.9, 0.999),
            n_epoch: 5,
            batch_size: 512,
        }
    }
}

#[inline]
fn bce(p: f64, y: f64) -> f64 {
    let eps = f64::EPSILON;
    let pc = p.clamp(eps, 1.0 - eps);
    -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln())
}

/// Mean weighted BCE over all rows (the `Trainer.eval` train-loss when test_set=None).
fn eval_loss<M: BatchModel>(m: &M, params: &[f64]) -> f64 {
    let n = m.n_rows();
    let all: Vec<usize> = (0..n).collect();
    let p = m.predict(params, &all);
    let mut loss = 0.0;
    for i in 0..n {
        loss += m.weight(i) * bce(p[i], m.y(i));
    }
    loss / n as f64
}

/// Train a model from its own `init_params`, returning the best parameters by eval loss.
pub fn train<M: BatchModel>(m: &M, tc: &TrainConfig) -> Vec<f64> {
    let init = m.init_params();
    train_with_init(m, tc, init)
}

/// Train a model from explicit initial parameters (mirrors `Trainer.train`).
pub fn train_with_init<M: BatchModel>(m: &M, tc: &TrainConfig, init: Vec<f64>) -> Vec<f64> {
    let n = m.n_rows();
    // Batching: stable sort by seq_len, then contiguous chunks of batch_size.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&r| m.seq_len(r)); // sort_by_key is stable
    let bs = tc.batch_size.max(1);
    let batches: Vec<Vec<usize>> = order.chunks(bs).map(|c| c.to_vec()).collect();
    let batch_nums = batches.len();
    let t_max = batch_nums * tc.n_epoch;

    let mut params = init;
    let mut adam = Adam::new(m.n_params(), tc.betas);
    let mut best_loss = f64::INFINITY;
    let mut best_w = params.clone();
    // BatchLoader uses a torch.Generator seeded 2023, advanced once per epoch by randperm.
    let mut gen = Mt19937::new(2023);
    let mut step = 0usize;
    let mut lr = tc.lr; // lr[0] = base; advanced recurrently after each step

    for _epoch in 0..tc.n_epoch {
        let loss = eval_loss(m, &params);
        if loss < best_loss {
            best_loss = loss;
            best_w = params.clone();
        }
        let order_b = gen.randperm(batch_nums);
        for &bi in &order_b {
            let g = m.grad(&params, &batches[bi]);
            adam.step(&mut params, &g, lr);
            m.clip_params(&mut params);
            lr = cosine_advance(lr, t_max, step);
            step += 1;
        }
    }
    let loss = eval_loss(m, &params);
    if loss < best_loss {
        best_w = params.clone();
    }
    best_w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn randperm_matches_torch_seed2023() {
        // Ground truth from torch 2.10 (CPU generator, manual_seed(2023)).
        let mut g = Mt19937::new(2023);
        assert_eq!(g.randperm(8), vec![7, 3, 2, 0, 5, 1, 4, 6]);
        assert_eq!(g.randperm(8), vec![4, 7, 6, 0, 3, 1, 5, 2]);
        assert_eq!(g.randperm(8), vec![7, 3, 0, 6, 2, 4, 1, 5]);

        let mut g80 = Mt19937::new(2023);
        let expected80 = vec![
            39, 42, 62, 41, 53, 19, 63, 60, 28, 67, 11, 66, 54, 73, 45, 8, 6, 77, 51, 4, 68,
            34, 37, 9, 20, 59, 10, 31, 25, 75, 71, 12, 74, 69, 38, 1, 13, 50, 17, 48, 29, 76,
            35, 5, 61, 3, 22, 47, 16, 36, 15, 27, 56, 43, 32, 7, 79, 65, 49, 24, 44, 18, 40,
            33, 57, 64, 46, 72, 30, 0, 2, 55, 52, 14, 58, 23, 26, 21, 78, 70,
        ];
        assert_eq!(g80.randperm(80), expected80);
    }
}
