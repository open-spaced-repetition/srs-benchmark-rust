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

/// CosineAnnealingLR with eta_min=0: lr(step) = base*(1+cos(pi*step/T_max))/2.
pub fn cosine_lr(base: f64, t_max: usize, step: usize) -> f64 {
    if t_max == 0 {
        return base;
    }
    base * (1.0 + (std::f64::consts::PI * step as f64 / t_max as f64).cos()) / 2.0
}

/// xorshift64* — deterministic, fast, good enough for shuffling batch order.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn shuffle(&mut self, v: &mut [usize]) {
        // Fisher–Yates
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
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

/// Train a model, returning the best parameters by eval loss (mirrors `Trainer.train`).
pub fn train<M: BatchModel>(m: &M, tc: &TrainConfig) -> Vec<f64> {
    let n = m.n_rows();
    // Batching: stable sort by seq_len, then contiguous chunks of batch_size.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&r| m.seq_len(r)); // sort_by_key is stable
    let bs = tc.batch_size.max(1);
    let batches: Vec<Vec<usize>> = order.chunks(bs).map(|c| c.to_vec()).collect();
    let batch_nums = batches.len();
    let t_max = batch_nums * tc.n_epoch;

    let mut params = m.init_params();
    let mut adam = Adam::new(m.n_params(), tc.betas);
    let mut best_loss = f64::INFINITY;
    let mut best_w = params.clone();
    let mut rng = Rng(2023);
    let mut step = 0usize;

    for _epoch in 0..tc.n_epoch {
        let loss = eval_loss(m, &params);
        if loss < best_loss {
            best_loss = loss;
            best_w = params.clone();
        }
        let mut order_b: Vec<usize> = (0..batch_nums).collect();
        rng.shuffle(&mut order_b);
        for &bi in &order_b {
            let g = m.grad(&params, &batches[bi]);
            let lr = cosine_lr(tc.lr, t_max, step);
            adam.step(&mut params, &g, lr);
            step += 1;
        }
    }
    let loss = eval_loss(m, &params);
    if loss < best_loss {
        best_w = params.clone();
    }
    best_w
}
