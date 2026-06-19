//! Native-Rust neural models (NN-17 deferred; GRU + LSTM via the `candle` ML framework).
//! Gated behind the `neural` cargo feature. This module holds the shared infrastructure:
//! NumPy RNG (for the Reptile finetune's pandas shuffle), the `.pth` checkpoint loader,
//! candle layer ops, an AdamW optimizer matching torch, and the Reptile finetune driver.

pub mod numpy_rng;

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, Var, D};
use candle_nn::ops;

use numpy_rng::MT19937;

/// Device for neural work. Built with `--features neural-cuda`, candle's CUDA backend is
/// compiled in and this returns the GPU (matching the original srs-benchmark, which trains
/// GRU/LSTM on CUDA); otherwise (plain `neural`, or no GPU present) it falls back to CPU.
pub fn device() -> Device {
    match Device::cuda_if_available(0) {
        Ok(d) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| eprintln!("[neural] device = {d:?}"));
            d
        }
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| eprintln!("[neural] CUDA init failed ({e}); falling back to CPU"));
            Device::Cpu
        }
    }
}

/// Load a PyTorch `.pth` checkpoint into a name→Tensor map (f32, CPU).
pub fn load_pth(path: &str) -> Result<HashMap<String, Tensor>> {
    let tensors = candle_core::pickle::read_all(path)?;
    Ok(tensors.into_iter().collect())
}

// ----------------------------------------------------------------------------------------
// Layer ops (functional; operate on candle Tensors, differentiable wrt the weight tensors)
// ----------------------------------------------------------------------------------------

/// `y = x @ w.T + b`, `x` is `[N, in]`, `w` is `[out, in]`, `b` is `[out]`.
pub fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.matmul(&w.t()?)?.broadcast_add(b)
}

/// `LayerNorm` over the last dim with NO bias (torch `nn.LayerNorm(H, bias=False)`):
/// `(x - mean) / sqrt(var + eps) * weight`, `var` = biased variance, `eps = 1e-5`.
pub fn layernorm_no_bias(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    xn.broadcast_mul(weight)
}

/// `LayerNorm` over the last dim WITH bias (torch `nn.LayerNorm(H)`):
/// `(x - mean) / sqrt(var + eps) * weight + bias`.
pub fn layernorm_with_bias(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    layernorm_no_bias(x, weight, eps)?.broadcast_add(bias)
}

/// One step of a torch `nn.GRU` cell (gate order `[r, z, n]`):
///   r = σ(ih_r + hh_r),  z = σ(ih_z + hh_z),  n = tanh(ih_n + r·hh_n),
///   h' = (1-z)·n + z·h.
/// `input` `[B, in]`, `h` `[B, H]`, `w_ih` `[3H, in]`, `w_hh` `[3H, H]`, biases `[3H]`.
pub fn gru_cell(
    input: &Tensor,
    h: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
) -> Result<Tensor> {
    let gi = input.matmul(&w_ih.t()?)?.broadcast_add(b_ih)?;
    let gh = h.matmul(&w_hh.t()?)?.broadcast_add(b_hh)?;
    let gi = gi.chunk(3, 1)?;
    let gh = gh.chunk(3, 1)?;
    let r = ops::sigmoid(&(&gi[0] + &gh[0])?)?;
    let z = ops::sigmoid(&(&gi[1] + &gh[1])?)?;
    let n = (&gi[2] + (&r * &gh[2])?)?.tanh()?;
    // h' = z*h + (1-z)*n  ==  z*h - (z-1)*n
    (&z * h)? - ((&z - 1.0)? * n)?
}

/// One step of a torch `nn.LSTM` cell (gate order `[i, f, g, o]`):
///   i = σ(...), f = σ(...), g = tanh(...), o = σ(...),
///   c' = f·c + i·g,  h' = o·tanh(c').
/// Returns `(h', c')`. `w_ih` `[4H, in]`, `w_hh` `[4H, H]`, biases `[4H]`.
#[allow(clippy::too_many_arguments)]
pub fn lstm_cell(
    input: &Tensor,
    h: &Tensor,
    c: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let gates = (input.matmul(&w_ih.t()?)?.broadcast_add(b_ih)?
        + h.matmul(&w_hh.t()?)?.broadcast_add(b_hh)?)?;
    let chunks = gates.chunk(4, 1)?;
    let i = ops::sigmoid(&chunks[0])?;
    let f = ops::sigmoid(&chunks[1])?;
    let g = chunks[2].tanh()?;
    let o = ops::sigmoid(&chunks[3])?;
    let c_new = ((&f * c)? + (&i * &g)?)?;
    let h_new = (&o * c_new.tanh()?)?;
    Ok((h_new, c_new))
}

/// Binary cross-entropy with `reduction="none"`, matching torch: each log term is clamped to
/// a minimum of −100. `p` and `y` are `[B]`. Returns `[B]`.
pub fn bce(p: &Tensor, y: &Tensor) -> Result<Tensor> {
    let log_p = p.log()?.clamp(-100.0, f32::MAX)?;
    let log_1mp = (1.0 - p)?.log()?.clamp(-100.0, f32::MAX)?;
    let term1 = (y * log_p)?;
    let term2 = ((1.0 - y)? * log_1mp)?;
    (term1 + term2)?.neg()
}

/// Gather rows `idx[b]` along dim 1 of a `[B, L, H]` tensor → `[B, H]`.
pub fn gather_last(x: &Tensor, idx: &[u32]) -> Result<Tensor> {
    let (b, _l, h) = x.dims3()?;
    let dev = x.device();
    // index tensor [B, 1, H] each row filled with idx[b]
    let mut data = Vec::with_capacity(b * h);
    for &i in idx {
        for _ in 0..h {
            data.push(i);
        }
    }
    let index = Tensor::from_vec(data, (b, 1, h), dev)?;
    x.gather(&index, 1)?.reshape((b, h))
}

// ----------------------------------------------------------------------------------------
// AdamW optimizer (matches torch.optim.AdamW: decoupled weight decay, bias-corrected).
// ----------------------------------------------------------------------------------------

pub struct AdamW {
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    step: i64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl AdamW {
    /// Cold start (m = v = 0, step = 0) — the benchmark loads an EMPTY optimizer state for the
    /// finetune (`*_opt_pretrain.pth` has no per-param state), so this is correct.
    pub fn new(vars: &[Var], beta1: f64, beta2: f64, eps: f64) -> Result<Self> {
        let mut m = Vec::with_capacity(vars.len());
        let mut v = Vec::with_capacity(vars.len());
        for var in vars {
            m.push(var.as_tensor().zeros_like()?);
            v.push(var.as_tensor().zeros_like()?);
        }
        Ok(Self { m, v, step: 0, beta1, beta2, eps })
    }

    /// One AdamW step. `grads[k]` is the (already grad-clipped) gradient for `vars[k]`,
    /// `None` if that var had no gradient this batch.
    pub fn step(&mut self, vars: &[Var], grads: &[Option<Tensor>], lr: f64, wd: f64) -> Result<()> {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        let bc2_sqrt = bc2.sqrt();
        for (k, var) in vars.iter().enumerate() {
            // Detach the gradient and the current param: the optimizer math must NOT retain any
            // autograd graph (torch's optimizer runs under `no_grad`). Without this, each step's
            // forward graph stays alive through `m`/`v`, so memory grows with
            // `inner_steps × batches` and OOMs on large users (and on the 12 GB GPU).
            let g = match &grads[k] {
                Some(g) => g.detach(),
                None => continue,
            };
            // decoupled weight decay: param *= (1 - lr*wd)
            let p0 = (var.as_tensor().detach() * (1.0 - lr * wd))?;
            // m = beta1*m + (1-beta1)*g ; v = beta2*v + (1-beta2)*g^2 (stored detached → no chain)
            self.m[k] = ((&self.m[k] * self.beta1)? + (&g * (1.0 - self.beta1))?)?.detach();
            self.v[k] = ((&self.v[k] * self.beta2)? + (g.sqr()? * (1.0 - self.beta2))?)?.detach();
            // denom = sqrt(v)/sqrt(bc2) + eps ; p -= (lr/bc1) * m/denom
            let denom = ((self.v[k].sqrt()? / bc2_sqrt)? + self.eps)?;
            let update = ((&self.m[k] / denom)? * (lr / bc1))?;
            let p = (p0 - update)?;
            var.set(&p.detach())?;
        }
        Ok(())
    }
}

/// Global L2 grad-norm clip (torch `clip_grad_norm_`): if `total_norm > max_norm`, scale every
/// gradient by `max_norm / (total_norm + 1e-6)`. Returns the per-var (possibly scaled) grads.
pub fn clip_grad_norm(
    vars: &[Var],
    grads: &candle_core::backprop::GradStore,
    max_norm: f64,
) -> Result<Vec<Option<Tensor>>> {
    let mut out: Vec<Option<Tensor>> = Vec::with_capacity(vars.len());
    let mut total: f64 = 0.0;
    for var in vars {
        match grads.get(var.as_tensor()) {
            Some(g) => {
                let n = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                total += n;
                out.push(Some(g.clone()));
            }
            None => out.push(None),
        }
    }
    let total_norm = total.sqrt();
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for g in out.iter_mut().flatten() {
            *g = (&*g * scale)?;
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------------------
// Reptile finetune (script.py GRU/LSTM per-user path → reptile_trainer{,_gru}.py::finetune)
// ----------------------------------------------------------------------------------------

/// One training example: the prior-review feature sequence + the current interval + label.
pub struct SeqItem {
    /// Prior reviews, each a raw feature vector (`[delta]` or `[delta, duration]`, then the
    /// rating appended last). Length = `seq_len`.
    pub seq: Vec<Vec<f32>>,
    pub delta_t: f32,
    pub y: f32,
}

impl SeqItem {
    pub fn seq_len(&self) -> usize {
        self.seq.len()
    }
}

/// A trainable neural model with a differentiable batched forward.
pub trait NeuralModel {
    /// Trainable variables in a stable order (used by AdamW, the reg term, and grad clipping).
    fn vars(&self) -> Vec<Var>;
    /// Retentions `[B]` for the batch (differentiable wrt `vars`).
    fn forward(&self, batch: &[&SeqItem]) -> Result<Tensor>;
}

/// Reptile finetune hyperparameters (from `reptile_trainer{,_gru}.py`'s DEFAULT_FINETUNE_PARAMS).
#[derive(Clone, Copy)]
pub struct FinetuneParams {
    pub lr_start_raw: f64,
    pub lr_middle_raw: f64,
    pub lr_end_raw: f64,
    pub warmup_steps: usize,
    pub batch_size_exp: f64,
    pub clip_norm: f64,
    pub reg_scale: f64,
    pub inner_steps: usize,
    pub recency_weight: f64,
    pub recency_degree: f64,
    pub weight_decay: f64,
    pub inner_adam_beta1: f64,
    pub inner_adam_beta2: f64,
    /// `BATCH_SIZE` of the model's trainer (GRU 8192, LSTM 16384).
    pub batch_size: usize,
}

const MAX_SEQ_LEN: usize = 64;
const ADAM_EPS: f64 = 1e-8;

/// Piecewise-linear LR (warmup then linear-to-end), evaluated at inner step `k` (0-indexed),
/// matching `PiecewiseLinearScheduler` advanced once per inner step.
fn piecewise_lr(p: &FinetuneParams, k: usize) -> f64 {
    let scale = 16000f64.powf(1.0 - p.batch_size_exp);
    let lr_start = p.lr_start_raw * scale;
    let lr_middle = p.lr_middle_raw * scale;
    let lr_end = p.lr_end_raw * scale;
    let nw = p.warmup_steps as f64;
    let nt = p.inner_steps as f64;
    if (k as f64) < nw {
        lr_start + (lr_middle - lr_start) * (k as f64) / nw
    } else {
        lr_middle + (lr_end - lr_middle) * (k as f64 - nw) / (nt - nw)
    }
}

/// Run the Reptile finetune on `model` (whose vars start at the meta/pretrain weights) over the
/// training `items` (in review_th order). Mutates the model's vars in place.
pub fn finetune<M: NeuralModel>(model: &M, items: &[SeqItem], p: &FinetuneParams) -> Result<()> {
    let l = items.len();
    if l == 0 {
        return Ok(());
    }
    let vars = model.vars();
    let dev = device();

    // Snapshot meta params for the L2-to-meta penalty (detached constants).
    let meta: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().detach()).collect();

    // Recency weights: w = 1 + rw * x^rd, x = linspace(0,1,L), then normalised to mean 1.
    let mut weights = vec![0f64; l];
    let mut wsum = 0.0;
    for (k, w) in weights.iter_mut().enumerate() {
        let x = if l <= 1 { 0.0 } else { k as f64 / (l as f64 - 1.0) };
        *w = 1.0 + p.recency_weight * x.powf(p.recency_degree);
        wsum += *w;
    }
    let wnorm = l as f64 / wsum;
    for w in weights.iter_mut() {
        *w *= wnorm;
    }

    // Shuffle rows once (pandas df.sample(frac=1, random_state=2025)), then drop seq_len>64,
    // then chunk into contiguous batches of 8192 (matching BatchDataset, sort_by_length=False).
    let perm = MT19937::permutation(2025, l);
    let kept: Vec<usize> = perm
        .into_iter()
        .filter(|&i| items[i].seq_len() <= MAX_SEQ_LEN)
        .collect();
    let batches: Vec<Vec<usize>> = kept
        .chunks(p.batch_size)
        .map(|c| c.to_vec())
        .collect();
    if batches.is_empty() {
        return Ok(());
    }

    let mut opt = AdamW::new(&vars, p.inner_adam_beta1, p.inner_adam_beta2, ADAM_EPS)?;

    for k in 0..p.inner_steps {
        let lr = piecewise_lr(p, k);
        for batch in &batches {
            let bs = batch.len();
            let batch_items: Vec<&SeqItem> = batch.iter().map(|&i| &items[i]).collect();
            let batch_y: Vec<f32> = batch.iter().map(|&i| items[i].y).collect();
            let batch_w: Vec<f32> = batch.iter().map(|&i| weights[i] as f32).collect();

            let retentions = model.forward(&batch_items)?; // [bs]
            let y_t = Tensor::from_vec(batch_y, bs, &dev)?;
            let w_t = Tensor::from_vec(batch_w, bs, &dev)?;

            let loss_vec = (bce(&retentions, &y_t)? * w_t)?; // [bs]
            let data_mean = loss_vec.mean_all()?; // scalar
            let scaled = (data_mean * (bs as f64).powf(p.batch_size_exp))?;

            // reg = reg_scale * Σ_v (v - meta_v)^2
            let mut reg = Tensor::zeros((), DType::F32, &dev)?;
            for (var, m) in vars.iter().zip(meta.iter()) {
                reg = (reg + (var.as_tensor() - m)?.sqr()?.sum_all()?)?;
            }
            let loss = (scaled + (reg * p.reg_scale)?)?;

            let grads = loss.backward()?;
            let clipped = clip_grad_norm(&vars, &grads, p.clip_norm)?;
            opt.step(&vars, &clipped, lr, p.weight_decay)?;
        }
    }
    Ok(())
}
