//! LSTM — `models/lstm.py`. A pretrained meta-model (loaded from `pretrain/<name>_pretrain.pth`)
//! is fine-tuned per user/split via the Reptile `finetune` (`reptile_trainer.py`), then predicts.
//! Built on the `candle` ML framework (gated behind the `neural` cargo feature).
//!
//! Architecture (matches lstm.py, `n_hidden=20`, `n_curves=3`): a `process` Sequential
//!   Linear(n_in→20) → SiLU → LN(no bias) → Linear(20→20) → SiLU →
//!   ResBlock[ LN(no bias) → LSTM(20→20) ] →
//!   ResBlock[ LN(WITH bias) → LSTM(20→20) ] →
//!   ResBlock[ LN(no bias) → Linear → SiLU → LN(no bias) → Linear → SiLU ] →
//!   LN(no bias) → Linear(20→20) → SiLU
//! then three heads w_fc/s_fc/d_fc (Linear 20→3). `ResBlock(x) = module(x) + x`. The single
//! delta feature is `log(1e-5+Δt)`; with `--duration` a second feature `log(clamp(dur,100,60000))`
//! is added; both are normalised by the loaded `input_mean`/`input_std` (length 1 or 2). Rating is
//! one-hot(4)-expanded. Output is a 3-curve mixture forgetting curve. The eval row-set equals every
//! other matching `--short[ --secs]` model (shared base pipeline), so `size` is exact by construction.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Result, Tensor, Var};
use candle_nn::ops;

use super::ModelOutput;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::neural::{
    self, bce as _bce, finetune, gather_last, layernorm_no_bias, layernorm_with_bias, linear,
    lstm_cell, FinetuneParams, NeuralModel, SeqItem,
};
use crate::split::time_series_split;

const N_HIDDEN: usize = 20;
const N_CURVES: usize = 3;
const LN_EPS: f64 = 1e-5;
const PREDICT_BATCH: usize = 8192;

/// Trainable LSTM model. Weights are stored as candle `Var`s keyed by their torch state_dict
/// name so they load directly from the `.pth`.
pub struct Lstm {
    w: HashMap<String, Var>,
    names: Vec<String>,
    pretrain: HashMap<String, Tensor>,
    /// `input_mean`/`input_std` buffers: length 1 (no duration) or 2 (with duration).
    input_mean: Vec<f32>,
    input_std: Vec<f32>,
    use_duration: bool,
    n_in: usize,
    device: Device,
}

/// The 31 trainable parameter names, in a stable order (the LSTM `.pth` layout).
const PARAM_NAMES: [&str; 31] = [
    "process.0.weight",
    "process.0.bias",
    "process.2.weight",
    "process.3.weight",
    "process.3.bias",
    "process.5.module.0.weight",
    "process.5.module.1.module.weight_ih_l0",
    "process.5.module.1.module.weight_hh_l0",
    "process.5.module.1.module.bias_ih_l0",
    "process.5.module.1.module.bias_hh_l0",
    "process.6.module.0.weight",
    "process.6.module.0.bias",
    "process.6.module.1.module.weight_ih_l0",
    "process.6.module.1.module.weight_hh_l0",
    "process.6.module.1.module.bias_ih_l0",
    "process.6.module.1.module.bias_hh_l0",
    "process.7.module.0.weight",
    "process.7.module.1.weight",
    "process.7.module.1.bias",
    "process.7.module.3.weight",
    "process.7.module.4.weight",
    "process.7.module.4.bias",
    "process.8.weight",
    "process.9.weight",
    "process.9.bias",
    "w_fc.weight",
    "w_fc.bias",
    "s_fc.weight",
    "s_fc.bias",
    "d_fc.weight",
    "d_fc.bias",
];

impl Lstm {
    fn load(path: &str, use_duration: bool) -> Result<Self> {
        let dev = neural::device();
        let tensors = neural::load_pth(path)?;
        let mut w = HashMap::new();
        let mut pretrain = HashMap::new();
        let mut names = Vec::new();
        for name in PARAM_NAMES {
            // `.pth` tensors load on CPU; move them to the compute device (GPU when built with
            // `neural-cuda`) so weights and inputs live together.
            let t = tensors
                .get(name)
                .unwrap_or_else(|| panic!("LSTM pretrain missing {name}"))
                .to_dtype(DType::F32)?
                .to_device(&dev)?;
            pretrain.insert(name.to_string(), t.clone());
            w.insert(name.to_string(), Var::from_tensor(&t)?);
            names.push(name.to_string());
        }
        let input_mean = tensors["input_mean"].to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let input_std = tensors["input_std"].to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let n_in = if use_duration { 6 } else { 5 };
        Ok(Self { w, names, pretrain, input_mean, input_std, use_duration, n_in, device: dev })
    }

    /// Reset all trainable vars to the pretrained (meta) weights — done before each split's
    /// finetune (script.py re-creates the model fresh per split).
    fn reset_to_pretrain(&self) -> Result<()> {
        for name in &self.names {
            self.w[name].set(&self.pretrain[name])?;
        }
        Ok(())
    }

    /// Build the `[B, L, n_in]` network input from a batch of items (plain Rust; these are input
    /// features, no gradient needed). Padded steps use raw `(Δt=0, dur=0, rating=0)` then the same
    /// transform (matching torch's `pad_sequence(0)` then `forward`); they sit past `seq_len-1` and
    /// the causal LSTM ignores them.
    fn build_input(&self, batch: &[&SeqItem], max_len: usize) -> Result<Tensor> {
        let b = batch.len();
        let mut data = Vec::with_capacity(b * max_len * self.n_in);
        for item in batch {
            for l in 0..max_len {
                // raw [delta, (duration), rating]; rating is always last in the seq vector.
                let (delta, dur, rating) = if l < item.seq.len() {
                    let s = &item.seq[l];
                    if self.use_duration {
                        (s[0], s[1], s[2])
                    } else {
                        (s[0], 0.0, s[1])
                    }
                } else {
                    (0.0, 0.0, 0.0)
                };
                let x_delay = ((1e-5f32 + delta).ln() - self.input_mean[0]) / self.input_std[0];
                data.push(x_delay);
                if self.use_duration {
                    let x_dur =
                        (dur.clamp(100.0, 60000.0).ln() - self.input_mean[1]) / self.input_std[1];
                    data.push(x_dur);
                }
                let ri = (rating.max(1.0) as usize - 1).min(3); // clamp min 1, one-hot index
                for c in 0..4 {
                    data.push(if c == ri { 1.0 } else { 0.0 });
                }
            }
        }
        Tensor::from_vec(data, (b, max_len, self.n_in), &self.device)
    }

    /// Core forward → retentions `[B]`. `detach` drops gradient tracking (prediction path).
    fn forward_impl(&self, batch: &[&SeqItem], detach: bool) -> Result<Tensor> {
        let g = |name: &str| -> Tensor {
            let t = self.w[name].as_tensor();
            if detach {
                t.detach()
            } else {
                t.clone()
            }
        };
        let b = batch.len();
        let max_len = batch.iter().map(|it| it.seq.len()).max().unwrap_or(1).max(1);
        let x = self.build_input(batch, max_len)?; // [B, L, n_in]

        // process[0,1,2]: Linear(n_in→20) → SiLU → LayerNorm(no bias)
        let h = linear(&x.reshape((b * max_len, self.n_in))?, &g("process.0.weight"), &g("process.0.bias"))?;
        let h = ops::silu(&h)?;
        let h = h.reshape((b, max_len, N_HIDDEN))?;
        let h = layernorm_no_bias(&h, &g("process.2.weight"), LN_EPS)?;

        // process[3,4]: Linear(20→20) → SiLU
        let h = linear(&h.reshape((b * max_len, N_HIDDEN))?, &g("process.3.weight"), &g("process.3.bias"))?;
        let h = ops::silu(&h)?;
        let mut h = h.reshape((b, max_len, N_HIDDEN))?;

        // process[5]: ResBlock( LN(no bias) → LSTM )
        let r = layernorm_no_bias(&h, &g("process.5.module.0.weight"), LN_EPS)?;
        let r = self.lstm_seq(
            &r,
            &g("process.5.module.1.module.weight_ih_l0"),
            &g("process.5.module.1.module.weight_hh_l0"),
            &g("process.5.module.1.module.bias_ih_l0"),
            &g("process.5.module.1.module.bias_hh_l0"),
        )?;
        h = (r + h)?;

        // process[6]: ResBlock( LN(WITH bias) → LSTM )
        let r = layernorm_with_bias(
            &h,
            &g("process.6.module.0.weight"),
            &g("process.6.module.0.bias"),
            LN_EPS,
        )?;
        let r = self.lstm_seq(
            &r,
            &g("process.6.module.1.module.weight_ih_l0"),
            &g("process.6.module.1.module.weight_hh_l0"),
            &g("process.6.module.1.module.bias_ih_l0"),
            &g("process.6.module.1.module.bias_hh_l0"),
        )?;
        h = (r + h)?;

        // process[7]: ResBlock( LN → Linear → SiLU → LN → Linear → SiLU )
        let r = layernorm_no_bias(&h, &g("process.7.module.0.weight"), LN_EPS)?;
        let r = linear(&r.reshape((b * max_len, N_HIDDEN))?, &g("process.7.module.1.weight"), &g("process.7.module.1.bias"))?;
        let r = ops::silu(&r)?;
        let r = r.reshape((b, max_len, N_HIDDEN))?;
        let r = layernorm_no_bias(&r, &g("process.7.module.3.weight"), LN_EPS)?;
        let r = linear(&r.reshape((b * max_len, N_HIDDEN))?, &g("process.7.module.4.weight"), &g("process.7.module.4.bias"))?;
        let r = ops::silu(&r)?;
        let r = r.reshape((b, max_len, N_HIDDEN))?;
        h = (r + h)?;

        // process[8,9,10]: LayerNorm(no bias) → Linear(20→20) → SiLU
        let h = layernorm_no_bias(&h, &g("process.8.weight"), LN_EPS)?;
        let h = linear(&h.reshape((b * max_len, N_HIDDEN))?, &g("process.9.weight"), &g("process.9.bias"))?;
        let h = ops::silu(&h)?;
        let x_lnh = h.reshape((b, max_len, N_HIDDEN))?; // [B, L, 20]

        // gather the hidden state at each row's last real step (seq_len-1)
        let idx: Vec<u32> = batch.iter().map(|it| (it.seq.len() - 1) as u32).collect();
        let last = gather_last(&x_lnh, &idx)?; // [B, 20]

        // heads
        let w_lnh = ops::softmax(&linear(&last, &g("w_fc.weight"), &g("w_fc.bias"))?, 1)?; // [B,3]
        let s_lnh = linear(&last, &g("s_fc.weight"), &g("s_fc.bias"))?.clamp(-25.0, 25.0)?.exp()?;
        let d_lnh = linear(&last, &g("d_fc.weight"), &g("d_fc.bias"))?.clamp(-25.0, 25.0)?.exp()?;

        // forgetting curve: (1-1e-7) * Σ_c w_c * (1 + Δt/(1e-7+s_c))^(-d_c)
        let delta: Vec<f32> = batch.iter().map(|it| it.delta_t).collect();
        let delta = Tensor::from_vec(delta, (b, 1), &self.device)?; // [B,1]
        let inner = delta
            .broadcast_div(&s_lnh.affine(1.0, 1e-7)?)?
            .affine(1.0, 1.0)?; // 1 + Δt/(1e-7+s)
        let pow = (d_lnh.neg()? * inner.log()?)?.exp()?; // (inner)^(-d)
        let mixed = (w_lnh * pow)?.sum(1)?; // [B]
        mixed.affine(1.0 - 1e-7, 0.0)
    }

    /// Run the LSTM over the time dim of `x` `[B, L, H]` → outputs `[B, L, H]`.
    fn lstm_seq(&self, x: &Tensor, wih: &Tensor, whh: &Tensor, bih: &Tensor, bhh: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let mut h = Tensor::zeros((b, N_HIDDEN), DType::F32, &self.device)?;
        let mut c = Tensor::zeros((b, N_HIDDEN), DType::F32, &self.device)?;
        let mut outs = Vec::with_capacity(l);
        for t in 0..l {
            let xt = x.i((.., t, ..))?.contiguous()?;
            let (h_new, c_new) = lstm_cell(&xt, &h, &c, wih, whh, bih, bhh)?;
            h = h_new;
            c = c_new;
            outs.push(h.clone());
        }
        Tensor::stack(&outs, 1)
    }

    /// Predict (no grad) over `items`, batched in original order (no seq_len drop). Returns
    /// retentions in row order.
    fn predict(&self, items: &[SeqItem]) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(items.len());
        for chunk in items.chunks(PREDICT_BATCH) {
            let batch: Vec<&SeqItem> = chunk.iter().collect();
            let ret = self.forward_impl(&batch, true)?;
            for v in ret.to_vec1::<f32>()? {
                out.push(v as f64);
            }
        }
        Ok(out)
    }
}

impl NeuralModel for Lstm {
    fn vars(&self) -> Vec<Var> {
        self.names.iter().map(|n| self.w[n].clone()).collect()
    }
    fn forward(&self, batch: &[&SeqItem]) -> Result<Tensor> {
        self.forward_impl(batch, false)
    }
}

/// LSTM finetune hyperparameters (reptile_trainer.py DEFAULT_FINETUNE_PARAMS; `BATCH_SIZE=16384`,
/// inner Adam betas `(0.0, 0.999)`).
fn lstm_finetune_params() -> FinetuneParams {
    FinetuneParams {
        lr_start_raw: 0.0019622,
        lr_middle_raw: 0.006455344,
        lr_end_raw: 0.0034213,
        warmup_steps: 5,
        batch_size_exp: 1.2103,
        clip_norm: 7050.0,
        reg_scale: 0.000244,
        inner_steps: 20,
        recency_weight: 6.49,
        recency_degree: 2.4758,
        weight_decay: 0.04855,
        inner_adam_beta1: 0.0,
        inner_adam_beta2: 0.999,
        batch_size: 16384,
    }
}

/// Build training/prediction items from dataset rows: each row's prior reviews as
/// `[delta_t, (duration,) rating]` sequence, plus the current `delta_t` (forgetting-curve input)
/// and label.
fn build_items(ds: &Dataset, rows: &[Row], use_duration: bool) -> Vec<SeqItem> {
    rows.iter()
        .map(|r| {
            let dts = ds.prior_dt_active(r);
            let rs = ds.prior_ratings(r);
            let seq: Vec<Vec<f32>> = if use_duration {
                let durs = ds.prior_durations(r);
                dts.iter()
                    .zip(rs.iter())
                    .zip(durs.iter())
                    .map(|((&dt, &rt), &du)| vec![dt as f32, du as f32, rt as f32])
                    .collect()
            } else {
                dts.iter()
                    .zip(rs.iter())
                    .map(|(&dt, &rt)| vec![dt as f32, rt as f32])
                    .collect()
            };
            SeqItem { seq, delta_t: r.delta_t as f32, y: r.y as f32 }
        })
        .collect()
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let path = format!("pretrain/{}_pretrain.pth", cfg.evaluation_file_name());
    let use_duration = cfg.lstm_use_duration;
    let model = Lstm::load(&path, use_duration).unwrap_or_else(|e| panic!("LSTM load {path}: {e}"));
    let params = lstm_finetune_params();
    let rows = &ds.rows;

    let mut eval_rows: Vec<Row> = Vec::new();
    let mut p: Vec<f64> = Vec::new();

    // Build (train_rows, test_rows) per split: equalize splits if set, else TimeSeriesSplit.
    let folds: Vec<(Vec<Row>, Vec<Row>)> = if let Some(eq) = &ds.equalize_splits {
        eq.iter()
            .map(|sp| {
                let train: Vec<Row> = rows[..sp.train_end].to_vec();
                let test: Vec<Row> = sp.test_idx.iter().map(|&i| rows[i].clone()).collect();
                (train, test)
            })
            .collect()
    } else {
        time_series_split(rows.len(), cfg.n_splits)
            .into_iter()
            .map(|s| {
                let train: Vec<Row> = rows[..s.test_start].to_vec();
                let test: Vec<Row> = rows[s.test_start..s.test_end].to_vec();
                (train, test)
            })
            .collect()
    };

    for (train, test) in folds {
        model.reset_to_pretrain().expect("reset");
        if !cfg.default_params {
            let train_items = build_items(ds, &train, use_duration);
            finetune(&model, &train_items, &params).expect("finetune");
        }
        let test_items = build_items(ds, &test, use_duration);
        let preds = model.predict(&test_items).expect("predict");
        for (i, pr) in preds.into_iter().enumerate() {
            eval_rows.push(test[i].clone());
            p.push(pr);
        }
    }

    let _ = (N_CURVES, _bce);
    ModelOutput { eval_rows, p, params: Params::None }
}
