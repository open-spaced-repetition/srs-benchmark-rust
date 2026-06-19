//! GRU — `models/gru.py`. A pretrained meta-model (loaded from `pretrain/<name>_pretrain.pth`)
//! is fine-tuned per user/split via the Reptile `finetune` (`reptile_trainer_gru.py`), then
//! predicts. Built on the `candle` ML framework (gated behind the `neural` cargo feature).
//!
//! Architecture (matches gru.py): a `process` Sequential
//!   Linear(5→7) → SiLU → LayerNorm(7,no bias) → GRU(7→7) → LayerNorm → Linear(7→7) → SiLU →
//!   LayerNorm
//! then three heads w_fc/s_fc/d_fc (Linear 7→2). Rating is one-hot(4)-expanded and the single
//! delta feature is `log(1e-5+Δt)` normalised by the loaded `input_mean`/`input_std`. Output is
//! a 2-curve mixture forgetting curve. The eval row-set equals every other `--short --secs`
//! model (shared base pipeline), so `size` is exact by construction.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::ops;

use super::ModelOutput;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::neural::{
    self, bce as _bce, finetune, gather_last, gru_cell, layernorm_no_bias, linear, FinetuneParams,
    NeuralModel, SeqItem,
};
use crate::split::time_series_split;

const N_HIDDEN: usize = 7;
const N_CURVES: usize = 2;
const LN_EPS: f64 = 1e-5;
const PREDICT_BATCH: usize = 8192;

/// Trainable GRU model. Weights are stored as candle `Var`s keyed by their torch state_dict
/// name so they load directly from the `.pth`.
pub struct Gru {
    w: HashMap<String, Var>,
    names: Vec<String>,
    pretrain: HashMap<String, Tensor>,
    input_mean: f32,
    input_std: f32,
    device: Device,
}

/// The 17 trainable parameter names, in a stable order (the GRU `.pth` layout).
const PARAM_NAMES: [&str; 17] = [
    "process.0.weight",
    "process.0.bias",
    "process.2.weight",
    "process.3.module.weight_ih_l0",
    "process.3.module.weight_hh_l0",
    "process.3.module.bias_ih_l0",
    "process.3.module.bias_hh_l0",
    "process.4.weight",
    "process.5.weight",
    "process.5.bias",
    "process.7.weight",
    "w_fc.weight",
    "w_fc.bias",
    "s_fc.weight",
    "s_fc.bias",
    "d_fc.weight",
    "d_fc.bias",
];

impl Gru {
    fn load(path: &str) -> Result<Self> {
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
                .unwrap_or_else(|| panic!("GRU pretrain missing {name}"))
                .to_dtype(DType::F32)?
                .to_device(&dev)?;
            pretrain.insert(name.to_string(), t.clone());
            w.insert(name.to_string(), Var::from_tensor(&t)?);
            names.push(name.to_string());
        }
        let input_mean = tensors["input_mean"].to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
        let input_std = tensors["input_std"].to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
        Ok(Self { w, names, pretrain, input_mean, input_std, device: dev })
    }

    /// Reset all trainable vars to the pretrained (meta) weights — done before each split's
    /// finetune (script.py re-creates the model fresh per split).
    fn reset_to_pretrain(&self) -> Result<()> {
        for name in &self.names {
            self.w[name].set(&self.pretrain[name])?;
        }
        Ok(())
    }

    /// Build the `[B, L, 5]` network input from a batch of items (plain Rust; no gradient needed
    /// — these are input features). Padded steps use `(Δt=0, rating=0)`, matching torch's
    /// `pad_sequence(padding_value=0)`; they sit past `seq_len-1` and the causal GRU ignores them.
    fn build_input(&self, batch: &[&SeqItem], max_len: usize) -> Result<Tensor> {
        let b = batch.len();
        let mut data = Vec::with_capacity(b * max_len * 5);
        for item in batch {
            for l in 0..max_len {
                let (delta, rating) = if l < item.seq.len() {
                    (item.seq[l][0], item.seq[l][1])
                } else {
                    (0.0, 0.0)
                };
                let x_main = ((1e-5f32 + delta).ln() - self.input_mean) / self.input_std;
                data.push(x_main);
                let ri = (rating.max(1.0) as usize - 1).min(3); // clamp min 1, one-hot index
                for c in 0..4 {
                    data.push(if c == ri { 1.0 } else { 0.0 });
                }
            }
        }
        Tensor::from_vec(data, (b, max_len, 5), &self.device)
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
        let x = self.build_input(batch, max_len)?; // [B, L, 5]

        // process[0,1,2]: Linear(5→7) → SiLU → LayerNorm(no bias)
        let h = linear(&x.reshape((b * max_len, 5))?, &g("process.0.weight"), &g("process.0.bias"))?;
        let h = ops::silu(&h)?;
        let h = h.reshape((b, max_len, N_HIDDEN))?;
        let h = layernorm_no_bias(&h, &g("process.2.weight"), LN_EPS)?;

        // process[3]: GRU over the sequence
        let h = self.gru_seq(
            &h,
            &g("process.3.module.weight_ih_l0"),
            &g("process.3.module.weight_hh_l0"),
            &g("process.3.module.bias_ih_l0"),
            &g("process.3.module.bias_hh_l0"),
        )?;

        // process[4,5,6,7]: LayerNorm → Linear(7→7) → SiLU → LayerNorm
        let h = layernorm_no_bias(&h, &g("process.4.weight"), LN_EPS)?;
        let h = linear(&h.reshape((b * max_len, N_HIDDEN))?, &g("process.5.weight"), &g("process.5.bias"))?;
        let h = ops::silu(&h)?;
        let h = h.reshape((b, max_len, N_HIDDEN))?;
        let x_lnh = layernorm_no_bias(&h, &g("process.7.weight"), LN_EPS)?; // [B, L, 7]

        // gather the hidden state at each row's last real step (seq_len-1)
        let idx: Vec<u32> = batch.iter().map(|it| (it.seq.len() - 1) as u32).collect();
        let last = gather_last(&x_lnh, &idx)?; // [B, 7]

        // heads
        let w_lnh = ops::softmax(&linear(&last, &g("w_fc.weight"), &g("w_fc.bias"))?, 1)?; // [B,2]
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

    /// Run the GRU over the time dim of `x` `[B, L, H]` → outputs `[B, L, H]`.
    fn gru_seq(&self, x: &Tensor, wih: &Tensor, whh: &Tensor, bih: &Tensor, bhh: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let mut h = Tensor::zeros((b, N_HIDDEN), DType::F32, &self.device)?;
        let mut outs = Vec::with_capacity(l);
        for t in 0..l {
            let xt = x.i((.., t, ..))?.contiguous()?;
            h = gru_cell(&xt, &h, wih, whh, bih, bhh)?;
            outs.push(h.clone());
        }
        Tensor::stack(&outs, 1)
    }

    /// Predict (no grad) over `items`, batched in original order (Collection.batch_predict:
    /// batch_size=8192, sort_by_length=False, no seq_len drop). Returns retentions in row order.
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

impl NeuralModel for Gru {
    fn vars(&self) -> Vec<Var> {
        self.names.iter().map(|n| self.w[n].clone()).collect()
    }
    fn forward(&self, batch: &[&SeqItem]) -> Result<Tensor> {
        self.forward_impl(batch, false)
    }
}

use candle_core::IndexOp;

/// GRU finetune hyperparameters (reptile_trainer_gru.py DEFAULT_FINETUNE_PARAMS).
fn gru_finetune_params() -> FinetuneParams {
    FinetuneParams {
        lr_start_raw: 0.002470,
        lr_middle_raw: 0.005842,
        lr_end_raw: 0.001227,
        warmup_steps: 7,
        batch_size_exp: 1.155,
        clip_norm: 270.9,
        reg_scale: 0.0007204,
        inner_steps: 14,
        recency_weight: 5.242,
        recency_degree: 2.518,
        weight_decay: 0.03116,
        inner_adam_beta1: 0.4077,
        inner_adam_beta2: 0.9570,
        batch_size: 8192,
    }
}

/// Build training/prediction items from dataset rows: each row's prior reviews as
/// `[delta_t, rating]` sequence, plus the current `delta_t` (forgetting-curve input) and label.
fn build_items(ds: &Dataset, rows: &[Row]) -> Vec<SeqItem> {
    rows.iter()
        .map(|r| {
            let dts = ds.prior_dt_active(r);
            let rs = ds.prior_ratings(r);
            let seq: Vec<Vec<f32>> = dts
                .iter()
                .zip(rs.iter())
                .map(|(&dt, &rt)| vec![dt as f32, rt as f32])
                .collect();
            SeqItem { seq, delta_t: r.delta_t as f32, y: r.y as f32 }
        })
        .collect()
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let path = format!("pretrain/{}_pretrain.pth", cfg.evaluation_file_name());
    let model = Gru::load(&path).unwrap_or_else(|e| panic!("GRU load {path}: {e}"));
    let params = gru_finetune_params();
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
            let train_items = build_items(ds, &train);
            finetune(&model, &train_items, &params).expect("finetune");
        }
        let test_items = build_items(ds, &test);
        let preds = model.predict(&test_items).expect("predict");
        for (i, pr) in preds.into_iter().enumerate() {
            eval_rows.push(test[i].clone());
            p.push(pr);
        }
    }

    let _ = (N_CURVES, _bce);
    ModelOutput { eval_rows, p, params: Params::None }
}
