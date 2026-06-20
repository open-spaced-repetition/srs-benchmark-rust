//! Model processing — one model per file, mirroring the Python `models/` layout.
//!
//! Each model exposes `process(ds, cfg) -> ModelOutput`, returning the evaluation rows
//! (concatenation of the per-split test folds, in split order), the matching predictions
//! `p`, and any trained parameters to record.

pub mod act_r;
pub mod anki;
pub mod avg;
pub mod dash;
pub mod dash_act_r;
pub mod dash_act_r_grad;
pub mod ebisu;
pub mod fsrs_init;
#[cfg(feature = "fsrs-rs")]
pub mod fsrs_rs;
pub mod fsrs_v1;
pub mod fsrs_v1_grad;
pub mod fsrs_v2;
pub mod fsrs_v2_grad;
pub mod fsrs_v3;
pub mod fsrs_v3_grad;
pub mod fsrs_v4;
pub mod fsrs_v4_grad;
pub mod fsrs_v4dot5;
pub mod fsrs_v4dot5_grad;
pub mod fsrs_v5;
pub mod fsrs_v5_grad;
pub mod fsrs_v6;
pub mod fsrs_v6_grad;
pub mod fsrs_v6_one_step;
pub mod fsrs_v7;
pub mod fsrs_v7_grad;
pub mod fsrs_v7_simd;
#[cfg(feature = "neural")]
pub mod gru;
pub mod hlr;
#[cfg(feature = "neural")]
pub mod lstm;
pub mod logistic_regression;
pub mod moving_avg;
pub mod rmse_bins_exploit;
pub mod sm2;
pub mod sm2_trainable;
pub mod sm2_trainable_grad;

use crate::autodiff::round_scalar as r;
use crate::eval::Params;
use crate::features::Row;

/// Result of running a model over one user's dataset.
pub struct ModelOutput {
    pub eval_rows: Vec<Row>,
    pub p: Vec<f64>,
    pub params: Params,
}

/// Recency weights `0.25 + 0.75*x^3`, x = linspace(0,1,N) (`_apply_recency_weighting`).
/// Shared by the Adam-trained models.
pub(crate) fn recency_weights(n: usize, recency: bool) -> Vec<f64> {
    if !recency {
        return vec![1.0; n];
    }
    (0..n)
        .map(|k| {
            let x = if n <= 1 { 0.0 } else { r(k as f64 / (n as f64 - 1.0)) };
            r(0.25 + 0.75 * r(r(x * x) * x))
        })
        .collect()
}

/// FSRS-7's own recency weights (`_apply_recency_weighting`, model_name == "FSRS-7"):
/// `0.0667 + 0.9333 * (k/n)^11.25`, k 0-based, denominator `n` (NOT n-1).
pub(crate) fn recency_weights_fsrs7(n: usize, recency: bool) -> Vec<f64> {
    if !recency {
        return vec![1.0; n];
    }
    let denom = n.max(1) as f64;
    (0..n)
        .map(|k| r(0.0667 + 0.9333 * r((r(k as f64 / denom)).powf(11.25))))
        .collect()
}
