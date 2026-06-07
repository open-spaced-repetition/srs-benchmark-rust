//! Model processing — one model per file, mirroring the Python `models/` layout.
//!
//! Each model exposes `process(ds, cfg) -> ModelOutput`, returning the evaluation rows
//! (concatenation of the per-split test folds, in split order), the matching predictions
//! `p`, and any trained parameters to record.

pub mod avg;
pub mod dash;
pub mod fsrs_init;
pub mod fsrs_v1;
pub mod fsrs_v2;
pub mod fsrs_v3;
pub mod fsrs_v4;
pub mod fsrs_v4dot5;
pub mod fsrs_v5;
pub mod fsrs_v6;
pub mod hlr;
pub mod moving_avg;
pub mod rmse_bins_exploit;
pub mod sm2;

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
            let x = if n <= 1 { 0.0 } else { k as f64 / (n as f64 - 1.0) };
            0.25 + 0.75 * x * x * x
        })
        .collect()
}
