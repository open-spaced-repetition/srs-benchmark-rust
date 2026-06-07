//! Model processing. Each `process_*` returns the evaluation rows (concatenation of the
//! per-split test folds, in split order), the matching predictions `p`, and any trained
//! parameters to record.

use crate::config::Config;
use crate::eval::Params;
use crate::features::Row;
use crate::split::time_series_split;

/// Result of running a model over one user's dataset.
pub struct ModelOutput {
    pub eval_rows: Vec<Row>,
    pub p: Vec<f64>,
    pub params: Params,
}

/// AVG baseline (`model_processors.baseline`): each split predicts the train fold's mean y.
pub fn process_avg(rows: &[Row], cfg: &Config) -> ModelOutput {
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    for s in splits {
        let train = &rows[..s.test_start];
        let avg_p = train.iter().map(|r| r.y as f64).sum::<f64>() / train.len() as f64;
        for r in &rows[s.test_start..s.test_end] {
            eval_rows.push(r.clone());
            p.push(avg_p);
        }
    }
    ModelOutput {
        eval_rows,
        p,
        params: Params::None,
    }
}
