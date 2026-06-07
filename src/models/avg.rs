//! AVG baseline — `model_processors.baseline`. Predicts each split's train-fold mean y.

use super::ModelOutput;
use crate::config::Config;
use crate::eval::Params;
use crate::features::Dataset;
use crate::split::time_series_split;

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
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
