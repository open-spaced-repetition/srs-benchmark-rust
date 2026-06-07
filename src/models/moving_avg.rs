//! MOVING-AVG — `model_processors.moving_avg`. A sequential logistic update over all rows
//! in review_th order; predictions recorded only from the first test index onward.

use super::ModelOutput;
use crate::config::Config;
use crate::eval::Params;
use crate::features::Dataset;
use crate::split::time_series_split;

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let first = splits[0].test_start;
    let mut x = 1.2f64;
    let w = 0.3f64;
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let y_pred = 1.0 / ((-x).exp() + 1.0);
        if i >= first {
            eval_rows.push(r.clone());
            p.push(y_pred);
        }
        if r.y == 1 {
            x += w / (x.exp() + 1.0);
        } else {
            x -= w * x.exp() / (x.exp() + 1.0);
        }
    }
    ModelOutput {
        eval_rows,
        p,
        params: Params::None,
    }
}
