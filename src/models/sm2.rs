//! SM-2 (untrainable) — `models/sm2.py` + `model_processors.process_untrainable`.
//! Predicts `0.9^(delta_t / sm2_ivl(prior ratings))` per test row.

use super::ModelOutput;
use crate::config::Config;
use crate::eval::Params;
use crate::features::Dataset;
use crate::metrics::py_round_half_even;
use crate::split::time_series_split;

/// SM-2 interval from a prior-rating sequence (`models/sm2.py::sm2`).
fn sm2_ivl(prior: &[i64], s_max: f64) -> f64 {
    let mut ivl = 0.0f64;
    let mut ef = 2.5f64;
    let mut reps = 0i64;
    for &r in prior {
        let rating = r + 1;
        if rating > 2 {
            if reps == 0 {
                ivl = 1.0;
                reps = 1;
            } else if reps == 1 {
                ivl = 6.0;
                reps = 2;
            } else {
                ivl *= ef;
                reps += 1;
            }
        } else {
            ivl = 1.0;
            reps = 0;
        }
        let q = 5.0 - rating as f64;
        ef = 1.3f64.max(ef + (0.1 - q * (0.08 + q * 0.02)));
        ivl = py_round_half_even(ivl + 0.01).max(1.0).min(s_max);
    }
    ivl
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let ln09 = 0.9f64.ln();
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    for s in splits {
        for r in &rows[s.test_start..s.test_end] {
            let ivl = sm2_ivl(ds.prior_ratings(r), cfg.s_max);
            eval_rows.push(r.clone());
            p.push((ln09 * r.delta_t / ivl).exp());
        }
    }
    ModelOutput {
        eval_rows,
        p,
        params: Params::None,
    }
}
