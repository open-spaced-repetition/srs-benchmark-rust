//! FSRS-rs — `models/fsrs_rs.py`. Delegates training to the real `fsrs` crate (the exact git
//! rev that `fsrs-rs-python` 0.8.2 wraps), so the trained weights match the upstream reference
//! bit-for-bit. Prediction reuses stock FSRS-6 (`fsrs_v6::predict`), exactly as the Python does
//! via `fsrs_optimizer.Collection`, so the eval row-set — and `size` — equals plain FSRS-6.
//!
//! Gated behind the `fsrs-rs` cargo feature (the `fsrs` crate pulls in the heavy burn ML
//! framework); see `Cargo.toml`.

use fsrs::{ComputeParametersInput, FSRSItem, FSRSReview, FSRS};

use super::{fsrs_v6, ModelOutput};
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;

/// One `FSRSItem` per training review: its full `(delta_t, rating)` history (priors + current),
/// `delta_t = max(0, int)`. Port of `convert_to_items`. Items are produced in `review_th` order
/// (`train` is already sorted). A review with no priors is skipped — Python's `convert_to_items`
/// errors on the empty `t_history` string, so such reviews never occur for emitted users.
fn build_items(ds: &Dataset, train: &[Row]) -> Vec<FSRSItem> {
    let mut items = Vec::with_capacity(train.len());
    for r in train {
        let prior_dt = ds.prior_dt_active(r);
        let prior_r = ds.prior_ratings(r);
        if prior_dt.is_empty() {
            continue;
        }
        let mut reviews = Vec::with_capacity(prior_dt.len() + 1);
        for k in 0..prior_dt.len() {
            reviews.push(FSRSReview {
                rating: prior_r[k] as u32,
                delta_t: prior_dt[k].max(0.0) as u32,
            });
        }
        reviews.push(FSRSReview {
            rating: r.rating as u32,
            delta_t: r.delta_t.max(0.0) as u32,
        });
        items.push(FSRSItem { reviews });
    }
    items
}

/// Train via `fsrs::FSRS::benchmark` (the same call `fsrs-rs-python` makes), then round to 4 dp
/// as the Python does before prediction.
fn train_weights(ds: &Dataset, train: &[Row]) -> Vec<f64> {
    let items = build_items(ds, train);
    let model = FSRS::new(Some(&[])).expect("FSRS::new(empty params)");
    let w: Vec<f32> = model.benchmark(ComputeParametersInput {
        train_set: items,
        progress: None,
        enable_short_term: true,
        num_relearning_steps: None,
    });
    w.iter().map(|&x| ((x as f64) * 1e4).round() / 1e4).collect()
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = fsrs_v6::INIT_W.to_vec();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            fsrs_v6::INIT_W.to_vec()
        } else {
            train_weights(ds, train)
        };
        let test = &rows[s.test_start..s.test_end];
        for (i, pr) in fsrs_v6::predict(ds, test, &w, cfg).into_iter().enumerate() {
            eval_rows.push(test[i].clone());
            p.push(pr);
        }
        last_w = w;
    }

    ModelOutput {
        eval_rows,
        p,
        params: Params::Partitioned(vec![("0".to_string(), last_w)]),
    }
}
