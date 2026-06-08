//! LogisticRegression — `models/logistic_regression.py`. A linear model over 34 hand-crafted
//! features with standardized coefficients (`coef = coef_res·std + mean`), trained with AdamW.
//!
//! Features (`logistic_regression.create_features`) are computed once over each card's
//! surviving rows (cumulative same/non-same-day pass/fail counts, log1p transforms, one-hot
//! prior-rating, transformed elapsed/cumulative times). They don't depend on the params, so
//! we precompute them; the model is `sigmoid(x·coef)`. The loss is convex, so the exact
//! (unseeded) shuffle order doesn't matter — it converges to the same optimum.
//!
//! Only the `--secs` feature path is implemented (the references are `-short-secs-recency`);
//! the `--equalize_test_with_non_secs` variant is not yet supported.

use super::ModelOutput;
use crate::config::Config;
use crate::eval::Params;
use crate::features::Dataset;
use crate::split::time_series_split;
use crate::train::Mt19937;

const NF: usize = 34;

const MEAN: [f64; NF] = [
    -0.9012, -0.6941, -0.6258, -0.4678, 0.3926, 0.1316, 0.3508, -0.3259, -0.2125, -0.1402, 1.2775,
    0.0669, 0.3914, 0.5828, 0.9292, 1.1583, 1.4103, 0.9388, 0.9164, 0.9797, -0.6411, 0.4951,
    0.1038, -0.1564, 0.4332, 0.3287, 0.3429, 0.4068, -0.1779, -0.0585, -0.1062, -0.3521, -0.1613,
    -0.1758,
];
const STD: [f64; NF] = [
    0.2869, 0.2811, 0.2910, 0.2166, 0.1740, 0.1612, 0.1428, 0.1352, 0.1284, 0.0331, 0.4013, 0.2065,
    0.2022, 0.1670, 0.2454, 0.2954, 0.2896, 0.3473, 0.5685, 0.2795, 0.1298, 0.0467, 0.1200, 0.1317,
    0.0785, 0.1190, 0.0854, 0.1634, 0.1892, 0.1052, 0.0750, 0.1913, 0.1372, 0.0783,
];

#[inline]
fn transform_elapsed(x: f64) -> f64 {
    ((x + 1e-5).ln() + 1.3) / 5.0
}
#[inline]
fn log1p(x: f64) -> f64 {
    (1.0 + x).ln()
}

/// Compute the 34 features for every row, aligned with `ds.rows`. Mirrors
/// `logistic_regression.create_features` (cumulative state over each card's surviving rows).
fn compute_features(ds: &Dataset) -> Vec<[f64; NF]> {
    let rows = &ds.rows;
    let n = rows.len();
    let mut feats = vec![[0.0f64; NF]; n];

    // Group row indices by card, in review_th (= pos) order. ds.rows is review_th-sorted, so
    // appending per card preserves within-card order.
    use std::collections::HashMap;
    let mut by_card: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        by_card.entry(r.card_idx).or_default().push(i);
    }

    for idxs in by_card.values() {
        // Per-card running state (cumulative, inclusive of current row).
        let (mut n_sd_fail, mut n_nsd_fail, mut n_sd_pass, mut n_nsd_pass) = (0.0, 0.0, 0.0, 0.0);
        let (mut n_pass, mut n_same_day, mut n_non_same_day) = (0.0, 0.0, 0.0);
        let mut times = 0.0f64; // cumsum of prior delta_t_secs
        let mut running_max_flt = f64::NEG_INFINITY;

        // `feature_rating` is set BEFORE the delta_t>0 filter, so it shifts over the FULL
        // per-card sequence: feature_rating(row) = card.ratings[pos-1] (the dropped new-card
        // review counts), 0 for pos==0. `first_rating` = feature_rating of the card's first
        // surviving row = card.ratings[first_pos-1]. (feat_elapsed below uses the surviving
        // prior, since it's computed after the filter.)
        let card_ratings = &ds.cards[rows[idxs[0]].card_idx as usize].ratings;
        let first_pos = rows[idxs[0]].pos as usize;
        let first_r = if first_pos > 0 { card_ratings[first_pos - 1] } else { 0 };
        let mut first_oh = [0.0f64; 3];
        if first_r > 1 {
            first_oh[((first_r - 2).clamp(0, 2)) as usize] = 1.0;
        }
        let fr_gt1 = if first_r > 1 { 1.0 } else { 0.0 };

        for (j, &i) in idxs.iter().enumerate() {
            let row = &rows[i];
            let is_first = j == 0;
            let not_first = if is_first { 0.0 } else { 1.0 };
            // feature_rating = full-sequence prior rating; feat_elapsed = surviving prior.
            let pos = row.pos as usize;
            let r = if pos > 0 { card_ratings[pos - 1] } else { 0 };
            let prev = if is_first { None } else { Some(&rows[idxs[j - 1]]) };
            let feat_elapsed_real = prev.map(|p| p.delta_t).unwrap_or(0.0); // prior surviving delta_t_secs
            let feat_elapsed_int = prev.map(|p| p.elapsed_days.max(0)).unwrap_or(0); // prior surviving delta_t_int

            let same_day = if feat_elapsed_int == 0 { 1.0 } else { 0.0 };
            let success = if r > 1 { 1.0 } else { 0.0 };
            let fail = if r == 1 { 1.0 } else { 0.0 };
            let is_hard = if r == 2 { 1.0 } else { 0.0 };
            let better_than_hard = if r > 2 { 1.0 } else { 0.0 };

            let label_int = row.elapsed_days.max(0); // current delta_t_int
            let label_real = row.delta_t; // current delta_t_secs
            let label_is_same_day = if label_int == 0 { 1.0 } else { 0.0 };

            // rating one-hot (classes 0,1,2 = ratings 2,3,4), zeroed when r<=1.
            let mut rating_oh = [0.0f64; 3];
            if r > 1 {
                let c = ((r - 2).clamp(0, 2)) as usize;
                rating_oh[c] = 1.0;
            }

            // Update cumulative counts (inclusive of current row).
            n_sd_fail += same_day * not_first * fail;
            n_nsd_fail += (1.0 - same_day) * not_first * fail;
            n_sd_pass += same_day * not_first * success;
            n_nsd_pass += (1.0 - same_day) * not_first * success;
            n_pass += success;
            n_same_day += same_day;
            n_non_same_day += 1.0 - same_day;
            let has_passed = if n_pass > 0.0 { 1.0 } else { 0.0 };

            times += feat_elapsed_real;
            let flt = if is_first || success == 0.0 { times } else { 0.0 };
            running_max_flt = running_max_flt.max(flt);
            let time_since_lapse = times - running_max_flt;

            let t_elapsed_real = transform_elapsed(feat_elapsed_real);
            let t_label_real = transform_elapsed(label_real);
            let t_time_since_lapse = transform_elapsed(time_since_lapse);
            let t_times = transform_elapsed(times);
            let v = t_label_real;

            // deg1 block (×v), 10 features.
            let deg1 = [
                1.0,
                rating_oh[0],
                rating_oh[1],
                rating_oh[2],
                is_hard * (is_first as i64 as f64),
                better_than_hard * (is_first as i64 as f64),
                log1p(n_sd_fail),
                log1p(n_nsd_pass * (1.0 - label_is_same_day)),
                log1p(n_nsd_fail * (1.0 - label_is_same_day)),
                log1p(n_sd_pass),
            ];
            // deg0 block, 24 features. first_rating_onehot terms are always 0 (first_r=0).
            let deg0 = [
                1.0,
                rating_oh[0],
                rating_oh[1],
                rating_oh[2],
                first_oh[0],
                first_oh[1],
                first_oh[2],
                first_oh[0] * (is_first as i64 as f64),
                first_oh[1] * (is_first as i64 as f64),
                first_oh[2] * (is_first as i64 as f64),
                log1p(n_sd_fail),
                log1p(n_nsd_pass * (1.0 - label_is_same_day)),
                t_elapsed_real,
                log1p(n_nsd_fail),
                log1p(n_nsd_pass),
                log1p(n_sd_pass * label_is_same_day),
                has_passed,
                t_time_since_lapse,
                label_is_same_day,
                t_times,
                fr_gt1 * log1p(n_sd_fail),
                fr_gt1 * log1p(n_nsd_fail),
                fr_gt1 * log1p(n_same_day),
                fr_gt1 * log1p(n_non_same_day),
            ];

            let f = &mut feats[i];
            for k in 0..10 {
                f[k] = deg1[k] * v;
            }
            for k in 0..24 {
                f[10 + k] = deg0[k];
            }
        }
    }
    feats
}

/// AdamW (decoupled weight decay) training of `coef_res`, returning the final coefficients.
/// Mirrors `LogisticRegression.optimize`: lr=0.2, betas (0,0.85), eps 1e-8, wd 0.3, 10 epochs,
/// batch 2048, recency weights `0.1 + 0.9·x⁴`, CosineAnnealingLR(eta_min=0). Convex loss, so
/// the (unseeded) shuffle order is immaterial — we visit batches in order.
fn train(feats: &[[f64; NF]], y: &[f64]) -> [f64; NF] {
    let b = feats.len();
    if b == 0 {
        return MEAN; // coef_res = 0 -> coef = mean
    }
    let (lr0, wd, b1, b2, eps) = (0.2f64, 0.3f64, 0.0f64, 0.85f64, 1e-8f64);
    let n_epoch = 10usize;
    let bs = 2048usize;
    // recency weights 0.1 + 0.9 x^4
    let weights: Vec<f64> = (0..b)
        .map(|i| {
            let x = if b <= 1 { 0.0 } else { i as f64 / (b as f64 - 1.0) };
            0.1 + 0.9 * x.powi(4)
        })
        .collect();

    let mut coef_res = [0.0f64; NF];
    let mut m = [0.0f64; NF];
    let mut v = [0.0f64; NF];
    let steps_per_epoch = b.div_ceil(bs);
    let t_max = (n_epoch * steps_per_epoch) as f64;
    let mut step = 0.0f64;
    let mut t = 0.0f64; // adam timestep
    // Shuffle the batch order each epoch like torch's `randperm(B)` (convex loss + finite
    // epochs ⇒ order matters). torch's RNG is unseeded; a fixed deterministic perm lands at
    // the same optimum within tolerance.
    let mut gen = Mt19937::new(2023);

    for _ in 0..n_epoch {
        let perm = gen.randperm(b);
        let mut start = 0;
        while start < b {
            let end = (start + bs).min(b);
            // coefficients = coef_res*std + mean
            let mut coef = [0.0f64; NF];
            for k in 0..NF {
                coef[k] = coef_res[k] * STD[k] + MEAN[k];
            }
            // gradient of sum(weight * BCE_with_logits(x·coef, y)) w.r.t. coef_res.
            // dL/dlogit = weight*(sigmoid(logit) - y); dlogit/dcoef_k = x_k; dcoef_k/dcoef_res_k = std_k.
            let mut grad = [0.0f64; NF];
            for &r in &perm[start..end] {
                let x = &feats[r];
                let mut logit = 0.0;
                for k in 0..NF {
                    logit += x[k] * coef[k];
                }
                let p = 1.0 / (1.0 + (-logit).exp());
                let dl = weights[r] * (p - y[r]);
                for k in 0..NF {
                    grad[k] += dl * x[k] * STD[k];
                }
            }
            // cosine lr for this step
            let lr = lr0 * 0.5 * (1.0 + (std::f64::consts::PI * step / t_max).cos());
            t += 1.0;
            let bc1 = 1.0 - b1.powf(t);
            let bc2 = 1.0 - b2.powf(t);
            for k in 0..NF {
                m[k] = b1 * m[k] + (1.0 - b1) * grad[k];
                v[k] = b2 * v[k] + (1.0 - b2) * grad[k] * grad[k];
                let mhat = m[k] / bc1;
                let vhat = v[k] / bc2;
                // AdamW: decoupled weight decay on the parameter (coef_res).
                coef_res[k] -= lr * (mhat / (vhat.sqrt() + eps) + wd * coef_res[k]);
            }
            step += 1.0;
            start = end;
        }
    }

    let mut coef = [0.0f64; NF];
    for k in 0..NF {
        coef[k] = coef_res[k] * STD[k] + MEAN[k];
    }
    coef
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let feats = compute_features(ds);
    let y: Vec<f64> = rows.iter().map(|r| r.y as f64).collect();
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_coef = MEAN;

    for s in splits {
        let train_feats = &feats[..s.test_start];
        let train_y = &y[..s.test_start];
        let coef = if cfg.default_params {
            MEAN
        } else {
            train(train_feats, train_y)
        };
        for i in s.test_start..s.test_end {
            let x = &feats[i];
            let mut logit = 0.0;
            for k in 0..NF {
                logit += x[k] * coef[k];
            }
            eval_rows.push(rows[i].clone());
            p.push(1.0 / (1.0 + (-logit).exp()));
        }
        last_coef = coef;
    }

    ModelOutput {
        eval_rows,
        p,
        params: Params::Flat(last_coef.to_vec()),
    }
}
