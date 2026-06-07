//! Evaluation metrics, ported from `utils.evaluate` / `utils.rmse_matrix`.
//!
//! Implemented: LogLoss (the rule-#5 metric), RMSE, RMSE(bins), AUC, MBE, precision@90,
//! recall@90. ICI (lowess) and smECE (relplot) are added in a later phase; they don't
//! affect the LogLoss/size verification.

use crate::features::Row;

/// sklearn `log_loss(y, p, labels=[0,1])`: mean binary cross-entropy with eps clipping.
pub fn log_loss(y: &[i64], p: &[f64]) -> f64 {
    let eps = f64::EPSILON; // sklearn "auto" eps = finfo(float64).eps
    let n = y.len() as f64;
    let mut s = 0.0;
    for (&yi, &pi) in y.iter().zip(p) {
        let pc = pi.clamp(eps, 1.0 - eps);
        s += if yi == 1 { -pc.ln() } else { -(1.0 - pc).ln() };
    }
    s / n
}

/// Root mean squared error between predictions and labels.
pub fn rmse(y: &[i64], p: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mut s = 0.0;
    for (&yi, &pi) in y.iter().zip(p) {
        let d = pi - yi as f64;
        s += d * d;
    }
    (s / n).sqrt()
}

/// Mean bias error: mean(p - y).
pub fn mean_bias_error(y: &[i64], p: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mut s = 0.0;
    for (&yi, &pi) in y.iter().zip(p) {
        s += pi - yi as f64;
    }
    s / n
}

/// ROC AUC via the Mann–Whitney U statistic with average ranks for ties. Returns None for
/// a single-class label set (matches sklearn raising → Python `None`).
pub fn auc(y: &[i64], p: &[f64]) -> Option<f64> {
    let n_pos = y.iter().filter(|&&v| v == 1).count();
    let n_neg = y.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    // Rank predictions ascending, averaging tied ranks.
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap());
    let mut ranks = vec![0.0f64; p.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && p[idx[j]] == p[idx[i]] {
            j += 1;
        }
        // average rank (1-based) for ties idx[i..j]
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of (i+1 .. j) inclusive
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    let sum_pos_ranks: f64 = y
        .iter()
        .zip(ranks.iter())
        .filter(|(&yi, _)| yi == 1)
        .map(|(_, &r)| r)
        .sum();
    let u = sum_pos_ranks - (n_pos as f64) * (n_pos as f64 + 1.0) / 2.0;
    Some(u / (n_pos as f64 * n_neg as f64))
}

/// precision@90 and recall@90: threshold predictions at 0.9, zero_division=0.
pub fn precision_recall_at_90(y: &[i64], p: &[f64]) -> (f64, f64) {
    let mut tp = 0i64;
    let mut fp = 0i64;
    let mut fn_ = 0i64;
    for (&yi, &pi) in y.iter().zip(p) {
        let pred = pi >= 0.9;
        if pred && yi == 1 {
            tp += 1;
        } else if pred && yi == 0 {
            fp += 1;
        } else if !pred && yi == 1 {
            fn_ += 1;
        }
    }
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    (precision, recall)
}

/// RMSE (bins) — port of `utils.rmse_matrix`. Bins by (delta_t, i, lapse), groups, then
/// computes the sample-weighted RMSE of (mean y) vs (mean p) per bin.
pub fn rmse_bins(rows: &[Row], p: &[f64], weights: Option<&[f64]>) -> f64 {
    use std::collections::HashMap;

    // Bin keys are rounded floats; encode as ordered integer bits for hashing exactness.
    #[derive(PartialEq, Eq, Hash)]
    struct Key(u64, u64, u64);

    fn bin_delta_t(dt: f64) -> f64 {
        let x = dt.max(1e-6);
        round2(2.48 * 3.62f64.powf((x.ln() / 3.62f64.ln()).floor()))
    }
    fn bin_i(i: f64) -> f64 {
        round0(1.99 * 1.89f64.powf((i.ln() / 1.89f64.ln()).floor()))
    }
    fn bin_lapse(x: f64) -> f64 {
        if x != 0.0 {
            round0(1.65 * 1.73f64.powf((x.ln() / 1.73f64.ln()).floor()))
        } else {
            0.0
        }
    }
    fn round2(x: f64) -> f64 {
        (x * 100.0).round() / 100.0
    }
    fn round0(x: f64) -> f64 {
        x.round()
    }

    struct Acc {
        sy: f64,
        sp: f64,
        w: f64,
    }
    let mut groups: HashMap<Key, Acc> = HashMap::new();
    for (k, r) in rows.iter().enumerate() {
        let dt = bin_delta_t(r.delta_t);
        let ii = bin_i(r.i as f64);
        let lp = bin_lapse(r.rmse_bins_lapse as f64);
        let w = weights.map(|w| w[k]).unwrap_or(1.0);
        let key = Key(dt.to_bits(), ii.to_bits(), lp.to_bits());
        let e = groups.entry(key).or_insert(Acc {
            sy: 0.0,
            sp: 0.0,
            w: 0.0,
        });
        e.sy += r.y as f64 * w;
        e.sp += p[k] * w;
        e.w += w;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for acc in groups.values() {
        let ym = acc.sy / acc.w;
        let pm = acc.sp / acc.w;
        let d = ym - pm;
        num += acc.w * d * d;
        den += acc.w;
    }
    (num / den).sqrt()
}

/// Round to 6 decimals (matches Python `round(x, 6)`).
pub fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Python `round(x)` (no ndigits): round half to even ("banker's rounding").
pub fn py_round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // exact tie -> round to even
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}
