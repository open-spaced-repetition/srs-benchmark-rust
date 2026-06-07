//! Model processing. Each `process_*` returns the evaluation rows (concatenation of the
//! per-split test folds, in split order), the matching predictions `p`, and any trained
//! parameters to record.

use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

/// Result of running a model over one user's dataset.
pub struct ModelOutput {
    pub eval_rows: Vec<Row>,
    pub p: Vec<f64>,
    pub params: Params,
}

/// AVG baseline (`model_processors.baseline`): each split predicts the train fold's mean y.
pub fn process_avg(ds: &Dataset, cfg: &Config) -> ModelOutput {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bce(p: f64, y: f64) -> f64 {
        let e = f64::EPSILON;
        let pc = p.clamp(e, 1.0 - e);
        -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln())
    }

    #[test]
    fn hlr_grad_matches_finite_difference() {
        // Synthetic rows with varied success/fail counts, intervals, labels.
        let x0 = vec![1.0, 1.4142135, 2.0, 0.0, 1.7320508];
        let x1 = vec![0.0, 1.0, 1.4142135, 1.0, 0.0];
        let dt = vec![1.5, 10.0, 0.5, 3.0, 100.0];
        let yv = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        let wv = vec![1.0, 0.7, 1.3, 1.0, 0.4];
        let m = Hlr {
            x0,
            x1,
            dt,
            yv: yv.clone(),
            wv: wv.clone(),
        };
        let w = vec![2.5819, -0.8674, 2.7245];
        let idx: Vec<usize> = (0..5).collect();
        let g = m.grad(&w, &idx);

        // Numeric gradient of sum_i wv_i * BCE(p_i, y_i)
        let loss = |w: &[f64]| -> f64 {
            let p = m.predict(w, &idx);
            (0..5).map(|i| wv[i] * bce(p[i], yv[i])).sum()
        };
        let h = 1e-6;
        for k in 0..3 {
            let mut wp = w.clone();
            let mut wm = w.clone();
            wp[k] += h;
            wm[k] -= h;
            let num = (loss(&wp) - loss(&wm)) / (2.0 * h);
            assert!(
                (num - g[k]).abs() < 1e-4,
                "param {k}: analytic {} vs numeric {}",
                g[k],
                num
            );
        }
    }

    #[test]
    fn dash_grad_matches_finite_difference() {
        let feat = vec![
            [1.0, 1.0, 3.0, 2.0, 5.0, 3.0, 5.0, 3.0],
            [0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 4.0, 2.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ];
        let yv = vec![1.0, 0.0, 1.0];
        let wv = vec![1.0, 0.8, 1.2];
        let m = Dash {
            feat,
            yv: yv.clone(),
            wv: wv.clone(),
        };
        let w = vec![-0.1766, 0.4483, -0.3618, 0.5953, -0.5104, 0.8609, -0.3643, 0.6447, 1.2815];
        let idx: Vec<usize> = (0..3).collect();
        let g = m.grad(&w, &idx);
        let loss = |w: &[f64]| -> f64 {
            let p = m.predict(w, &idx);
            (0..3).map(|i| wv[i] * bce(p[i], yv[i])).sum()
        };
        let h = 1e-6;
        for k in 0..9 {
            let mut wp = w.clone();
            let mut wm = w.clone();
            wp[k] += h;
            wm[k] -= h;
            let num = (loss(&wp) - loss(&wm)) / (2.0 * h);
            assert!((num - g[k]).abs() < 1e-4, "param {k}: {} vs {}", g[k], num);
        }
    }
}

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
        ivl = crate::metrics::py_round_half_even(ivl + 0.01).max(1.0).min(s_max);
    }
    ivl
}

/// SM2 (untrainable): predict `0.9^(delta_t / sm2_ivl(prior ratings))` per test row.
pub fn process_sm2(ds: &Dataset, cfg: &Config) -> ModelOutput {
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

// ---------------------------------------------------------------------------------------
// HLR (Half-Life Regression) — first Adam-trained model.
// stability = 2^(w0*sqrt(#success) + w1*sqrt(#fail) + bias); retention = 0.5^(delta_t/s).
// ---------------------------------------------------------------------------------------

/// √success-count and √failure-count from a prior-rating sequence.
fn hlr_features(prior: &[i64]) -> (f64, f64) {
    let mut succ = 0u32;
    let mut fail = 0u32;
    for &r in prior {
        if r == 1 {
            fail += 1;
        } else {
            succ += 1;
        }
    }
    ((succ as f64).sqrt(), (fail as f64).sqrt())
}

struct Hlr {
    x0: Vec<f64>,
    x1: Vec<f64>,
    dt: Vec<f64>,
    yv: Vec<f64>,
    wv: Vec<f64>,
}

impl Hlr {
    const INIT_W: [f64; 3] = [2.5819, -0.8674, 2.7245];

    fn from_rows(ds: &Dataset, rows: &[Row], weights: &[f64]) -> Self {
        let mut x0 = Vec::with_capacity(rows.len());
        let mut x1 = Vec::with_capacity(rows.len());
        let mut dt = Vec::with_capacity(rows.len());
        let mut yv = Vec::with_capacity(rows.len());
        for r in rows {
            let (s, f) = hlr_features(ds.prior_ratings(r));
            x0.push(s);
            x1.push(f);
            dt.push(r.delta_t);
            yv.push(r.y as f64);
        }
        Hlr {
            x0,
            x1,
            dt,
            yv,
            wv: weights.to_vec(),
        }
    }

    #[inline]
    fn p_row(&self, w: &[f64], i: usize) -> f64 {
        let d = w[0] * self.x0[i] + w[1] * self.x1[i] + w[2];
        let s = (LN2 * d).exp(); // 2^d
        (-LN2 * self.dt[i] / s).exp() // 0.5^(dt/s)
    }
}

const LN2: f64 = std::f64::consts::LN_2;

impl BatchModel for Hlr {
    fn n_params(&self) -> usize {
        3
    }
    fn init_params(&self) -> Vec<f64> {
        Hlr::INIT_W.to_vec()
    }
    fn n_rows(&self) -> usize {
        self.x0.len()
    }
    fn seq_len(&self, _row: usize) -> usize {
        2 // tensor is always [sqrt_succ, sqrt_fail]
    }
    fn y(&self, row: usize) -> f64 {
        self.yv[row]
    }
    fn weight(&self, row: usize) -> f64 {
        self.wv[row]
    }
    fn predict(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        idx.iter().map(|&i| self.p_row(params, i)).collect()
    }
    fn grad(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let ln2sq = LN2 * LN2;
        let mut g = [0.0f64; 3];
        for &i in idx {
            let d = params[0] * self.x0[i] + params[1] * self.x1[i] + params[2];
            let s = (LN2 * d).exp();
            let a = self.dt[i] / s;
            let p = (-LN2 * a).exp();
            // torch BCELoss backward: dBCE/dp = (p-y)/max(p*(1-p), 1e-12); then dp/dd =
            // p*ln2^2*a. (For non-extreme p this equals (p-y)*ln2^2*a/(1-p).)
            let denom = (p * (1.0 - p)).max(1e-12);
            let gd = self.wv[i] * (p - self.yv[i]) / denom * p * ln2sq * a;
            g[0] += gd * self.x0[i];
            g[1] += gd * self.x1[i];
            g[2] += gd;
        }
        g.to_vec()
    }
}

// ---------------------------------------------------------------------------------------
// DASH — sigmoid(W·log(x+1)+b) over 8 time-window features. Well-behaved (bounded).
// ---------------------------------------------------------------------------------------

/// 8 DASH time-window features (no decay): per window [1,7,30,inf] days, the count of
/// prior reviews whose time-to-now ≤ window, and how many were successes (rating>1).
/// `intervals` = `dt_active[1..=pos]` (same length as `prior_ratings`).
fn dash_features(prior_ratings: &[i64], intervals: &[f64]) -> [f64; 8] {
    let n = prior_ratings.len();
    // cumulative_times[k] = sum(intervals[k..]) — time from review k to now.
    let mut cum = vec![0.0f64; n];
    let mut s = 0.0;
    for k in (0..n).rev() {
        s += intervals[k];
        cum[k] = s;
    }
    let windows = [1.0, 7.0, 30.0, f64::INFINITY];
    let mut f = [0.0f64; 8];
    for (j, &w) in windows.iter().enumerate() {
        for k in 0..n {
            if cum[k] <= w {
                f[2 * j] += 1.0;
                if prior_ratings[k] > 1 {
                    f[2 * j + 1] += 1.0;
                }
            }
        }
    }
    f
}

struct Dash {
    feat: Vec<[f64; 8]>,
    yv: Vec<f64>,
    wv: Vec<f64>,
}

impl Dash {
    /// init_w depends on config (short-term vs not, MCM variant) — see models/dash.py.
    fn init_w(cfg: &Config) -> Vec<f64> {
        if cfg.include_short_term {
            vec![
                -0.1766, 0.4483, -0.3618, 0.5953, -0.5104, 0.8609, -0.3643, 0.6447, 1.2815,
            ]
        } else if !cfg.model_name.contains("MCM") {
            vec![
                0.2024, 0.5967, 0.1255, 0.6039, -0.1485, 0.572, 0.0933, 0.4801, 0.787,
            ]
        } else {
            vec![
                0.2783, 0.8131, 0.4252, 1.0056, -0.1527, 0.6455, 0.1409, 0.669, 0.843,
            ]
        }
    }

    fn from_rows(ds: &Dataset, rows: &[Row], weights: &[f64]) -> Self {
        let mut feat = Vec::with_capacity(rows.len());
        let mut yv = Vec::with_capacity(rows.len());
        for r in rows {
            feat.push(dash_features(ds.prior_ratings(r), ds.intervals_from_second(r)));
            yv.push(r.y as f64);
        }
        Dash {
            feat,
            yv,
            wv: weights.to_vec(),
        }
    }

    #[inline]
    fn z(&self, w: &[f64], i: usize) -> f64 {
        let mut z = w[8];
        for k in 0..8 {
            z += w[k] * (self.feat[i][k] + 1.0).ln();
        }
        z
    }
}

impl BatchModel for Dash {
    fn n_params(&self) -> usize {
        9
    }
    fn init_params(&self) -> Vec<f64> {
        unreachable!("Dash init comes from config; set via process_dash")
    }
    fn n_rows(&self) -> usize {
        self.feat.len()
    }
    fn seq_len(&self, _row: usize) -> usize {
        8
    }
    fn y(&self, row: usize) -> f64 {
        self.yv[row]
    }
    fn weight(&self, row: usize) -> f64 {
        self.wv[row]
    }
    fn predict(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        idx.iter()
            .map(|&i| 1.0 / (1.0 + (-self.z(params, i)).exp()))
            .collect()
    }
    fn grad(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let mut g = vec![0.0f64; 9];
        for &i in idx {
            let p = 1.0 / (1.0 + (-self.z(params, i)).exp());
            let pq = p * (1.0 - p);
            // torch: grad_z = (p-y)/clamp(pq,1e-12) * pq  (≈ p-y for non-extreme p)
            let grad_z = self.wv[i] * (p - self.yv[i]) / pq.max(1e-12) * pq;
            for k in 0..8 {
                g[k] += grad_z * (self.feat[i][k] + 1.0).ln();
            }
            g[8] += grad_z;
        }
        g
    }
}

/// DASH (Adam-trained). Per split: train on `rows[0..test_start]`, predict the test fold.
pub fn process_dash(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let tc = TrainConfig::default();
    let init = Dash::init_w(cfg);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = init.clone();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            init.clone()
        } else {
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Dash::from_rows(ds, train, &weights);
            train::train_with_init(&model, &tc, init.clone())
        };
        let test = &rows[s.test_start..s.test_end];
        let test_model = Dash::from_rows(ds, test, &vec![1.0; test.len()]);
        let all: Vec<usize> = (0..test.len()).collect();
        for (r, pr) in test.iter().zip(test_model.predict(&w, &all)) {
            eval_rows.push(r.clone());
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

/// Recency weights `0.25 + 0.75*x^3`, x = linspace(0,1,N) (`_apply_recency_weighting`).
fn recency_weights(n: usize, recency: bool) -> Vec<f64> {
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

/// HLR (Adam-trained). Per split: train on `rows[0..test_start]`, predict the test fold.
pub fn process_hlr(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let tc = TrainConfig::default();
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = Hlr::INIT_W.to_vec();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            Hlr::INIT_W.to_vec()
        } else {
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Hlr::from_rows(ds, train, &weights);
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let test_model = Hlr::from_rows(ds, test, &vec![1.0; test.len()]);
        let all: Vec<usize> = (0..test.len()).collect();
        let preds = test_model.predict(&w, &all);
        for (r, pr) in test.iter().zip(preds) {
            eval_rows.push(r.clone());
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

/// MOVING-AVG (`model_processors.moving_avg`): a sequential logistic update over all rows
/// in review_th order; predictions recorded only from the first test index onward.
pub fn process_moving_avg(ds: &Dataset, cfg: &Config) -> ModelOutput {
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
