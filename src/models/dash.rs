//! DASH — `models/dash.py`. sigmoid(W·log(x+1)+b) over 8 time-window features.
//! Well-behaved (bounded sigmoid output, bounded gradient).

use super::{recency_weights, ModelOutput};
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

/// 8 DASH time-window features (no decay): per window [1,7,30,inf] days, the count of prior
/// reviews whose time-to-now ≤ window, and how many were successes (rating>1).
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
        unreachable!("Dash init comes from config; train_with_init is used")
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
pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bce(p: f64, y: f64) -> f64 {
        let e = f64::EPSILON;
        let pc = p.clamp(e, 1.0 - e);
        -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln())
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
