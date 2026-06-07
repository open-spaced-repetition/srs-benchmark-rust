//! HLR (Half-Life Regression) — `models/hlr.py`. First Adam-trained model.
//! stability = 2^(w0·√#success + w1·√#fail + bias); retention = 0.5^(delta_t/stability).

use super::{recency_weights, ModelOutput};
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const LN2: f64 = std::f64::consts::LN_2;

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

/// HLR (Adam-trained). Per split: train on `rows[0..test_start]`, predict the test fold.
pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
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
    fn hlr_grad_matches_finite_difference() {
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
}
