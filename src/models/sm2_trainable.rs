//! SM2-trainable — `models/sm2_trainable.py`. A 6-param Adam-trained SM-2: an
//! interval/ease-factor/reps state machine over the rating history, with a fixed
//! `0.9^(t/s)` forgetting curve. Uses the same BaseModel training hyperparameters as FSRS
//! (lr=4e-2, n_epoch=5, Adam no weight decay), so `TrainConfig::default()` applies.
//!
//! Note: `step` ignores the interval (`delta_t`) entirely — the state depends only on the
//! ratings — so the prior-interval slice is not needed; only the current `delta_t` feeds the
//! forgetting curve. Branches key off `reps` (a param-independent success count), so the
//! forward-mode autodiff just carries the selected branch's dual.

use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 6;
const INIT_W: [f64; NP] = [1.0, 6.0, 2.5, 0.02, 7.0, 0.18];

/// Forgetting curve `0.9^(t/s)`.
#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>) -> Dual<P> {
    Dual::<P>::c(0.9).powd(Dual::<P>::c(t).div(s))
}

/// Replay the rating history to the stability (interval) at the prediction point, then apply
/// the forgetting curve at `cur_dt`. Mirrors `SM2.forward` + `forgetting_curve`.
fn retention<const P: usize>(prior_r: &[i64], cur_dt: f64, w: &[Dual<P>; NP], s_min: f64, s_max: f64) -> Dual<P> {
    let mut ivl = Dual::<P>::c(0.0);
    let mut ef = w[2]; // state[:,1] initialized to w[2]
    let mut reps: i64 = 0;
    for &rating in prior_r {
        let success = rating > 1;
        let new_reps = if success { reps + 1 } else { 1 };
        // new_ivl = where(reps==1, w0, where(reps==2, w1, ivl*ef))
        let new_ivl = if new_reps == 1 {
            w[0]
        } else if new_reps == 2 {
            w[1]
        } else {
            ivl.mul(ef)
        };
        // EF' = ef - w3*(q - w4)^2 + w5, with q = rating + 1.
        let q = (rating + 1) as f64;
        let diff = Dual::<P>::c(q).sub(w[4]);
        let new_ef = ef.sub(w[3].mul(diff).mul(diff)).add(w[5]);
        ivl = new_ivl.clamp(s_min, s_max);
        ef = new_ef.clamp(1.3, 10.0);
        reps = new_reps;
    }
    fc(cur_dt, ivl)
}

struct Model<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    s_min: f64,
    s_max: f64,
    init_s_max: f64,
}

impl<'a> Model<'a> {
    fn build(ds: &'a Dataset, rows: &[Row], weights: &[f64], max_seq_len: Option<usize>, cfg: &Config) -> Self {
        let mut out_rows = Vec::with_capacity(rows.len());
        let mut out_w = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            if let Some(m) = max_seq_len {
                if r.pos as usize > m {
                    continue;
                }
            }
            out_rows.push(r.clone());
            out_w.push(weights[i]);
        }
        Model {
            ds,
            rows: out_rows,
            weights: out_w,
            s_min: cfg.s_min,
            s_max: cfg.s_max,
            init_s_max: cfg.init_s_max,
        }
    }
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.prior_ratings(row), row.delta_t, w, self.s_min, self.s_max)
    }
}

impl BatchModel for Model<'_> {
    fn n_params(&self) -> usize {
        NP
    }
    fn init_params(&self) -> Vec<f64> {
        INIT_W.to_vec()
    }
    fn n_rows(&self) -> usize {
        self.rows.len()
    }
    fn seq_len(&self, row: usize) -> usize {
        self.rows[row].pos as usize
    }
    fn y(&self, row: usize) -> f64 {
        self.rows[row].y as f64
    }
    fn weight(&self, row: usize) -> f64 {
        self.weights[row]
    }
    fn predict(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(params[k]));
        idx.iter().map(|&i| self.ret(&wd, &self.rows[i]).v).collect()
    }
    fn grad(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, params[k]));
        let mut g = vec![0.0f64; NP];
        for &i in idx {
            let ret = self.ret(&wd, &self.rows[i]);
            let p = ret.v;
            let denom = (p * (1.0 - p)).max(1e-12);
            let dl = self.weights[i] * (p - self.rows[i].y as f64) / denom;
            for k in 0..NP {
                g[k] += dl * ret.g[k];
            }
        }
        g
    }
    fn clip_params(&self, w: &mut [f64]) {
        // SM2ParameterClipper (models/sm2_trainable.py).
        w[0] = w[0].clamp(self.s_min, self.init_s_max);
        w[1] = w[1].clamp(self.s_min, self.init_s_max);
        w[2] = w[2].clamp(1.3, 10.0);
        w[3] = w[3].max(0.0);
        w[4] = w[4].max(5.0);
        // w[5] is unclipped.
    }
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let tc = TrainConfig::default();
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = INIT_W.to_vec();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            INIT_W.to_vec()
        } else {
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Model::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Model::build(ds, test, &vec![1.0; test.len()], None, cfg);
        let all: Vec<usize> = (0..tm.rows.len()).collect();
        for (i, pr) in tm.predict(&w, &all).into_iter().enumerate() {
            eval_rows.push(tm.rows[i].clone());
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

    #[test]
    fn sm2_trainable_grad_matches_finite_difference() {
        let prior_r = [3i64, 1, 3, 4, 2, 3, 3];
        let (cur_dt, s_min, s_max) = (7.0, 0.0001, 36500.0);
        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, INIT_W[k]));
            retention(&prior_r, cur_dt, &wd, s_min, s_max).g
        };
        let val = |w: [f64; NP]| {
            let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(w[k]));
            retention(&prior_r, cur_dt, &wd, s_min, s_max).v
        };
        let h = 1e-6;
        for k in 0..NP {
            let mut wp = INIT_W;
            let mut wm = INIT_W;
            wp[k] += h;
            wm[k] -= h;
            let num = (val(wp) - val(wm)) / (2.0 * h);
            assert!((num - grad[k]).abs() < 1e-5, "param {k}: {} vs {}", grad[k], num);
        }
    }
}
