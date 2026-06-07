//! DASH[ACT-R] — `models/dash_act_r.py`. 5 params, static (sum over prior reviews):
//! retention = sigmoid(w0·log(1 + clamp_min(Σ_k term_k, 0)) + w4), where for each prior
//! review k with time-to-now `t_k` (> 0.1 days) and success `r_k`,
//! term_k = t_k^-w1 · (r_k ? w3 : w2).

use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 5;
const INIT_W: [f64; NP] = [1.4164, 0.516, -0.0564, 1.9223, 1.0549];

/// `intervals` = `dt_active[1..=pos]`; time-to-now is its reverse cumulative sum.
fn retention<const P: usize>(prior_ratings: &[i64], intervals: &[f64], w: &[Dual<P>; NP]) -> Dual<P> {
    let n = intervals.len();
    let mut ttn = vec![0.0f64; n];
    let mut acc = 0.0;
    for k in (0..n).rev() {
        acc += intervals[k];
        ttn[k] = acc;
    }
    let mut sum = Dual::<P>::c(0.0);
    for k in 0..n {
        let t = ttn[k]; // time from review k to now (days); clamp_min(0.1) then where(==0.1,0,..)
        if t > 0.1 {
            let mult = if prior_ratings[k] > 1 { w[3] } else { w[2] };
            sum = sum.add(Dual::<P>::c(t).powd(w[1].neg()).mul(mult)); // t^-w1 · mult
        }
    }
    // sigmoid(w0·log(1 + max(sum,0)) + w4)
    let inner = w[0].mul(sum.clamp_min(0.0).add_c(1.0).ln()).add(w[4]);
    Dual::<P>::c(1.0).div(inner.neg().exp().add_c(1.0))
}

struct Model<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
}

impl<'a> Model<'a> {
    fn build(ds: &'a Dataset, rows: &[Row], weights: &[f64]) -> Self {
        Model { ds, rows: rows.to_vec(), weights: weights.to_vec() }
    }
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.prior_ratings(row), self.ds.intervals_from_second(row), w)
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
        // tensor is [pos, 2]; length = pos.
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
        w[0] = w[0].max(0.001);
        w[1] = w[1].max(0.001);
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
            let model = Model::build(ds, train, &weights);
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Model::build(ds, test, &vec![1.0; test.len()]);
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
    fn dash_actr_grad_matches_finite_difference() {
        let prior_r = [3i64, 1, 4, 2, 3];
        let intervals = [2.0, 9.0, 0.05, 1.5, 30.0];
        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, INIT_W[k]));
            retention(&prior_r, &intervals, &wd).g
        };
        let val = |w: [f64; NP]| {
            let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(w[k]));
            retention(&prior_r, &intervals, &wd).v
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
