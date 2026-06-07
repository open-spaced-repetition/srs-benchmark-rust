//! FSRS v1 — `models/fsrs_v1.py`. 3-state recurrence (stability, difficulty, lapses).
//! Lapses depend only on ratings (constant w.r.t. params). `forgetting = 0.9^(t/s)`.
//! 7 params, per-step clipper, no S0 fit / penalty.

use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 7;
const INIT_W: [f64; NP] = [2.0, 5.0, 3.0, -0.7, -0.2, 1.0, -0.3];

#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>) -> Dual<P> {
    Dual::<P>::c(t).div(s).mul_c(0.9f64.ln()).exp()
}

fn retention<const P: usize>(
    prior_dt: &[f64],
    prior_r: &[i64],
    cur_dt: f64,
    w: &[Dual<P>; NP],
    s_min: f64,
    s_max: f64,
) -> Dual<P> {
    let mut s = Dual::<P>::c(0.0);
    let mut d = Dual::<P>::c(0.0);
    let mut l = 0.0f64; // lapse count (constant w.r.t. params)
    for k in 0..prior_r.len() {
        let rating = prior_r[k] as f64;
        let pow2 = 2f64.powf(rating - 1.0);
        let relu_l = (2.0 - rating).max(0.0);
        let (ns, nd, nl) = if k == 0 {
            // first learn (no clamp on d)
            let ns = w[0].mul_c(0.25 * pow2);
            let nd = w[1].add_c(3.0 - rating); // w1 - rating + 3
            (ns, nd, relu_l)
        } else {
            let r = fc(prior_dt[k], s);
            // new_d = relu(d + r - 0.25*2^(rating-1) + 0.1)
            let nd = d.add(r).add_c(0.1 - 0.25 * pow2).clamp_min(0.0);
            let ns = if rating > 1.0 {
                // s*(1 + exp(w2)*(nd+0.1)^w3 * s^w4 * (exp((1-r)*w5)-1))
                let term = w[2]
                    .exp()
                    .mul(nd.add_c(0.1).powd(w[3]))
                    .mul(s.powd(w[4]))
                    .mul(r.c_sub(1.0).mul(w[5]).exp().add_c(-1.0));
                s.mul(term.add_c(1.0))
            } else {
                // w0 * exp(w6 * old_lapses)
                w[0].mul(w[6].mul_c(l).exp())
            };
            (ns, nd, l + relu_l)
        };
        s = ns.clamp(s_min, s_max);
        d = nd;
        l = nl;
    }
    fc(cur_dt, s)
}

struct Fsrs1<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    s_min: f64,
    s_max: f64,
}

impl<'a> Fsrs1<'a> {
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
        Fsrs1 { ds, rows: out_rows, weights: out_w, s_min: cfg.s_min, s_max: cfg.s_max }
    }
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.prior_dt_active(row), self.ds.prior_ratings(row), row.delta_t, w, self.s_min, self.s_max)
    }
}

impl BatchModel for Fsrs1<'_> {
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
        w[0] = w[0].clamp(0.1, 10.0);
        w[1] = w[1].clamp(1.0, 10.0);
        w[2] = w[2].clamp(0.01, 10.0);
        w[3] = w[3].clamp(-1.0, -0.01);
        w[4] = w[4].clamp(-1.0, -0.01);
        w[5] = w[5].clamp(0.01, 10.0);
        w[6] = w[6].clamp(-1.0, -0.01);
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
            let model = Fsrs1::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Fsrs1::build(ds, test, &vec![1.0; test.len()], None, cfg);
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
    fn fsrs1_grad_matches_finite_difference() {
        let prior_dt = [0.0, 2.0, 9.0, 1.5];
        let prior_r = [3i64, 1, 3, 4];
        let (cur_dt, s_min, s_max) = (7.0, 0.0001, 36500.0);
        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, INIT_W[k]));
            retention(&prior_dt, &prior_r, cur_dt, &wd, s_min, s_max).g
        };
        let val = |w: [f64; NP]| {
            let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(w[k]));
            retention(&prior_dt, &prior_r, cur_dt, &wd, s_min, s_max).v
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
