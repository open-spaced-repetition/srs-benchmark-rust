//! FSRS v2 — `models/fsrs_v2.py`. 2-state recurrence, `forgetting = 0.9^(t/s)`. 14 params,
//! per-step clipper, no S0 fit / penalty.

use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 14;
const INIT_W: [f64; NP] = [
    1.0, 1.0, 1.0, -1.0, -1.0, 0.2, 3.0, -0.8, -0.2, 1.3, 2.6, -0.2, 0.6, 1.5,
];

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
    for k in 0..prior_r.len() {
        let rating = prior_r[k] as f64;
        let (ns, nd) = if k == 0 {
            // new_s = w0*(w1*(rating-1)+1); new_d = clamp(w2*(w3*(rating-4)+1), 1, 10)
            let ns = w[0].mul(w[1].mul_c(rating - 1.0).add_c(1.0));
            let nd = w[2]
                .mul(w[3].mul_c(rating - 4.0).add_c(1.0))
                .clamp(1.0, 10.0);
            (ns, nd)
        } else {
            let r = fc(prior_dt[k], s);
            // new_d = mean_reversion(w2*(1-w3), d + w4*(rating-3)); clamp(1,10)
            let nd0 = d.add(w[4].mul_c(rating - 3.0));
            let init_d = w[2].mul(w[3].c_sub(1.0)); // w2*(1 - w3)
            let nd = w[5]
                .mul(init_d)
                .add(w[5].c_sub(1.0).mul(nd0))
                .clamp(1.0, 10.0);
            let ns = if rating > 1.0 {
                // s*(1 + exp(w6)*nd^w7*s^w8*(exp((1-r)*w9)-1))
                let term = w[6]
                    .exp()
                    .mul(nd.powd(w[7]))
                    .mul(s.powd(w[8]))
                    .mul(r.c_sub(1.0).mul(w[9]).exp().add_c(-1.0));
                s.mul(term.add_c(1.0))
            } else {
                // w10*nd^w11*s^w12*(exp((1-r)*w13)-1)
                w[10]
                    .mul(nd.powd(w[11]))
                    .mul(s.powd(w[12]))
                    .mul(r.c_sub(1.0).mul(w[13]).exp().add_c(-1.0))
            };
            (ns, nd)
        };
        s = ns.clamp(s_min, s_max);
        d = nd;
    }
    fc(cur_dt, s)
}

struct Fsrs2<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    s_min: f64,
    s_max: f64,
}

impl<'a> Fsrs2<'a> {
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
        Fsrs2 { ds, rows: out_rows, weights: out_w, s_min: cfg.s_min, s_max: cfg.s_max }
    }
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.prior_dt_active(row), self.ds.prior_ratings(row), row.delta_t, w, self.s_min, self.s_max)
    }
}

impl BatchModel for Fsrs2<'_> {
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
        w[1] = w[1].clamp(0.01, 10.0);
        w[2] = w[2].clamp(1.0, 10.0);
        w[3] = w[3].clamp(-10.0, -0.01);
        w[4] = w[4].clamp(-10.0, -0.01);
        w[5] = w[5].clamp(0.0, 1.0);
        w[6] = w[6].clamp(0.0, 5.0);
        w[7] = w[7].clamp(-2.0, -0.01);
        w[8] = w[8].clamp(-2.0, -0.01);
        w[9] = w[9].clamp(0.01, 2.0);
        w[10] = w[10].clamp(0.0, 5.0);
        w[11] = w[11].clamp(-2.0, -0.01);
        w[12] = w[12].clamp(0.01, 1.0);
        w[13] = w[13].clamp(0.01, 2.0);
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
            let model = Fsrs2::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Fsrs2::build(ds, test, &vec![1.0; test.len()], None, cfg);
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
    fn fsrs2_grad_matches_finite_difference() {
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
