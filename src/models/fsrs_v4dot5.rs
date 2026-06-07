//! FSRS-4.5 — `models/fsrs_v4dot5.py`. Like v4 (S0 fit + freeze, trains on i>2, 17 params)
//! but with the power forgetting curve `(1 + factor·t/s)^decay` (decay=-0.5) and a
//! `min(·, old_s)` cap on the after-failure stability.

use super::fsrs_init::fit_s0;
use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 17;
const DECAY: f64 = -0.5;

const INIT_W_SECS: [f64; NP] = [
    0.0012, 0.0826, 0.8382, 26.2146, 4.8622, 1.0311, 0.8295, 0.0379, 2.0884, 0.4704, 1.2009,
    1.7196, 0.1874, 0.1593, 1.5636, 0.2358, 3.3175,
];
const INIT_W_DAYS: [f64; NP] = [
    0.4872, 1.4003, 3.7145, 13.8206, 5.1618, 1.2298, 0.8975, 0.031, 1.6474, 0.1367, 1.0461,
    2.1072, 0.0793, 0.3246, 1.587, 0.2272, 2.8755,
];

fn init_w(cfg: &Config) -> [f64; NP] {
    if cfg.use_secs_intervals {
        INIT_W_SECS
    } else {
        INIT_W_DAYS
    }
}

#[inline]
fn factor() -> f64 {
    0.9f64.powf(1.0 / DECAY) - 1.0
}

fn fc_val(t: f64, s: f64) -> f64 {
    (1.0 + factor() * t / s).powf(DECAY)
}

#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>) -> Dual<P> {
    Dual::<P>::c(factor() * t).div(s).add_c(1.0).powf_c(DECAY)
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
            let ns = w[(rating as usize) - 1];
            let nd = w[4].sub(w[5].mul_c(rating - 3.0)).clamp(1.0, 10.0);
            (ns, nd)
        } else {
            let r = fc(prior_dt[k], s);
            let ns = if rating > 1.0 {
                // stability_after_success (same as v4): uses OLD difficulty.
                let hard = if rating == 2.0 { w[15] } else { Dual::c(1.0) };
                let easy = if rating == 4.0 { w[16] } else { Dual::c(1.0) };
                let term = w[8]
                    .exp()
                    .mul(d.c_sub(11.0))
                    .mul(s.powd(w[9].neg()))
                    .mul(r.c_sub(1.0).mul(w[10]).exp().add_c(-1.0))
                    .mul(hard)
                    .mul(easy);
                s.mul(term.add_c(1.0))
            } else {
                // v4.5 after-failure: min(w11*d^-w12*((s+1)^w13-1)*exp((1-r)*w14), old_s)
                let nf = w[11]
                    .mul(d.powd(w[12].neg()))
                    .mul(s.add_c(1.0).powd(w[13]).add_c(-1.0))
                    .mul(r.c_sub(1.0).mul(w[14]).exp());
                nf.min(s)
            };
            let nd0 = d.sub(w[6].mul_c(rating - 3.0));
            let nd = w[7].mul(w[4]).add(w[7].c_sub(1.0).mul(nd0)).clamp(1.0, 10.0);
            (ns, nd)
        };
        s = ns.clamp(s_min, s_max);
        d = nd;
    }
    fc(cur_dt, s)
}

struct Model<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    init: [f64; NP],
    s_min: f64,
    s_max: f64,
}

impl<'a> Model<'a> {
    fn build(ds: &'a Dataset, rows: &[Row], weights: &[f64], train_filter: Option<usize>, cfg: &Config) -> Self {
        let mut out_rows = Vec::with_capacity(rows.len());
        let mut out_w = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            if let Some(m) = train_filter {
                if r.i <= 2 || r.pos as usize > m {
                    continue;
                }
            }
            out_rows.push(r.clone());
            out_w.push(weights[i]);
        }
        Model { ds, rows: out_rows, weights: out_w, init: init_w(cfg), s_min: cfg.s_min, s_max: cfg.s_max }
    }
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.prior_dt_active(row), self.ds.prior_ratings(row), row.delta_t, w, self.s_min, self.s_max)
    }
}

impl BatchModel for Model<'_> {
    fn n_params(&self) -> usize {
        NP
    }
    fn init_params(&self) -> Vec<f64> {
        self.init.to_vec()
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
        w[4] = w[4].clamp(0.0, 10.0);
        w[5] = w[5].clamp(0.01, 5.0);
        w[6] = w[6].clamp(0.01, 5.0);
        w[7] = w[7].clamp(0.0, 0.8);
        w[8] = w[8].clamp(0.0, 6.0);
        w[9] = w[9].clamp(0.0, 0.8);
        w[10] = w[10].clamp(0.01, 5.0);
        w[11] = w[11].clamp(0.2, 6.0);
        w[12] = w[12].clamp(0.01, 0.4);
        w[13] = w[13].clamp(0.01, 0.9);
        w[14] = w[14].clamp(0.01, 4.0);
        w[15] = w[15].clamp(0.0, 1.0);
        w[16] = w[16].clamp(1.0, 10.0);
    }
    fn grad_mask(&self, g: &mut [f64]) {
        for v in g.iter_mut().take(4) {
            *v = 0.0;
        }
    }
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let tc = TrainConfig::default();
    let iw = init_w(cfg);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = iw.to_vec();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            iw.to_vec()
        } else {
            let s0 = fit_s0(ds, train, cfg, [iw[0], iw[1], iw[2], iw[3]], fc_val);
            let mut init = iw.to_vec();
            init[0..4].copy_from_slice(&s0);
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Model::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
            train::train_with_init(&model, &tc, init)
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
    fn fsrs45_grad_matches_finite_difference() {
        let prior_dt = [0.0, 2.0, 9.0, 1.5, 30.0];
        let prior_r = [3i64, 1, 3, 4, 2];
        let (cur_dt, s_min, s_max) = (7.0, 0.0001, 36500.0);
        let w0 = INIT_W_SECS;
        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w0[k]));
            retention(&prior_dt, &prior_r, cur_dt, &wd, s_min, s_max).g
        };
        let val = |w: [f64; NP]| {
            let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(w[k]));
            retention(&prior_dt, &prior_r, cur_dt, &wd, s_min, s_max).v
        };
        let h = 1e-6;
        for k in 0..NP {
            let mut wp = w0;
            let mut wm = w0;
            wp[k] += h;
            wm[k] -= h;
            let num = (val(wp) - val(wm)) / (2.0 * h);
            assert!((num - grad[k]).abs() < 2e-4, "param {k}: {} vs {}", grad[k], num);
        }
    }
}
