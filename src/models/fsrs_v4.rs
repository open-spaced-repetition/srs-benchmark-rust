//! FSRS v4 — `models/fsrs_v4.py`. 17 params; `forgetting = (1 + t/(9s))^-1`.
//! S0 (w[0..4]) fitted via `fit_s0` then FROZEN during training; trains on `i > 2` rows.

use super::fsrs_init::fit_s0;
use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 17;
const INIT_W: [f64; NP] = [
    0.4, 0.9, 2.3, 10.9, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29,
    2.61,
];

/// forgetting curve (1 + t/(9s))^-1; value form (for the S0 fit).
fn fc_val(t: f64, s: f64) -> f64 {
    (1.0 + t / (9.0 * s)).powf(-1.0)
}

#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>) -> Dual<P> {
    Dual::<P>::c(t).div(s.mul_c(9.0)).add_c(1.0).powf_c(-1.0)
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
            // new_s = w[rating-1]; new_d = clamp(w4 - w5*(rating-3), 1, 10)
            let ns = w[(rating as usize) - 1];
            let nd = w[4].sub(w[5].mul_c(rating - 3.0)).clamp(1.0, 10.0);
            (ns, nd)
        } else {
            let r = fc(prior_dt[k], s);
            let ns = if rating > 1.0 {
                // s*(1 + exp(w8)*(11-d)*s^-w9*(exp((1-r)*w10)-1)*hard*easy)  [d = OLD difficulty]
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
                // w11 * d^-w12 * ((s+1)^w13 - 1) * exp((1-r)*w14)
                w[11]
                    .mul(d.powd(w[12].neg()))
                    .mul(s.add_c(1.0).powd(w[13]).add_c(-1.0))
                    .mul(r.c_sub(1.0).mul(w[14]).exp())
            };
            // new_d = mean_reversion(w4, d - w6*(rating-3)) = w7*w4 + (1-w7)*(d - w6*(rating-3))
            let nd0 = d.sub(w[6].mul_c(rating - 3.0));
            let nd = w[7].mul(w[4]).add(w[7].c_sub(1.0).mul(nd0)).clamp(1.0, 10.0);
            (ns, nd)
        };
        s = ns.clamp(s_min, s_max);
        d = nd;
    }
    fc(cur_dt, s)
}

struct Fsrs4<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    s_min: f64,
    s_max: f64,
}

impl<'a> Fsrs4<'a> {
    /// `train_filter` (Some when building the training set) keeps `i > 2` and
    /// `pos <= max_seq_len`; eval keeps all rows.
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
        Fsrs4 { ds, rows: out_rows, weights: out_w, s_min: cfg.s_min, s_max: cfg.s_max }
    }
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.prior_dt_active(row), self.ds.prior_ratings(row), row.delta_t, w, self.s_min, self.s_max)
    }
}

impl BatchModel for Fsrs4<'_> {
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
        // FSRS4ParameterClipper clamps w[4..=16]; w[0..3] (S0) are left as fitted/frozen.
        w[4] = w[4].clamp(1.0, 10.0);
        w[5] = w[5].clamp(0.1, 5.0);
        w[6] = w[6].clamp(0.1, 5.0);
        w[7] = w[7].clamp(0.0, 0.5);
        w[8] = w[8].clamp(0.0, 3.0);
        w[9] = w[9].clamp(0.1, 0.8);
        w[10] = w[10].clamp(0.01, 2.5);
        w[11] = w[11].clamp(0.5, 5.0);
        w[12] = w[12].clamp(0.01, 0.2);
        w[13] = w[13].clamp(0.01, 0.9);
        w[14] = w[14].clamp(0.01, 2.0);
        w[15] = w[15].clamp(0.0, 1.0);
        w[16] = w[16].clamp(1.0, 4.0);
    }
    fn grad_mask(&self, g: &mut [f64]) {
        // Freeze initial stability (w[0..4]) — apply_gradient_constraints.
        for v in g.iter_mut().take(4) {
            *v = 0.0;
        }
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
            // S0 fit on the full train slice (i==2 rows), then train (freeze w[0..4]).
            let s0 = fit_s0(ds, train, cfg, [INIT_W[0], INIT_W[1], INIT_W[2], INIT_W[3]], fc_val);
            let mut init = INIT_W.to_vec();
            init[0..4].copy_from_slice(&s0);
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Fsrs4::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
            train::train_with_init(&model, &tc, init)
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Fsrs4::build(ds, test, &vec![1.0; test.len()], None, cfg);
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
    fn fsrs4_grad_matches_finite_difference() {
        let prior_dt = [0.0, 2.0, 9.0, 1.5, 30.0];
        let prior_r = [3i64, 1, 3, 4, 2];
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
            assert!((num - grad[k]).abs() < 2e-4, "param {k}: {} vs {}", grad[k], num);
        }
    }
}
