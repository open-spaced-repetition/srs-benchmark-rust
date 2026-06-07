//! FSRS v3 — `models/fsrs_v3.py` (forward from v2, `forgetting_curve = 0.9^(t/s)` from v1).
//! 13 params, direct init (no S0 fit), no penalty, with a per-step parameter clipper.
//!
//! The per-review recurrence is written once over `Dual<P>`: `P = 0` gives a fast
//! value-only forward (predict), `P = NP` gives parameter gradients (train).

use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 13;
const INIT_W: [f64; NP] = [
    0.9605, 1.7234, 4.8527, -1.1917, -1.2956, 0.0573, 1.7352, -0.1673, 1.065, 1.8907, -0.3832,
    0.5867, 1.0721,
];

/// forgetting curve 0.9^(t/s) = exp(ln(0.9) · t / s); `t` is a constant, `s` a dual.
#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>) -> Dual<P> {
    Dual::<P>::c(t).div(s).mul_c(0.9f64.ln()).exp()
}

/// Retention for one review: run the recurrence over the prior reviews, then the forgetting
/// curve at the current interval. Generic over the gradient width `P`.
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
            // first learn
            let ns = w[0].add(w[1].mul_c(rating - 1.0));
            let nd = w[2].add(w[3].mul_c(rating - 3.0)).clamp(1.0, 10.0);
            (ns, nd)
        } else {
            let r = fc(prior_dt[k], s);
            // difficulty: mean_reversion(w2, d + w4*(rating-3)); clamp(1,10)
            let mut nd = d.add(w[4].mul_c(rating - 3.0));
            nd = w[5].mul(w[2]).add(w[5].c_sub(1.0).mul(nd)); // w5*w2 + (1-w5)*nd
            nd = nd.clamp(1.0, 10.0);
            let ns = if rating > 1.0 {
                // s * (1 + exp(w6)*(11-nd)*s^w7*(exp((1-r)*w8)-1))
                let term = w[6]
                    .exp()
                    .mul(nd.c_sub(11.0))
                    .mul(s.powd(w[7]))
                    .mul(r.c_sub(1.0).mul(w[8]).exp().add_c(-1.0));
                s.mul(term.add_c(1.0))
            } else {
                // w9 * nd^w10 * s^w11 * exp((1-r)*w12)
                w[9]
                    .mul(nd.powd(w[10]))
                    .mul(s.powd(w[11]))
                    .mul(r.c_sub(1.0).mul(w[12]).exp())
            };
            (ns, nd)
        };
        s = ns.clamp(s_min, s_max);
        d = nd;
    }
    fc(cur_dt, s)
}

struct Fsrs3<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    s_min: f64,
    s_max: f64,
}

impl<'a> Fsrs3<'a> {
    /// `max_seq_len = Some(m)` filters training rows with prefix length > m (BatchDataset);
    /// `None` keeps all rows (eval).
    fn build(
        ds: &'a Dataset,
        rows: &[Row],
        weights: &[f64],
        max_seq_len: Option<usize>,
        cfg: &Config,
    ) -> Self {
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
        Fsrs3 {
            ds,
            rows: out_rows,
            weights: out_w,
            s_min: cfg.s_min,
            s_max: cfg.s_max,
        }
    }

    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(
            self.ds.prior_dt_active(row),
            self.ds.prior_ratings(row),
            row.delta_t,
            w,
            self.s_min,
            self.s_max,
        )
    }
}

impl BatchModel for Fsrs3<'_> {
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
            let dloss_dp = self.weights[i] * (p - self.rows[i].y as f64) / denom;
            for k in 0..NP {
                g[k] += dloss_dp * ret.g[k];
            }
        }
        g
    }
    fn clip_params(&self, w: &mut [f64]) {
        w[0] = w[0].clamp(0.1, 10.0);
        w[1] = w[1].clamp(0.1, 5.0);
        w[2] = w[2].clamp(1.0, 10.0);
        w[3] = w[3].clamp(-5.0, -0.1);
        w[4] = w[4].clamp(-5.0, -0.1);
        w[5] = w[5].clamp(0.05, 0.5);
        w[6] = w[6].clamp(0.0, 2.0);
        w[7] = w[7].clamp(-0.8, -0.15);
        w[8] = w[8].clamp(0.01, 1.5);
        w[9] = w[9].clamp(0.5, 5.0);
        w[10] = w[10].clamp(-2.0, -0.01);
        w[11] = w[11].clamp(0.01, 0.9);
        w[12] = w[12].clamp(0.01, 2.0);
    }
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let tc = TrainConfig::default(); // FSRS v1–v6 use BaseModel defaults
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = INIT_W.to_vec();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            INIT_W.to_vec()
        } else {
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Fsrs3::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let test_model = Fsrs3::build(ds, test, &vec![1.0; test.len()], None, cfg);
        let all: Vec<usize> = (0..test_model.rows.len()).collect();
        for (i, pr) in test_model.predict(&w, &all).into_iter().enumerate() {
            eval_rows.push(test_model.rows[i].clone());
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
    fn fsrs3_grad_matches_finite_difference() {
        // Recurrence over a small synthetic prior sequence, both branches exercised.
        let prior_dt = [0.0, 2.0, 9.0, 1.5];
        let prior_r = [3i64, 1, 3, 4];
        let cur_dt = 7.0;
        let s_min = 0.0001;
        let s_max = 36500.0;
        let w0 = INIT_W;

        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w0[k]));
            let r = retention(&prior_dt, &prior_r, cur_dt, &wd, s_min, s_max);
            // d(retention)/dw
            r.g
        };
        let val = |w: [f64; NP]| -> f64 {
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
            assert!(
                (num - grad[k]).abs() < 1e-5,
                "param {k}: dual {} vs numeric {}",
                grad[k],
                num
            );
        }
    }
}
