//! ACT-R — `models/act_r.py`. Activation-based declarative memory. 5 params
//! (a=decay intercept, c=decay scale, s=noise, tau=threshold, h=interference).
//!
//! Per review at card position `pos`: let `sp = cumsum(dt_active[0..=pos])` (days). The
//! activation recurrence is `m[i] = log Σ_{j<i} ((sp[i]-sp[j])·86400·h).clamp_min(1)
//! ^ -(c·exp(m[j]) + a)`, with `exp(m[0]) = 0`. retention = `1/(1+exp((tau-m[pos])/s))`.

use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 5;
const INIT_W: [f64; NP] = [
    0.176_786_766_570_677,
    0.216_967_308_403_809,
    0.254_893_976_981_164,
    -0.704_205_679_427_144,
    0.025,
];

/// Base-level activation `m[i]` for the WHOLE card prefix `dt_incl` (= `dt_active[0..=pos]`, length
/// pos+1): `m[0] = 0` (only `exp(m[0]) = 0` matters, via `exponent[0] = -a`); for i≥1, `m[i] =
/// ln Σ_{j<i} ((sp[i]-sp[j])·86400·h).clamp_min(1)^-(c·exp(m[j])+a)`. Returns ALL `m[i]` so every row
/// of the card reuses one O(n²) pass — the row at pos p only needs `m[p]` — instead of recomputing
/// the recurrence from scratch per row (which made the per-card cost O(n³)).
fn recurrence<const P: usize>(dt_incl: &[f64], w: &[Dual<P>; NP]) -> Vec<Dual<P>> {
    let n = dt_incl.len(); // pos+1
    // sp[i] = cumulative days up to review i.
    let mut sp = vec![0.0f64; n];
    let mut acc = 0.0;
    for i in 0..n {
        acc += dt_incl[i];
        sp[i] = acc;
    }
    let mut m = vec![Dual::<P>::c(0.0); n]; // m[0] = 0 by convention
    // exponent[j] = -(c·exp(m[j]) + a) depends only on j — hoist it out of the inner loop.
    let mut exponent: Vec<Dual<P>> = vec![Dual::c(0.0); n];
    exponent[0] = w[0].neg(); // -(c·0 + a)
    for i in 1..n {
        let mut sum = Dual::<P>::c(0.0);
        for j in 0..i {
            let dt_sec = (sp[i] - sp[j]) * 86400.0;
            let a = w[4].mul_c(dt_sec).clamp_min(1.0); // (dt_sec·h).clamp_min(1)
            // a^exponent[j] = exp(exponent[j]·ln a). Computing ln(a) ONCE here and reusing it for
            // both the value and the gradient saves one transcendental per pair vs `a.powd(e)`
            // (which does powf — itself ln+exp — PLUS a separate ln for the exponent-derivative).
            // The value becomes exp(e·ln a) instead of powf(a,e) — same math, ~1 ULP different.
            // a clamped to 1 ⇒ ln a = 0 ⇒ term = 1 (zero grad), matching the clamp.
            sum = sum.add(a.ln().mul(exponent[j]).exp());
        }
        m[i] = sum.ln(); // m[i]
        exponent[i] = w[1].mul(m[i].exp()).add(w[0]).neg();
    }
    m
}

/// retention from activation `m[pos]`: `1 / (1 + exp((tau - m)/s))`, tau=w3, s=w2.
#[inline]
fn ret_from_m<const P: usize>(m_pos: Dual<P>, w: &[Dual<P>; NP]) -> Dual<P> {
    let z = w[3].sub(m_pos).div(w[2]); // (tau - m)/s
    Dual::<P>::c(1.0).div(z.exp().add_c(1.0))
}

/// `dt_incl` = `dt_active[0..=pos]`. Single-row retention (recurrence then the last activation) —
/// kept for the gradient unit test; the hot path uses [`Model::retentions`] (one recurrence per card).
fn retention<const P: usize>(dt_incl: &[f64], w: &[Dual<P>; NP]) -> Dual<P> {
    let m = recurrence(dt_incl, w);
    ret_from_m(m[m.len() - 1], w)
}

struct Model<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
}

impl<'a> Model<'a> {
    fn build(ds: &'a Dataset, rows: &[Row], weights: &[f64], max_seq_len: Option<usize>) -> Self {
        let mut out_rows = Vec::with_capacity(rows.len());
        let mut out_w = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            if let Some(m) = max_seq_len {
                // tensor length is pos+1; drop rows whose tensor exceeds max_seq_len.
                if (r.pos as usize) + 1 > m {
                    continue;
                }
            }
            out_rows.push(r.clone());
            out_w.push(weights[i]);
        }
        Model { ds, rows: out_rows, weights: out_w }
    }
    /// Retention for each `idx` row, grouped by card so the O(n²) recurrence runs ONCE per card (up
    /// to the max pos needed), reused by every row of that card. Output is in `idx` order.
    fn retentions<const P: usize>(&self, w: &[Dual<P>; NP], idx: &[usize]) -> Vec<Dual<P>> {
        use std::collections::HashMap;
        let mut by_card: HashMap<u32, Vec<usize>> = HashMap::new();
        for (k, &i) in idx.iter().enumerate() {
            by_card.entry(self.rows[i].card_idx).or_default().push(k);
        }
        let mut out = vec![Dual::<P>::c(0.0); idx.len()];
        for ks in by_card.values() {
            // One recurrence per card, up to the deepest pos any of its rows here needs.
            let kmax = *ks.iter().max_by_key(|&&k| self.rows[idx[k]].pos).unwrap();
            let m = recurrence(self.ds.dt_active_incl(&self.rows[idx[kmax]]), w);
            for &k in ks {
                out[k] = ret_from_m(m[self.rows[idx[k]].pos as usize], w);
            }
        }
        out
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
        self.rows[row].pos as usize + 1
    }
    fn y(&self, row: usize) -> f64 {
        self.rows[row].y as f64
    }
    fn weight(&self, row: usize) -> f64 {
        self.weights[row]
    }
    fn predict(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(params[k]));
        self.retentions(&wd, idx).iter().map(|r| r.v).collect()
    }
    fn grad(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, params[k]));
        let rets = self.retentions(&wd, idx);
        let mut g = vec![0.0f64; NP];
        for (kk, &i) in idx.iter().enumerate() {
            let ret = rets[kk];
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
        w[0] = w[0].clamp(0.001, 1.0);
        w[1] = w[1].clamp(0.001, 1.0);
        w[2] = w[2].clamp(0.001, 1.0);
        w[3] = w[3].min(-0.001); // clamp_max(-0.001)
        w[4] = w[4].clamp(0.001, 1.0);
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
            let model = Model::build(ds, train, &weights, Some(cfg.max_seq_len));
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Model::build(ds, test, &vec![1.0; test.len()], None);
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
    #[cfg(feature = "fp64")] // finite-diff (h=1e-6) needs f64; f32 rounding noise dominates
    #[test]
    fn actr_grad_matches_finite_difference() {
        let dt_incl = [0.0, 1.0, 7.0, 0.5, 20.0];
        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, INIT_W[k]));
            retention(&dt_incl, &wd).g
        };
        let val = |w: [f64; NP]| {
            let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(w[k]));
            retention(&dt_incl, &wd).v
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
