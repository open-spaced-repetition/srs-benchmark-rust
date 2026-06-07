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

/// `dt_incl` = `dt_active[0..=pos]` (length pos+1). Returns the retention dual.
fn retention<const P: usize>(dt_incl: &[f64], w: &[Dual<P>; NP]) -> Dual<P> {
    let n = dt_incl.len(); // pos+1
    // sp[i] = cumulative days up to review i.
    let mut sp = vec![0.0f64; n];
    let mut acc = 0.0;
    for i in 0..n {
        acc += dt_incl[i];
        sp[i] = acc;
    }
    // m[0] = -inf (only used as exp(m[0]) = 0). m[i] for i>=1 via recurrence.
    // exponent[j] = -(c·exp(m[j]) + a) depends only on j — hoist it out of the inner loop.
    let mut mp = Dual::<P>::c(0.0);
    let mut exponent: Vec<Dual<P>> = vec![Dual::c(0.0); n];
    exponent[0] = w[0].neg(); // -(c·0 + a)
    for i in 1..n {
        let mut sum = Dual::<P>::c(0.0);
        for j in 0..i {
            let dt_sec = (sp[i] - sp[j]) * 86400.0;
            let a = w[4].mul_c(dt_sec).clamp_min(1.0); // (dt_sec·h).clamp_min(1)
            sum = sum.add(a.powd(exponent[j]));
        }
        mp = sum.ln(); // m[i]
        exponent[i] = w[1].mul(mp.exp()).add(w[0]).neg();
    }
    // activation(m[pos]) = 1 / (1 + exp((tau - m)/s)),  tau=w3, s=w2 (m[pos] = last m)
    let z = w[3].sub(mp).div(w[2]); // (tau - m)/s
    Dual::<P>::c(1.0).div(z.exp().add_c(1.0))
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
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.dt_active_incl(row), w)
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
