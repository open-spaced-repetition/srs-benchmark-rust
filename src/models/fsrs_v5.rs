//! FSRS-5 — `models/fsrs_v5.py`. 19 params, power curve `(1+factor·t/s)^-0.5`, short-term
//! stability branch, L2 penalty toward the (S0-initialized) start. S0 is fitted then
//! TRAINED (no freeze); trains on all rows (no i>2 filter).

use super::fsrs_init::fit_s0;
use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 19;
const DECAY: f64 = -0.5;
const GAMMA: f64 = 1.0;
const INIT_W: [f64; NP] = [
    0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, 1.01925,
    1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
];
const STDDEV: [f64; NP] = [
    6.61, 9.52, 17.69, 27.74, 0.55, 0.28, 0.67, 0.12, 0.4, 0.18, 0.34, 0.27, 0.08, 0.14, 0.57,
    0.25, 1.03, 0.27, 0.39,
];

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
            // init_d = w4 - exp(w5*(rating-1)) + 1
            let nd = w[4].sub(w[5].mul_c(rating - 1.0).exp()).add_c(1.0).clamp(1.0, 10.0);
            (ns, nd)
        } else {
            let r = fc(prior_dt[k], s);
            let short_term = prior_dt[k] < 1.0;
            let success = rating > 1.0;
            let ns = if short_term {
                // s * exp(w17*(rating-3+w18))
                s.mul(w[18].add_c(rating - 3.0).mul(w[17]).exp())
            } else if success {
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
                // min( w11*d^-w12*((s+1)^w13-1)*exp((1-r)*w14),  s/exp(w17*w18) )
                let nf = w[11]
                    .mul(d.powd(w[12].neg()))
                    .mul(s.add_c(1.0).powd(w[13]).add_c(-1.0))
                    .mul(r.c_sub(1.0).mul(w[14]).exp());
                let min_s = s.div(w[17].mul(w[18]).exp());
                nf.min(min_s)
            };
            // next_d
            let delta_d = w[6].mul_c(-(rating - 3.0));
            let damp = delta_d.mul(d.c_sub(10.0).mul_c(1.0 / 9.0)); // delta_d*(10-d)/9
            let nd0 = d.add(damp);
            let initd4 = w[4].sub(w[5].mul_c(3.0).exp()).add_c(1.0);
            let nd = w[7].mul(initd4).add(w[7].c_sub(1.0).mul(nd0)).clamp(1.0, 10.0);
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
    init_ref: [f64; NP],
    s_min: f64,
    s_max: f64,
}

impl<'a> Model<'a> {
    fn build(
        ds: &'a Dataset,
        rows: &[Row],
        weights: &[f64],
        max_seq_len: Option<usize>,
        init_ref: [f64; NP],
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
        Model { ds, rows: out_rows, weights: out_w, init_ref, s_min: cfg.s_min, s_max: cfg.s_max }
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
        self.init_ref.to_vec()
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
        // L2 penalty gradient: d/dw[k] of (Σ(w-w0)²/σ² · batch · γ / N_train)
        let scale = idx.len() as f64 * GAMMA / self.rows.len() as f64;
        for k in 0..NP {
            g[k] += 2.0 * (params[k] - self.init_ref[k]) / (STDDEV[k] * STDDEV[k]) * scale;
        }
        g
    }
    fn eval_penalty(&self, params: &[f64]) -> f64 {
        let mut s = 0.0;
        for k in 0..NP {
            let d = params[k] - self.init_ref[k];
            s += d * d / (STDDEV[k] * STDDEV[k]);
        }
        s * GAMMA
    }
    fn clip_params(&self, w: &mut [f64]) {
        let (smn, smx) = (self.s_min, 100.0); // init_s_max = 100
        w[0] = w[0].clamp(smn, smx);
        w[1] = w[1].clamp(smn, smx);
        w[2] = w[2].clamp(smn, smx);
        w[3] = w[3].clamp(smn, smx);
        w[4] = w[4].clamp(1.0, 10.0);
        w[5] = w[5].clamp(0.001, 4.0);
        w[6] = w[6].clamp(0.001, 4.0);
        w[7] = w[7].clamp(0.001, 0.75);
        w[8] = w[8].clamp(0.0, 4.5);
        w[9] = w[9].clamp(0.0, 0.8);
        w[10] = w[10].clamp(0.001, 3.5);
        w[11] = w[11].clamp(0.001, 5.0);
        w[12] = w[12].clamp(0.001, 0.25);
        w[13] = w[13].clamp(0.001, 0.9);
        w[14] = w[14].clamp(0.0, 4.0);
        w[15] = w[15].clamp(0.0, 1.0);
        w[16] = w[16].clamp(1.0, 6.0);
        w[17] = w[17].clamp(0.0, 2.0);
        w[18] = w[18].clamp(0.0, 2.0);
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
        } else if cfg.only_s0 {
            // --S0: just S0 init, no training (FSRS Trainer returns init weights).
            let s0 = fit_s0(ds, train, cfg, [INIT_W[0], INIT_W[1], INIT_W[2], INIT_W[3]], fc_val);
            let mut init = INIT_W;
            init[0..4].copy_from_slice(&s0);
            init.to_vec()
        } else {
            let s0 = fit_s0(ds, train, cfg, [INIT_W[0], INIT_W[1], INIT_W[2], INIT_W[3]], fc_val);
            let mut init = INIT_W;
            init[0..4].copy_from_slice(&s0);
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Model::build(ds, train, &weights, Some(cfg.max_seq_len), init, cfg);
            train::train_with_init(&model, &tc, init.to_vec())
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Model::build(ds, test, &vec![1.0; test.len()], None, INIT_W, cfg);
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
    fn fsrs5_grad_matches_finite_difference() {
        // include a short-term (delta_t<1) step to exercise that branch.
        let prior_dt = [0.0, 0.3, 9.0, 1.5, 30.0];
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
