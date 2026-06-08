//! FSRS-6 — `models/fsrs_v6.py`. 21 params. Like v5 but the forgetting curve has a trained
//! decay parameter `w[20]` (`(1+factor·t/s)^decay`, decay=-w[20]) and the short-term
//! stability uses `s^-w19` with a `max(·,1)` floor on success. L2 penalty; S0 fit (trained).

use super::fsrs_init::fit_s0;
use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 21;
const GAMMA: f64 = 1.0;
/// FSRS-6 default parameters (`fsrs_optimizer.DEFAULT_PARAMETER`). Public so FSRS-6-one-step
/// can reuse them as its `init_w`.
pub const INIT_W: [f64; NP] = [
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796, 1.4835,
    0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
];
const STDDEV: [f64; NP] = [
    6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18, 0.33, 0.3, 0.09, 0.16, 0.57,
    0.25, 1.03, 0.31, 0.32, 0.14, 0.27,
];

/// forgetting curve with a trained decay: factor = 0.9^(1/decay) - 1; (1 + factor·t/s)^decay.
fn fc_val(t: f64, s: f64, decay: f64) -> f64 {
    let factor = 0.9f64.powf(1.0 / decay) - 1.0;
    (1.0 + factor * t / s).powf(decay)
}
#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>, decay: Dual<P>) -> Dual<P> {
    // factor = 0.9^(1/decay) - 1
    let factor = Dual::<P>::c(0.9).powd(Dual::<P>::c(1.0).div(decay)).add_c(-1.0);
    factor.mul_c(t).div(s).add_c(1.0).powd(decay) // (1 + factor*t/s)^decay
}

fn retention<const P: usize>(
    prior_dt: &[f64],
    prior_r: &[i64],
    cur_dt: f64,
    w: &[Dual<P>; NP],
    s_min: f64,
    s_max: f64,
) -> Dual<P> {
    let decay = w[20].neg(); // forgetting_curve decay = -w[20]
    let mut s = Dual::<P>::c(0.0);
    let mut d = Dual::<P>::c(0.0);
    for k in 0..prior_r.len() {
        let rating = prior_r[k] as f64;
        let (ns, nd) = if k == 0 {
            let ns = w[(rating as usize) - 1];
            let nd = w[4].sub(w[5].mul_c(rating - 1.0).exp()).add_c(1.0).clamp(1.0, 10.0);
            (ns, nd)
        } else {
            let r = fc(prior_dt[k], s, decay);
            let short_term = prior_dt[k] < 1.0;
            let success = rating > 1.0;
            let ns = if short_term {
                // sinc = exp(w17*(rating-3+w18))*s^-w19; new_s = s*(rating>=2 ? max(sinc,1) : sinc)
                let sinc = w[18]
                    .add_c(rating - 3.0)
                    .mul(w[17])
                    .exp()
                    .mul(s.powd(w[19].neg()));
                let f = if rating >= 2.0 { sinc.max(Dual::c(1.0)) } else { sinc };
                s.mul(f)
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
                let nf = w[11]
                    .mul(d.powd(w[12].neg()))
                    .mul(s.add_c(1.0).powd(w[13]).add_c(-1.0))
                    .mul(r.c_sub(1.0).mul(w[14]).exp());
                let min_s = s.div(w[17].mul(w[18]).exp());
                nf.min(min_s)
            };
            let delta_d = w[6].mul_c(-(rating - 3.0));
            let damp = delta_d.mul(d.c_sub(10.0).mul_c(1.0 / 9.0));
            let nd0 = d.add(damp);
            let initd4 = w[4].sub(w[5].mul_c(3.0).exp()).add_c(1.0);
            let nd = w[7].mul(initd4).add(w[7].c_sub(1.0).mul(nd0)).clamp(1.0, 10.0);
            (ns, nd)
        };
        s = ns.clamp(s_min, s_max);
        d = nd;
    }
    fc(cur_dt, s, decay)
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
    fn build(ds: &'a Dataset, rows: &[Row], weights: &[f64], max_seq_len: Option<usize>, init_ref: [f64; NP], cfg: &Config) -> Self {
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
        let (smn, smx) = (self.s_min, 100.0);
        for v in w.iter_mut().take(4) {
            *v = v.clamp(smn, smx);
        }
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
        w[19] = w[19].clamp(0.0, 0.8);
        w[20] = w[20].clamp(0.1, 0.8);
    }
}

/// Fit S0 then train one FSRS-6 weight set on `train` (or just the S0-init for `--S0`, or
/// `INIT_W` for `--default`). Shared by the global, train_equals_test, and per-partition paths.
fn train_weights(ds: &Dataset, train: &[Row], cfg: &Config, tc: &TrainConfig, default_decay: f64) -> Vec<f64> {
    if cfg.default_params {
        return INIT_W.to_vec();
    }
    let s0 = fit_s0(ds, train, cfg, [INIT_W[0], INIT_W[1], INIT_W[2], INIT_W[3]], |t, s| {
        fc_val(t, s, default_decay)
    });
    let mut init = INIT_W;
    init[0..4].copy_from_slice(&s0);
    if cfg.only_s0 {
        init.to_vec()
    } else {
        let weights = recency_weights(train.len(), cfg.use_recency_weighting);
        let model = Model::build(ds, train, &weights, Some(cfg.max_seq_len), init, cfg);
        train::train_with_init(&model, tc, init.to_vec())
    }
}

/// Predict retrievability for `test` rows under explicit FSRS-6 weights `w`, in `test` order.
/// Used by FSRS-6-one-step (which trains `w` by online SGD but predicts with stock FSRS-6).
pub fn predict(ds: &Dataset, test: &[Row], w: &[f64], cfg: &Config) -> Vec<f64> {
    let tm = Model::build(ds, test, &vec![1.0; test.len()], None, INIT_W, cfg);
    let all: Vec<usize> = (0..tm.rows.len()).collect();
    tm.predict(w, &all)
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let tc = TrainConfig::default();
    let default_decay = -INIT_W[20];

    if cfg.partitions != "none" {
        return process_partitioned(ds, cfg, &tc, default_decay);
    }

    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = INIT_W.to_vec();

    // `--train_equals_test`: train on ALL rows and test on the tail (rows from the first
    // split's test fold onward), as a single fold. Else the normal per-split loop.
    let iters: Vec<(usize, usize, usize)> = if cfg.train_equals_test {
        vec![(rows.len(), splits[0].test_start, rows.len())]
    } else {
        splits.iter().map(|s| (s.test_start, s.test_start, s.test_end)).collect()
    };

    for (train_end, test_start, test_end) in iters {
        let train = &rows[..train_end];
        let w = train_weights(ds, train, cfg, &tc, default_decay);
        let test = &rows[test_start..test_end];
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

/// `--partitions deck|preset`: train separate weights per partition (deck/preset id), then
/// predict each partition's test rows with its own weights (INIT_W if a partition has no
/// train data). The eval row-set is identical to the non-partitioned run, so `size` matches.
fn process_partitioned(ds: &Dataset, cfg: &Config, tc: &TrainConfig, default_decay: f64) -> ModelOutput {
    use std::collections::HashMap;
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_pw: Vec<(String, Vec<f64>)> = Vec::new();

    for s in splits {
        let train = &rows[..s.test_start];
        let test = &rows[s.test_start..s.test_end];

        let mut parts: Vec<i64> = train.iter().map(|r| r.partition).collect();
        parts.sort_unstable();
        parts.dedup();
        let mut pw: HashMap<i64, Vec<f64>> = HashMap::new();
        for &pt in &parts {
            let train_p: Vec<Row> = train.iter().filter(|r| r.partition == pt).cloned().collect();
            pw.insert(pt, train_weights(ds, &train_p, cfg, tc, default_decay));
        }

        let mut tparts: Vec<i64> = test.iter().map(|r| r.partition).collect();
        tparts.sort_unstable();
        tparts.dedup();
        for &pt in &tparts {
            let test_p: Vec<Row> = test.iter().filter(|r| r.partition == pt).cloned().collect();
            let w = pw.get(&pt).cloned().unwrap_or_else(|| INIT_W.to_vec());
            let tm = Model::build(ds, &test_p, &vec![1.0; test_p.len()], None, INIT_W, cfg);
            let all: Vec<usize> = (0..tm.rows.len()).collect();
            for (i, pr) in tm.predict(&w, &all).into_iter().enumerate() {
                eval_rows.push(tm.rows[i].clone());
                p.push(pr);
            }
        }
        last_pw = parts.iter().map(|&pt| (pt.to_string(), pw[&pt].clone())).collect();
    }

    ModelOutput { eval_rows, p, params: Params::Partitioned(last_pw) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fsrs6_grad_matches_finite_difference() {
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
            assert!((num - grad[k]).abs() < 3e-4, "param {k}: {} vs {}", grad[k], num);
        }
    }
}
