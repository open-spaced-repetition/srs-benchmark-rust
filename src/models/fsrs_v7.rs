//! FSRS-7 — `models/fsrs_v7.py`. Finished dual-stability FSRS-7 (34 params). The memory state
//! has three components: long-term stability, short-term stability, and difficulty. The
//! forgetting curve mixes a short-term recall component (driven by short S) and a long-term
//! component (driven by long S, difficulty applied to the long-term timescale).
//!
//! Adam-trained (no Reptile, no S0 fit — trains directly from the default params `INIT_W`):
//! n_epoch=9, batch_size=512, lr=0.0118, betas=(0.70,0.98), CosineAnnealingLR, keep-final-epoch
//! (no best-eval checkpoint). The only penalty active by default is the L2-to-default-params
//! prior (`PENALTY_W_L2 * Σ(w−w0)²/σ²`); the scheduling penalties (`--sched_penalties`) are
//! deferred. Param indices: 0–3 initial S by rating, 4–6 difficulty, 7–14 long-term S block,
//! 15–22 short-term S block, 23–33 forgetting curve.

use super::fsrs_v7_grad::{grad_one, predict_card, wconsts, StepCacheBox};
use super::fsrs_v7_simd::{grad_simd, WLanes};
use super::{recency_weights_fsrs7, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

pub const NP: usize = 34;
const PENALTY_W_L2: f64 = 0.3333;
const S_MAX: f64 = 36500.0;
const D_MIN: f64 = 1.0;
const D_MAX: f64 = 10.0;

/// FSRS-7 default parameters (`FSRS7.init_w`). Training starts here; `--default` keeps them.
pub const INIT_W: [f64; NP] = [
    0.1104, 2.2395, 3.9221, 11.7841, // 0-3   initial S by rating
    6.1686, 0.6457, 3.6807, // 4-6   difficulty
    1.9795, 0.0, 1.3826, 0.7024, 0.5999, 0.8146, 0.6398, 1.0, // 7-14  long-term S
    1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, // 15-22 short-term S
    0.1567, 0.0801, 0.2421, 0.9464, 0.1433, 0.7145, 0.0, 0.5667, 0.3734, 0.5333, 0.3048, // 23-33 curve
];

/// L2 prior sigmas (`FSRS7._l2_sigma` / PARAMS_STDDEV). 0..3 are 9999 ⇒ negligible L2 on S0.
const L2_SIGMA: [f64; NP] = [
    9999.0, 9999.0, 9999.0, 9999.0, 0.523, 0.2528, 0.4329, 0.2966, 0.2139, 0.2889, 0.1862, 0.175,
    0.3812, 0.3013, 0.9104, 0.3234, 0.2448, 0.3273, 0.1842, 0.1735, 0.4608, 0.311, 0.864, 0.0418,
    0.2596, 0.0798, 0.0682, 0.1282, 0.1397, 0.1407, 0.1489, 0.2, 0.15, 0.15,
];

/// Box-clamp bounds for the 34-param clipper (`FSRS7ParameterClipper`).
const CLIP_LO: [f64; NP] = [
    0.0001, 0.0001, 0.0001, 0.0001, 1.0, 0.001, 0.1, 0.0, 0.0, 0.3, 0.01, 0.1, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.5, 0.001, 0.001, 0.0, 0.0, 1.0, 0.01, 0.01, 0.2, 0.5, 0.01, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0,
];
const CLIP_HI: [f64; NP] = [
    50.0, 100.0, 100.0, 100.0, 10.0, 4.0, 4.0, 4.0, 1.2, 3.0, 1.5, 1.0, 3.5, 1.0, 7.0, 4.0, 2.0,
    6.0, 1.5, 1.0, 5.0, 1.0, 7.0, 0.25, 0.95, 0.85, 0.99, 1.0, 1.0, 0.9, 1.1, 1.0, 0.6, 0.6,
];

/// `init_d(rating) = w4 − exp(w5·(rating−1)) + 1` (unclamped; callers clamp where needed).
#[inline]
fn init_d<const P: usize>(w: &[Dual<P>; NP], rating: f64) -> Dual<P> {
    w[4].sub(w[5].mul_c(rating - 1.0).exp()).add_c(1.0)
}

/// Short-term recall component `r1` (driven by the short-term S; decay S-modulated via w33).
#[inline]
fn short_recall<const P: usize>(t: f64, s_short: Dual<P>, w: &[Dual<P>; NP]) -> Dual<P> {
    let t = t.max(0.0);
    let decay1 = w[23].mul(s_short.powd(w[33].add_c(-0.3))).clamp(0.01, 0.95).neg();
    // factor1 = exp(min(log(w25)/decay1, 60)) − 1
    let factor1 = w[25].ln().div(decay1).clamp_max(60.0).exp().add_c(-1.0);
    // (t/s_short · factor1 + 1)^decay1
    factor1.mul_c(t).div(s_short).add_c(1.0).powd(decay1)
}

/// Dual-stability forgetting curve `R(t, s_long, s_short, d)` (final 1e-5 rescale included).
#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>, s_short: Dual<P>, d: Dual<P>, w: &[Dual<P>; NP]) -> Dual<P> {
    let t = t.max(0.0);
    let r1 = short_recall(t, s_short, w);

    // Long-term component r2 (difficulty on the horizontal time-scale; decay2 not d-modulated).
    let decay2 = w[24].clamp(0.01, 0.95).neg();
    let decay2_inv = Dual::<P>::c(1.0).div(decay2);
    let factor2 = w[26].powd(decay2_inv).add_c(-1.0);
    let d_timescale = d.add_c(-5.0).mul(w[32].add_c(-0.3)).exp();
    let r2 = factor2.mul(d_timescale).mul_c(t).div(s).add_c(1.0).powd(decay2);

    // Mixture weights (t-independent); weight2 is D-modulated.
    let weight1 = w[27].mul(s_short.powd(w[29].neg()));
    let weight2 = w[28].mul(s.powd(w[30])).mul(d.add_c(-5.0).mul(w[31].add_c(-0.5)).exp());

    let retention = weight1.mul(r1).add(weight2.mul(r2)).div(weight1.add(weight2));
    retention.mul_c(1.0 - 2e-5).add_c(1e-5)
}

/// Stability after a review. `start` selects the block (7 long-term, 15 short-term).
/// Post-lapse stability is D-independent. `r` is the recall driving this trace.
#[inline]
fn next_stability<const P: usize>(
    last_s: Dual<P>,
    last_d: Dual<P>,
    r: Dual<P>,
    rating: f64,
    start: usize,
    w: &[Dual<P>; NP],
) -> Dual<P> {
    let new_s_fail = w[start + 3]
        .mul(last_s.add_c(1.0).powd(w[start + 4]).add_c(-1.0))
        .mul(r.c_sub(1.0).mul(w[start + 5]).exp());
    let pls = last_s.min(new_s_fail);
    if rating > 1.0 {
        let hard = if rating == 2.0 { w[start + 6] } else { Dual::c(1.0) };
        let easy = if rating == 4.0 { w[start + 7] } else { Dual::c(1.0) };
        // sinc = exp(w[start]−1.5)·(11−d)·s^(−w[start+1])·(exp((1−r)·w[start+2])−1)·hard·easy + 1
        let sinc = w[start]
            .add_c(-1.5)
            .exp()
            .mul(last_d.c_sub(11.0))
            .mul(last_s.powd(w[start + 1].neg()))
            .mul(r.c_sub(1.0).mul(w[start + 2]).exp().add_c(-1.0))
            .mul(hard)
            .mul(easy)
            .add_c(1.0);
        pls.max(last_s.mul(sinc))
    } else {
        pls
    }
}

/// Difficulty update with surprise-weighted lapse (scale δd by `retention+0.1` on a lapse).
#[inline]
fn next_difficulty<const P: usize>(last_d: Dual<P>, rating: f64, retention: Dual<P>, w: &[Dual<P>; NP]) -> Dual<P> {
    let delta_d = w[6].mul_c(-(rating - 3.0));
    let delta_d = if rating == 1.0 { delta_d.mul(retention.add_c(0.1)) } else { delta_d };
    // linear_damping: δd·(10−d)/9
    let new_d0 = last_d.add(delta_d.mul(last_d.c_sub(10.0).mul_c(1.0 / 9.0)));
    // mean_reversion (fixed 1%/99%): 0.01·init_d(4) + 0.99·new_d
    let new_d = init_d(w, 4.0).mul_c(0.01).add(new_d0.mul_c(0.99));
    new_d.clamp(D_MIN, D_MAX)
}

/// Run the dual-stability recurrence over the prior reviews, then predict at `cur_dt`.
/// Forward-mode (`Dual<P>`) reference implementation — kept as the gradient/predict oracle that
/// validates the faster hand-written reverse-mode path in [`super::fsrs_v7_grad`].
pub fn retention_dual<const P: usize>(
    prior_dt: &[f64],
    prior_r: &[i64],
    cur_dt: f64,
    w: &[Dual<P>; NP],
    s_min: f64,
) -> Dual<P> {
    let mut s = Dual::<P>::c(0.0); // long-term S
    let mut s_short = Dual::<P>::c(0.0);
    let mut d = Dual::<P>::c(0.0);
    for k in 0..prior_r.len() {
        let rating = prior_r[k] as f64;
        if k == 0 {
            let idx = (rating.clamp(1.0, 4.0) as usize) - 1;
            let init_s_long = w[idx];
            s = init_s_long.clamp(s_min, S_MAX);
            s_short = init_s_long.mul_c(0.8).clamp(s_min, S_MAX);
            d = init_d(w, rating).clamp(D_MIN, D_MAX);
        } else {
            let t = prior_dt[k];
            let last_s = s.clamp(s_min, S_MAX);
            let last_s_short = s_short.clamp(s_min, S_MAX);
            let last_d = d.clamp(D_MIN, D_MAX);
            let r = fc(t, last_s, last_s_short, last_d, w); // mixed retrievability
            let upd_s_long = next_stability(last_s, last_d, r, rating, 7, w);
            let r1 = short_recall(t, last_s_short, w);
            let mut upd_s_short = next_stability(last_s_short, last_d, r1, rating, 15, w);
            if rating == 1.0 {
                // post-lapse short-term reset: cap at 0.8 · post-lapse long-term S
                upd_s_short = upd_s_short.min(upd_s_long.mul_c(0.8));
            }
            let upd_d = next_difficulty(last_d, rating, r, w);
            s = upd_s_long.clamp(s_min, S_MAX);
            s_short = upd_s_short.clamp(s_min, S_MAX);
            d = upd_d.clamp(D_MIN, D_MAX);
        }
    }
    fc(cur_dt, s.clamp(s_min, S_MAX), s_short.clamp(s_min, S_MAX), d.clamp(D_MIN, D_MAX), w)
}

struct Model<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    s_min: f64,
}

impl<'a> Model<'a> {
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
        Model { ds, rows: out_rows, weights: out_w, s_min: cfg.s_min }
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
        self.rows[row].pos as usize
    }
    fn y(&self, row: usize) -> f64 {
        self.rows[row].y as f64
    }
    fn weight(&self, row: usize) -> f64 {
        self.weights[row]
    }
    fn predict(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let wc = wconsts(params, self.s_min);
        let mut out = vec![0.0f64; idx.len()];
        // Windowed O(C) predict: group output slots by card, replay each card's sequence ONCE
        // (emitting a prediction at each requested position). Sorting by (card_idx, pos) makes
        // each card's slots a contiguous run with positions already ascending. Bit-identical to
        // the per-row `predict_one`, just O(C) instead of O(C²) over a card's rows.
        let mut order: Vec<usize> = (0..idx.len()).collect();
        order.sort_by_key(|&s| {
            let r = &self.rows[idx[s]];
            (r.card_idx, r.pos)
        });
        let mut start = 0usize;
        while start < order.len() {
            let card_idx = self.rows[idx[order[start]]].card_idx;
            let mut end = start + 1;
            while end < order.len() && self.rows[idx[order[end]]].card_idx == card_idx {
                end += 1;
            }
            let slots = &order[start..end];
            let positions: Vec<usize> = slots.iter().map(|&s| self.rows[idx[s]].pos as usize).collect();
            let cur_dts: Vec<f64> = slots.iter().map(|&s| self.rows[idx[s]].delta_t).collect();
            let card = &self.ds.cards[card_idx as usize];
            let mut card_out = vec![0.0f64; slots.len()];
            predict_card(params, &card.dt_active, &card.ratings, &positions, &cur_dts, &wc, &mut card_out);
            for (j, &s) in slots.iter().enumerate() {
                out[s] = card_out[j];
            }
            start = end;
        }
        out
    }
    fn grad(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        // f64×4 SIMD reverse-mode gradient (4 per-prefix rows/lane). Same per-prefix batching as
        // the scalar path it replaces (validated bit-close in `fsrs_v7_simd`'s test), so the
        // training trajectory is preserved up to the ~1e-14 f64×4-transcendental difference.
        let wc = wconsts(params, self.s_min);
        let wl = WLanes::new(params, &wc);
        let mut g = vec![0.0f64; NP];
        let mut fallback: Vec<usize> = Vec::new();
        grad_simd(self.ds, &self.rows, &self.weights, idx, &wl, self.s_min, &mut g, &mut fallback);
        // pos==0 rows (empty prefix; rare) go through the scalar path — exactly as before.
        if !fallback.is_empty() {
            let mut caches: Vec<StepCacheBox> = Vec::new();
            for &i in &fallback {
                let row = &self.rows[i];
                grad_one(
                    params,
                    self.ds.prior_dt_active(row),
                    self.ds.prior_ratings(row),
                    row.delta_t,
                    row.y as f64,
                    self.weights[i],
                    &wc,
                    &mut g,
                    &mut caches,
                );
            }
        }
        // L2-to-default penalty gradient: PENALTY_W_L2 · (batch/epoch_len) · 2(w−w0)/σ².
        let scale = idx.len() as f64 * PENALTY_W_L2 / self.rows.len() as f64;
        for k in 0..NP {
            g[k] += 2.0 * (params[k] - INIT_W[k]) / (L2_SIGMA[k] * L2_SIGMA[k]) * scale;
        }
        g
    }
    fn clip_params(&self, w: &mut [f64]) {
        for k in 0..NP {
            w[k] = w[k].clamp(CLIP_LO[k], CLIP_HI[k]);
        }
        // Cross-parameter monotonicity (after the box clamps).
        w[1] = w[1].max(w[0]);
        w[2] = w[2].max(w[1]);
        w[3] = w[3].max(w[2]);
        w[26] = w[26].max(w[25]);
    }
}

fn train_config() -> TrainConfig {
    TrainConfig { lr: 0.0118, betas: (0.70, 0.98), n_epoch: 9, batch_size: 512, keep_final: true }
}

/// Train one FSRS-7 weight set on `train` (or `INIT_W` for `--default`).
fn train_weights(ds: &Dataset, train: &[Row], cfg: &Config, tc: &TrainConfig) -> Vec<f64> {
    if cfg.default_params {
        return INIT_W.to_vec();
    }
    let weights = recency_weights_fsrs7(train.len(), cfg.use_recency_weighting);
    let model = Model::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
    train::train_with_init(&model, tc, INIT_W.to_vec())
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let tc = train_config();

    if cfg.partitions != "none" {
        return process_partitioned(ds, cfg, &tc);
    }

    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = INIT_W.to_vec();

    // Folds: equalize splits if set, else train_equals_test (single fold), else TimeSeriesSplit.
    let folds: Vec<(Vec<Row>, Vec<Row>)> = if let Some(eq) = &ds.equalize_splits {
        eq.iter()
            .map(|sp| {
                let train: Vec<Row> = rows[..sp.train_end].to_vec();
                let test: Vec<Row> = sp.test_idx.iter().map(|&i| rows[i].clone()).collect();
                (train, test)
            })
            .collect()
    } else if cfg.train_equals_test {
        let splits = time_series_split(rows.len(), cfg.n_splits);
        vec![(rows.to_vec(), rows[splits[0].test_start..].to_vec())]
    } else {
        time_series_split(rows.len(), cfg.n_splits)
            .into_iter()
            .map(|s| (rows[..s.test_start].to_vec(), rows[s.test_start..s.test_end].to_vec()))
            .collect()
    };

    for (train, test) in folds {
        let w = train_weights(ds, &train, cfg, &tc);
        let tm = Model::build(ds, &test, &vec![1.0; test.len()], None, cfg);
        let all: Vec<usize> = (0..tm.rows.len()).collect();
        for (i, pr) in tm.predict(&w, &all).into_iter().enumerate() {
            eval_rows.push(tm.rows[i].clone());
            p.push(pr);
        }
        last_w = w;
    }

    ModelOutput { eval_rows, p, params: Params::Partitioned(vec![("0".to_string(), last_w)]) }
}

/// `--partitions deck|preset`: train separate weights per partition, predict each partition's
/// test rows with its own weights (INIT_W if the partition has no train data).
fn process_partitioned(ds: &Dataset, cfg: &Config, tc: &TrainConfig) -> ModelOutput {
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
            pw.insert(pt, train_weights(ds, &train_p, cfg, tc));
        }

        let mut tparts: Vec<i64> = test.iter().map(|r| r.partition).collect();
        tparts.sort_unstable();
        tparts.dedup();
        for &pt in &tparts {
            let test_p: Vec<Row> = test.iter().filter(|r| r.partition == pt).cloned().collect();
            let w = pw.get(&pt).cloned().unwrap_or_else(|| INIT_W.to_vec());
            let tm = Model::build(ds, &test_p, &vec![1.0; test_p.len()], None, cfg);
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
    fn fsrs7_grad_matches_finite_difference() {
        let prior_dt = [0.0, 0.3, 9.0, 1.5, 30.0, 0.02, 100.0];
        let prior_r = [3i64, 1, 3, 4, 2, 1, 3];
        let (cur_dt, s_min) = (7.0, 0.001);
        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, INIT_W[k]));
            retention_dual(&prior_dt, &prior_r, cur_dt, &wd, s_min).g
        };
        let val = |w: [f64; NP]| {
            let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(w[k]));
            retention_dual(&prior_dt, &prior_r, cur_dt, &wd, s_min).v
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
