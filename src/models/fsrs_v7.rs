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
        // f32×8 SIMD reverse-mode gradient (8 per-prefix rows/lane), genuine f32 like torch. Same
        // per-prefix batching as the scalar path (validated f32-close in `fsrs_v7_simd`'s test),
        // so the training trajectory is preserved up to the f32 transcendental difference.
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
    TrainConfig {
        lr: 0.0118,
        betas: (0.70, 0.98),
        n_epoch: 9,
        batch_size: 512,
        keep_final: true,
        hyper_beta: 0.0,
    }
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

    if cfg.partitions == "smart" {
        return process_smart(ds, cfg, &tc);
    }
    if cfg.partitions != "none" {
        return process_partitioned(ds, cfg, &tc);
    }
    if cfg.reopt_growth > 0.0 {
        return process_geometric(ds, cfg, &tc);
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

/// A train set is "adequate" iff at least one row survives FSRS-7's training filter (`pos`
/// within `max_seq_len`; FSRS-7's `filter_training_data` is a no-op, so this is the only drop).
fn adequate(set: &[Row], max_seq: usize) -> bool {
    set.iter().any(|r| (r.pos as usize) <= max_seq)
}

/// Train one weight vector per partition present in `train` — the per-deck step shared by
/// `--partitions deck` and `--partitions smart`. Applies `script.py`'s **double fallback**: a
/// partition whose training data is "inadequate" (empty after dropping rows with
/// `pos > max_seq_len`) falls back to **user-level weights** (trained on the whole split,
/// computed lazily), then to INIT_W. Returns the sorted partition ids, their weights, and the
/// user-level weights if they were computed (`None` if no partition needed them, the whole split
/// is inadequate, or `--default`).
fn train_partition_weights(
    ds: &Dataset,
    train: &[Row],
    cfg: &Config,
    tc: &TrainConfig,
) -> (Vec<i64>, std::collections::HashMap<i64, Vec<f64>>, Option<Vec<f64>>) {
    use std::collections::HashMap;
    let max_seq = cfg.max_seq_len;
    let mut parts: Vec<i64> = train.iter().map(|r| r.partition).collect();
    parts.sort_unstable();
    parts.dedup();
    // `Some(None)` = full split inadequate → defaults; `Some(Some(w))` = trained user-level.
    let mut user_level: Option<Option<Vec<f64>>> = None;
    let mut pw: HashMap<i64, Vec<f64>> = HashMap::new();
    for &pt in &parts {
        let train_p: Vec<Row> = train.iter().filter(|r| r.partition == pt).cloned().collect();
        let w = if cfg.default_params {
            // --default never trains (Python returns INIT_W before any inadequate check).
            INIT_W.to_vec()
        } else if adequate(&train_p, max_seq) {
            train_weights(ds, &train_p, cfg, tc)
        } else {
            if user_level.is_none() {
                user_level = Some(adequate(train, max_seq).then(|| train_weights(ds, train, cfg, tc)));
            }
            user_level.as_ref().unwrap().clone().unwrap_or_else(|| INIT_W.to_vec())
        };
        pw.insert(pt, w);
    }
    (parts, pw, user_level.flatten())
}

/// `--partitions deck|preset`: train separate weights per partition, predict each partition's
/// test rows with its own weights (with the double-fallback in `train_partition_weights`).
fn process_partitioned(ds: &Dataset, cfg: &Config, tc: &TrainConfig) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_pw: Vec<(String, Vec<f64>)> = Vec::new();

    for s in splits {
        let train = &rows[..s.test_start];
        let test = &rows[s.test_start..s.test_end];
        let (parts, pw, _ul) = train_partition_weights(ds, train, cfg, tc);

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

/// Method-independent per-split state for smart-preset assignment, computed ONCE and reused across
/// every clustering experiment in a sweep: the trained per-deck params, their whitened vectors, and
/// the user's global params (for test-only-deck / inadequate-cluster fallback).
struct DeckSplit {
    deck_ids: Vec<i64>,
    points: Vec<Vec<f64>>,
    global_w: Option<Vec<f64>>,
    /// KL-distance geometry (only when `--cluster_distance kl`); `points` is empty in that case.
    kl: Option<KlGeom>,
}

/// Prediction-space geometry for the KL-divergence clustering metric: the symmetric per-deck KL
/// distance matrix, plus the KL distance from the user's global model to each deck (for assigning
/// test-only decks). Computed once per split and shared across every clustering experiment.
struct KlGeom {
    /// `dist[i][j]` = mean over the user's rows of the symmetric KL divergence between deck `i`'s and
    /// deck `j`'s recall predictions.
    dist: Vec<Vec<f64>>,
    /// `global_dist[i]` = same KL distance from the user-level (global) model to deck `i`.
    global_dist: Option<Vec<f64>>,
}

/// Train per-deck params for one split and whiten them — the expensive step shared by all
/// clustering experiments (step 4). `global_w` is the user-level params for test-only-deck /
/// inadequate-cluster fallback: reuse the deck step's if it produced one, else train it only when
/// the split actually has a test-only deck.
fn compute_deck_split(
    ds: &Dataset,
    cfg: &Config,
    tc: &TrainConfig,
    cov: &crate::smart::Cov,
    train: &[Row],
    test: &[Row],
) -> DeckSplit {
    use std::collections::HashSet;
    let (deck_ids, deck_w, ul) = train_partition_weights(ds, train, cfg, tc);
    let deck_set: HashSet<i64> = deck_ids.iter().copied().collect();
    let has_test_only = test.iter().any(|r| !deck_set.contains(&r.partition));
    let global_w = ul.or_else(|| {
        if has_test_only && !cfg.default_params && adequate(train, cfg.max_seq_len) {
            Some(train_weights(ds, train, cfg, tc))
        } else {
            None
        }
    });
    if cfg.cluster_distance == "kl" {
        // Prediction-space geometry: cluster decks by how similarly their trained models predict on
        // the user's own rows (the covariance/whitening is unused on this path).
        let kl = compute_kl_geom(ds, cfg, train, &deck_ids, &deck_w, global_w.as_deref());
        DeckSplit { deck_ids, points: Vec::new(), global_w, kl: Some(kl) }
    } else {
        let points: Vec<Vec<f64>> = deck_ids.iter().map(|d| cov.whiten(&deck_w[d])).collect();
        DeckSplit { deck_ids, points, global_w, kl: None }
    }
}

/// Clamp predictions to (0,1) and precompute per-row logits. Symmetric Bernoulli KL collapses to
/// ½·(pₐ−p_b)·(logit pₐ − logit p_b), so the only `ln`s are these D×rows logits computed once per
/// deck — the O(decks²) distance loop ([`sym_kl`]) is then a plain dot product (critical: users have
/// up to ~5000 decks, and `ln` in that loop made the sweep take weeks).
fn kl_prep(p: &[f64]) -> (Vec<f64>, Vec<f64>) {
    const EPS: f64 = 1e-6;
    let pc: Vec<f64> = p.iter().map(|&x| x.clamp(EPS, 1.0 - EPS)).collect();
    let lg: Vec<f64> = pc.iter().map(|&x| x.ln() - (1.0 - x).ln()).collect();
    (pc, lg)
}

/// Mean symmetric Bernoulli KL between two prepped (clamped-p, logit) vectors: mean over rows of
/// ½·(pₐ−p_b)·(logitₐ − logit_b). Empty ⇒ 0.
fn sym_kl(pa: &[f64], la: &[f64], pb: &[f64], lb: &[f64]) -> f64 {
    if pa.is_empty() {
        return 0.0;
    }
    let mut s = 0.0;
    for r in 0..pa.len() {
        s += (pa[r] - pb[r]) * (la[r] - lb[r]);
    }
    0.5 * s / pa.len() as f64
}

/// Reference per-row symmetric KL (the slow `ln`-in-loop form) — kept only to prove `sym_kl` matches.
#[cfg(test)]
fn bern_kl(pa: f64, pb: f64) -> f64 {
    const EPS: f64 = 1e-6;
    let a = pa.clamp(EPS, 1.0 - EPS);
    let b = pb.clamp(EPS, 1.0 - EPS);
    a * (a / b).ln() + (1.0 - a) * ((1.0 - a) / (1.0 - b)).ln()
}

#[cfg(test)]
fn mean_sym_kl(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let mut s = 0.0;
    for k in 0..a.len() {
        s += 0.5 * (bern_kl(a[k], b[k]) + bern_kl(b[k], a[k]));
    }
    s / a.len() as f64
}

/// Cap on rows used to estimate each KL deck-distance. The KL distance is O(decks²·rows) and users
/// here have up to ~5000 decks, so the full row set is intractable (weeks). A strided subsample of
/// this many rows is an unbiased mean-KL estimate at the same scale, so the calibrated thresholds
/// still apply; clustering is robust to the small per-pair estimation noise.
const KL_DIST_CAP: usize = 256;

/// Up to `cap` row indices spread across `0..n` by stride (all of them if `n ≤ cap`).
fn kl_eval_indices(n: usize, cap: usize) -> Vec<usize> {
    if n <= cap {
        return (0..n).collect();
    }
    (0..cap).map(|i| i * n / cap).collect()
}

/// Build the KL geometry for one split: predict every deck's trained params on a single model built
/// over all of this split's training rows (so the prediction vectors are aligned), then form the
/// symmetric per-deck KL distance matrix and the global-model→deck distances. Distances are estimated
/// on a strided subsample of rows ([`KL_DIST_CAP`]) to keep the O(decks²·rows) cost tractable.
fn compute_kl_geom(
    ds: &Dataset,
    cfg: &Config,
    train: &[Row],
    deck_ids: &[i64],
    deck_w: &std::collections::HashMap<i64, Vec<f64>>,
    global_w: Option<&[f64]>,
) -> KlGeom {
    let nd = deck_ids.len();
    let tm = Model::build(ds, train, &vec![1.0; train.len()], None, cfg);
    let idx = kl_eval_indices(tm.rows.len(), KL_DIST_CAP);
    let prep: Vec<(Vec<f64>, Vec<f64>)> =
        deck_ids.iter().map(|d| kl_prep(&tm.predict(&deck_w[d], &idx))).collect();

    let mut dist = vec![vec![0.0f64; nd]; nd];
    for i in 0..nd {
        for j in (i + 1)..nd {
            let dval = sym_kl(&prep[i].0, &prep[i].1, &prep[j].0, &prep[j].1);
            dist[i][j] = dval;
            dist[j][i] = dval;
        }
    }
    let global_dist = global_w.map(|gw| {
        let (gpc, glg) = kl_prep(&tm.predict(gw, &idx));
        prep.iter().map(|(pc, lg)| sym_kl(&gpc, &glg, pc, lg)).collect::<Vec<f64>>()
    });
    maybe_dump_kl(&dist);
    KlGeom { dist, global_dist }
}

/// Calibration hook: if `SMART_KL_DUMP` is set to a path, append this split's off-diagonal KL
/// distances (one per line) so the 6 hierarchical KL thresholds can be chosen from real data.
fn maybe_dump_kl(dist: &[Vec<f64>]) {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    static DUMP: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let slot = DUMP.get_or_init(|| {
        std::env::var("SMART_KL_DUMP").ok().map(|path| {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("open SMART_KL_DUMP file");
            Mutex::new(f)
        })
    });
    if let Some(m) = slot {
        let n = dist.len();
        let mut buf = String::new();
        for i in 0..n {
            for j in (i + 1)..n {
                buf.push_str(&format!("{}\n", dist[i][j]));
            }
        }
        if !buf.is_empty() {
            let _ = m.lock().unwrap().write_all(buf.as_bytes());
        }
    }
}

/// Nearest cluster (by mean KL distance from the user's global model to the cluster's decks) — the
/// home for a deck that appears in this split's test set but had no training rows.
fn nearest_cluster_kl(global_dist: &[f64], labels: &[usize], nclusters: usize) -> usize {
    let mut best = (0usize, f64::INFINITY);
    for c in 0..nclusters {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for (i, &l) in labels.iter().enumerate() {
            if l == c {
                sum += global_dist[i];
                cnt += 1;
            }
        }
        let m = if cnt > 0 { sum / cnt as f64 } else { f64::INFINITY };
        if m < best.1 {
            best = (c, m);
        }
    }
    best.0
}

/// A single clustering experiment's spec: hierarchical linkage (method, threshold) or HDBSCAN
/// (min_cluster_size, min_samples, leaf-vs-eom). Produced by the sweep matrices below.
#[derive(Clone, Copy)]
enum SmartCluster {
    Hier(crate::cluster::Method, f64),
    Hdbscan { mcs: usize, ms: usize, leaf: bool },
}

/// Partition the (whitened) deck vectors per the spec → one 0-based cluster label per deck. HDBSCAN
/// noise decks are reassigned to the nearest cluster (the prototype's NOISE_HANDLING="nearest").
fn cluster_decks(points: &[Vec<f64>], spec: &SmartCluster) -> Vec<usize> {
    match spec {
        SmartCluster::Hier(method, threshold) => {
            crate::cluster::fcluster_distance(points, *method, *threshold)
        }
        SmartCluster::Hdbscan { mcs, ms, leaf } => {
            let raw = crate::hdbscan::hdbscan(points, *mcs, *ms, *leaf);
            crate::hdbscan::noise_to_nearest(points, &raw)
        }
    }
}

/// Same as [`cluster_decks`] but from a precomputed KL distance matrix (the `--cluster_distance kl`
/// path); HDBSCAN noise decks are reassigned to the nearest cluster by mean KL distance.
fn cluster_decks_dist(dist: &[Vec<f64>], spec: &SmartCluster) -> Vec<usize> {
    match spec {
        SmartCluster::Hier(method, threshold) => {
            crate::cluster::fcluster_distance_matrix(dist, *method, *threshold)
        }
        SmartCluster::Hdbscan { mcs, ms, leaf } => {
            let raw = crate::hdbscan::hdbscan_precomputed(dist, *mcs, *ms, *leaf);
            crate::hdbscan::noise_to_nearest_dist(dist, &raw)
        }
    }
}

/// One clustering experiment on a precomputed [`DeckSplit`]: cluster decks, train one param set per
/// cluster ("smart preset"), predict this split's test rows (appending to `eval_rows`/`p`). A deck
/// with test rows but no train rows that split is assigned to the cluster nearest (whitened) to the
/// user's global params. Returns this split's per-cluster params (`{cluster: weights}`).
fn smart_predict_split(
    ds: &Dataset,
    cfg: &Config,
    tc: &TrainConfig,
    cov: &crate::smart::Cov,
    d: &DeckSplit,
    train: &[Row],
    test: &[Row],
    spec: &SmartCluster,
    eval_rows: &mut Vec<Row>,
    p: &mut Vec<f64>,
) -> Vec<(String, Vec<f64>)> {
    use std::collections::HashMap;
    let max_seq = cfg.max_seq_len;
    let init_w = INIT_W.to_vec();

    let labels = match &d.kl {
        Some(kl) => cluster_decks_dist(&kl.dist, spec),
        None => cluster_decks(&d.points, spec),
    };
    let nclusters = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let deck2cluster: HashMap<i64, usize> =
        d.deck_ids.iter().zip(&labels).map(|(&id, &l)| (id, l)).collect();

    // Per-cluster training. Group train rows by their deck's cluster; an inadequate cluster falls
    // back to the user's global params, then INIT_W.
    let mut cluster_rows: Vec<Vec<Row>> = vec![Vec::new(); nclusters];
    for r in train {
        if let Some(&c) = deck2cluster.get(&r.partition) {
            cluster_rows[c].push(r.clone());
        }
    }
    let cluster_w: Vec<Vec<f64>> = cluster_rows
        .iter()
        .map(|crows| {
            if cfg.default_params {
                init_w.clone()
            } else if adequate(crows, max_seq) {
                train_weights(ds, crows, cfg, tc)
            } else {
                d.global_w.clone().unwrap_or_else(|| init_w.clone())
            }
        })
        .collect();

    // Nearest cluster to the user's global model — the home for test-only decks. KL path: mean KL
    // from the global model to the cluster's decks. Mahalanobis path: nearest whitened centroid.
    let nearest_c: Option<usize> = if nclusters == 0 {
        None
    } else {
        match &d.kl {
            Some(kl) => {
                kl.global_dist.as_ref().map(|gd| nearest_cluster_kl(gd, &labels, nclusters))
            }
            None => d.global_w.as_ref().map(|gw| {
                let dim = cov.dim();
                let mut centroids = vec![vec![0.0f64; dim]; nclusters];
                let mut counts = vec![0usize; nclusters];
                for (i, &c) in labels.iter().enumerate() {
                    for k in 0..dim {
                        centroids[c][k] += d.points[i][k];
                    }
                    counts[c] += 1;
                }
                for c in 0..nclusters {
                    let n = counts[c].max(1) as f64;
                    for k in 0..dim {
                        centroids[c][k] /= n;
                    }
                }
                let z = cov.whiten(gw);
                (0..nclusters)
                    .min_by(|&a, &b| sq_dist(&z, &centroids[a]).total_cmp(&sq_dist(&z, &centroids[b])))
                    .unwrap()
            }),
        }
    };

    // Predict: group test rows by assigned cluster (sentinel `nclusters` = INIT_W).
    let mut groups: HashMap<usize, Vec<Row>> = HashMap::new();
    for r in test {
        let c = match deck2cluster.get(&r.partition) {
            Some(&c) => c,
            None => nearest_c.unwrap_or(nclusters),
        };
        groups.entry(c).or_default().push(r.clone());
    }
    let mut keys: Vec<usize> = groups.keys().copied().collect();
    keys.sort_unstable();
    for c in keys {
        let group = &groups[&c];
        let w = if c < nclusters { &cluster_w[c] } else { &init_w };
        let tm = Model::build(ds, group, &vec![1.0; group.len()], None, cfg);
        let all: Vec<usize> = (0..tm.rows.len()).collect();
        for (i, pr) in tm.predict(w, &all).into_iter().enumerate() {
            eval_rows.push(tm.rows[i].clone());
            p.push(pr);
        }
    }
    (0..nclusters).map(|c| (c.to_string(), cluster_w[c].clone())).collect()
}

/// `--partitions smart`: smart-preset assignment for one (method, threshold). Eval row-set == the
/// non-partitioned run ⇒ `size` matches.
fn process_smart(ds: &Dataset, cfg: &Config, tc: &TrainConfig) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let cov = crate::smart::global();
    let method =
        crate::cluster::Method::parse(&cfg.cluster_method).unwrap_or(crate::cluster::Method::Ward);
    let spec = SmartCluster::Hier(method, cfg.cluster_threshold);

    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_pw: Vec<(String, Vec<f64>)> = Vec::new();
    for s in splits {
        let train = &rows[..s.test_start];
        let test = &rows[s.test_start..s.test_end];
        let d = compute_deck_split(ds, cfg, tc, cov, train, test);
        last_pw = smart_predict_split(ds, cfg, tc, cov, &d, train, test, &spec, &mut eval_rows, &mut p);
    }
    ModelOutput { eval_rows, p, params: Params::Partitioned(last_pw) }
}

/// Hierarchical sweep matrix (xlsx exps 1-30): 5 linkages × 6 thresholds, method outer.
const SWEEP_METHODS: [(&str, crate::cluster::Method); 5] = [
    ("single", crate::cluster::Method::Single),
    ("complete", crate::cluster::Method::Complete),
    ("average", crate::cluster::Method::Average),
    ("centroid", crate::cluster::Method::Centroid),
    ("ward", crate::cluster::Method::Ward),
];
const SWEEP_THRESHOLDS: [f64; 6] = [1.5, 2.0, 3.0, 5.0, 7.5, 12.0];

/// Hierarchical thresholds for the KL-divergence metric (`--cluster_distance kl`). The Mahalanobis
/// thresholds above are on a different scale, so these are calibrated from the observed pairwise-KL
/// distribution (see `_smart/kl_calibrate.py`) to span all-singleton → all-merged.
// Calibrated from 1.14M observed deck-pair distances (median 0.007, p90 0.019, p99 0.045): span
// fine (~p13) → collapse (~1 cluster = baseline). See `_smart/kl_calibrate.py`.
const KL_SWEEP_THRESHOLDS: [f64; 6] = [0.001, 0.003, 0.008, 0.02, 0.05, 0.15];

/// HDBSCAN sweep matrix (16 experiments): min_cluster_size × min_samples × {eom, leaf}.
const HDBSCAN_MCS: [usize; 4] = [2, 5, 10, 20];
const HDBSCAN_MS: [usize; 2] = [1, 5];

/// Filename suffixes for the hierarchical sweep, in run order (`[kl-]<method>-<threshold>`). `kl`
/// uses the KL thresholds and a `kl-` prefix so those files sit beside the mahalanobis ones.
pub fn hier_sweep_suffixes(kl: bool) -> Vec<String> {
    let prefix = if kl { "kl-" } else { "" };
    let thr: &[f64] = if kl { &KL_SWEEP_THRESHOLDS } else { &SWEEP_THRESHOLDS };
    let mut v = Vec::new();
    for (mname, _) in SWEEP_METHODS {
        for &t in thr {
            v.push(format!("{prefix}{mname}-{}", crate::config::fmt_threshold(t)));
        }
    }
    v
}

/// Filename suffixes for the HDBSCAN sweep, in run order (`[kl-]hdbscan-mcs<M>-ms<S>-<eom|leaf>`).
/// The mcs/ms grid is scale-free, so `kl` only changes the `kl-` prefix (and the geometry used).
pub fn hdbscan_sweep_suffixes(kl: bool) -> Vec<String> {
    let prefix = if kl { "kl-" } else { "" };
    let mut v = Vec::new();
    for mcs in HDBSCAN_MCS {
        for ms in HDBSCAN_MS {
            for leaf in [false, true] {
                v.push(format!(
                    "{prefix}hdbscan-mcs{mcs}-ms{ms}-{}",
                    if leaf { "leaf" } else { "eom" }
                ));
            }
        }
    }
    v
}

/// Run a list of clustering experiments for one user, sharing the per-deck training across them.
/// Returns `(ModelOutput, time_s)` per spec in input order; `time_s` attributes the shared deck
/// cost evenly across the experiments.
fn sweep_specs(ds: &Dataset, cfg: &Config, specs: &[SmartCluster]) -> Vec<(ModelOutput, f64)> {
    use std::time::Instant;
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let cov = crate::smart::global();
    let tc = &train_config();

    // Shared per-deck training for every split (the expensive step), timed once.
    let t_deck = Instant::now();
    let dsplits: Vec<(usize, usize, DeckSplit)> = splits
        .iter()
        .map(|s| {
            let train = &rows[..s.test_start];
            let test = &rows[s.test_start..s.test_end];
            (s.test_start, s.test_end, compute_deck_split(ds, cfg, tc, cov, train, test))
        })
        .collect();
    let deck_share = t_deck.elapsed().as_secs_f64() / specs.len().max(1) as f64;

    specs
        .iter()
        .map(|spec| {
            let t0 = Instant::now();
            let mut eval_rows = Vec::new();
            let mut p = Vec::new();
            let mut last_pw: Vec<(String, Vec<f64>)> = Vec::new();
            for (ts, te, d) in &dsplits {
                let train = &rows[..*ts];
                let test = &rows[*ts..*te];
                last_pw = smart_predict_split(ds, cfg, tc, cov, d, train, test, spec, &mut eval_rows, &mut p);
            }
            let time_s = deck_share + t0.elapsed().as_secs_f64();
            (ModelOutput { eval_rows, p, params: Params::Partitioned(last_pw) }, time_s)
        })
        .collect()
}

/// `--partitions smart --cluster_sweep`: the 30 hierarchical experiments (xlsx order), sharing
/// per-deck training.
pub fn process_smart_sweep(ds: &Dataset, cfg: &Config) -> Vec<(ModelOutput, f64)> {
    let thr: &[f64] =
        if cfg.cluster_distance == "kl" { &KL_SWEEP_THRESHOLDS } else { &SWEEP_THRESHOLDS };
    let specs: Vec<SmartCluster> = SWEEP_METHODS
        .iter()
        .flat_map(|&(_, m)| thr.iter().map(move |&t| SmartCluster::Hier(m, t)))
        .collect();
    sweep_specs(ds, cfg, &specs)
}

/// `--partitions smart --cluster_method hdbscan --cluster_sweep`: the 16 HDBSCAN experiments.
pub fn process_hdbscan_sweep(ds: &Dataset, cfg: &Config) -> Vec<(ModelOutput, f64)> {
    let mut specs = Vec::new();
    for mcs in HDBSCAN_MCS {
        for ms in HDBSCAN_MS {
            for leaf in [false, true] {
                specs.push(SmartCluster::Hdbscan { mcs, ms, leaf });
            }
        }
    }
    sweep_specs(ds, cfg, &specs)
}

// ===================== Optimal-partition (objective-driven) smart presets =====================
//
// Instead of clustering decks by a distance, search the space of deck *partitions* directly for the
// one minimizing an information criterion (AIC/BIC) on the TRAINING fold — no test peeking. Tiers by
// the number of decks N: N≤6 exhaustive over all set partitions (memoized, 2^N−1 distinct subset
// trainings cover them); 6<N≤12 greedy agglomerative merge on the objective; N>12 pre-merge the
// closest decks (Mahalanobis or KL) into 12 pseudo-decks, then greedy. The coarsest partition (one
// model over all decks) == the per-user global baseline and is always a candidate, so this directly
// tests whether any honestly-selected partition beats global. AIC and BIC share every subset
// training (only the penalty differs); a fallback cluster (too little data → reuses the global model
// or INIT_W) does NOT add to the parameter count k.

#[derive(Clone, Copy, PartialEq)]
enum OptObj {
    Aic,
    Bic,
}

#[derive(Clone, Copy, PartialEq)]
enum PreMerge {
    Maha,
    Kl,
}

/// A fitted cluster (memoized by its base-unit bitmask).
struct SubsetFit {
    params: Vec<f64>,
    nll: f64,      // Σ clamped BCE over the cluster's training rows under `params`
    n_rows: usize, // predicted training rows
    trained: bool, // params estimated from this cluster's own data (false = fallback → no k cost)
}

/// Clamped BCE, matching `train.rs::bce` (each log term floored at −100, like torch's
/// `binary_cross_entropy`).
#[inline]
fn opt_bce(p: f64, y: f64) -> f64 {
    -(y * p.ln().max(-100.0) + (1.0 - y) * (1.0 - p).ln().max(-100.0))
}

/// Base units for one fold (≤12): per-unit training rows + the original-deck → unit map.
struct OptBases {
    unit_rows: Vec<Vec<Row>>,
    deck_to_unit: std::collections::HashMap<i64, usize>,
}

/// Build the ≤12 base units for one fold: each deck if N≤12, else pre-merge the closest decks
/// (Mahalanobis on whitened params, or KL on predictions) down to 12 pseudo-decks.
fn build_opt_bases(
    ds: &Dataset,
    cfg: &Config,
    tc: &TrainConfig,
    cov: &crate::smart::Cov,
    train: &[Row],
    metric: PreMerge,
) -> OptBases {
    use std::collections::HashMap;
    let mut deck_ids: Vec<i64> = train.iter().map(|r| r.partition).collect();
    deck_ids.sort_unstable();
    deck_ids.dedup();
    let m = deck_ids.len();
    let rows_of = |d: i64| -> Vec<Row> { train.iter().filter(|r| r.partition == d).cloned().collect() };

    if m <= 12 {
        let unit_rows = deck_ids.iter().map(|&d| rows_of(d)).collect();
        let deck_to_unit = deck_ids.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        return OptBases { unit_rows, deck_to_unit };
    }

    // N>12: per-deck training for the pre-merge distance, then merge to 12 pseudo-decks.
    let (dids, deck_w, _ul) = train_partition_weights(ds, train, cfg, tc);
    let labels: Vec<usize> = match metric {
        PreMerge::Maha => {
            let points: Vec<Vec<f64>> = dids.iter().map(|d| cov.whiten(&deck_w[d])).collect();
            crate::cluster::fcluster_k_points(&points, crate::cluster::Method::Average, 12)
        }
        PreMerge::Kl => {
            let tm = Model::build(ds, train, &vec![1.0; train.len()], None, cfg);
            let idx = kl_eval_indices(tm.rows.len(), KL_DIST_CAP);
            let prep: Vec<(Vec<f64>, Vec<f64>)> =
                dids.iter().map(|d| kl_prep(&tm.predict(&deck_w[d], &idx))).collect();
            let nd = dids.len();
            let mut dist = vec![vec![0.0f64; nd]; nd];
            for i in 0..nd {
                for j in (i + 1)..nd {
                    let v = sym_kl(&prep[i].0, &prep[i].1, &prep[j].0, &prep[j].1);
                    dist[i][j] = v;
                    dist[j][i] = v;
                }
            }
            crate::cluster::fcluster_k_matrix(&dist, crate::cluster::Method::Average, 12)
        }
    };
    let k = labels.iter().copied().max().map(|x| x + 1).unwrap_or(0);
    let mut unit_rows = vec![Vec::new(); k];
    let mut deck_to_unit = HashMap::new();
    for (i, &d) in dids.iter().enumerate() {
        let u = labels[i];
        deck_to_unit.insert(d, u);
        unit_rows[u].extend(rows_of(d));
    }
    OptBases { unit_rows, deck_to_unit }
}

/// Fit (memoized) the cluster selected by `mask` over the base units' rows.
fn fit_subset(
    ds: &Dataset,
    cfg: &Config,
    tc: &TrainConfig,
    unit_rows: &[Vec<Row>],
    mask: u16,
    global_w: Option<&[f64]>,
    memo: &mut std::collections::HashMap<u16, SubsetFit>,
) {
    if memo.contains_key(&mask) {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for (i, ur) in unit_rows.iter().enumerate() {
        if mask & (1 << i) != 0 {
            rows.extend_from_slice(ur);
        }
    }
    let (params, trained) = if cfg.default_params {
        (INIT_W.to_vec(), false)
    } else if adequate(&rows, cfg.max_seq_len) {
        (train_weights(ds, &rows, cfg, tc), true)
    } else {
        (global_w.map(|g| g.to_vec()).unwrap_or_else(|| INIT_W.to_vec()), false)
    };
    let tm = Model::build(ds, &rows, &vec![1.0; rows.len()], None, cfg);
    let all: Vec<usize> = (0..tm.rows.len()).collect();
    let preds = tm.predict(&params, &all);
    let nll: f64 = preds.iter().enumerate().map(|(i, &pr)| opt_bce(pr, tm.rows[i].y as f64)).sum();
    memo.insert(mask, SubsetFit { params, nll, n_rows: tm.rows.len(), trained });
}

/// AIC/BIC of a partition (list of cluster masks) from cached fits. `n` = total training rows.
/// Fallback clusters contribute their NLL but not to `k` (per Andrew: don't penalize fallback).
fn partition_objective(
    masks: &[u16],
    memo: &std::collections::HashMap<u16, SubsetFit>,
    obj: OptObj,
    n: usize,
) -> f64 {
    let mut nll = 0.0;
    let mut ntrained = 0usize;
    for &m in masks {
        let f = &memo[&m];
        nll += f.nll;
        if f.trained {
            ntrained += 1;
        }
    }
    let k = (ntrained * INIT_W.len()) as f64;
    match obj {
        OptObj::Aic => 2.0 * nll + 2.0 * k,
        OptObj::Bic => 2.0 * nll + k * (n.max(1) as f64).ln(),
    }
}

/// All set partitions of `m` units as lists of base-unit bitmasks (restricted-growth strings).
fn set_partitions(m: usize) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    fn rec(i: usize, nb: usize, a: &mut [usize], m: usize, out: &mut Vec<Vec<u16>>) {
        if i == m {
            let mut masks = vec![0u16; nb];
            for (u, &lab) in a.iter().enumerate() {
                masks[lab] |= 1u16 << u;
            }
            out.push(masks);
            return;
        }
        for lab in 0..nb {
            a[i] = lab;
            rec(i + 1, nb, a, m, out);
        }
        a[i] = nb;
        rec(i + 1, nb + 1, a, m, out);
    }
    if m == 0 {
        return vec![vec![]];
    }
    let mut a = vec![0usize; m];
    rec(0, 0, &mut a, m, &mut out);
    out
}

/// Exhaustive search over all set partitions (m≤6): returns (best-BIC masks, best-AIC masks).
fn opt_exhaustive(
    ds: &Dataset,
    cfg: &Config,
    tc: &TrainConfig,
    unit_rows: &[Vec<Row>],
    global_w: &[f64],
    memo: &mut std::collections::HashMap<u16, SubsetFit>,
    n: usize,
) -> (Vec<u16>, Vec<u16>) {
    let m = unit_rows.len();
    for mask in 1u16..(1u16 << m) {
        fit_subset(ds, cfg, tc, unit_rows, mask, Some(global_w), memo);
    }
    let (mut bb, mut ba) = ((f64::INFINITY, Vec::new()), (f64::INFINITY, Vec::new()));
    for masks in set_partitions(m) {
        let bic = partition_objective(&masks, memo, OptObj::Bic, n);
        let aic = partition_objective(&masks, memo, OptObj::Aic, n);
        if bic < bb.0 {
            bb = (bic, masks.clone());
        }
        if aic < ba.0 {
            ba = (aic, masks);
        }
    }
    (bb.1, ba.1)
}

/// Greedy agglomerative search for one objective (6<m≤12): from singletons, repeatedly commit the
/// merge that most lowers `obj`, walking to one cluster; return the best partition seen on the path.
fn opt_greedy(
    ds: &Dataset,
    cfg: &Config,
    tc: &TrainConfig,
    unit_rows: &[Vec<Row>],
    global_w: &[f64],
    memo: &mut std::collections::HashMap<u16, SubsetFit>,
    obj: OptObj,
    n: usize,
) -> Vec<u16> {
    let m = unit_rows.len();
    let mut clusters: Vec<u16> = (0..m).map(|i| 1u16 << i).collect();
    for &c in &clusters {
        fit_subset(ds, cfg, tc, unit_rows, c, Some(global_w), memo);
    }
    let mut best = clusters.clone();
    let mut best_obj = partition_objective(&clusters, memo, obj, n);
    while clusters.len() > 1 {
        let (mut bi, mut bj, mut bmask, mut bval) = (0usize, 1usize, 0u16, f64::INFINITY);
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let merged = clusters[i] | clusters[j];
                fit_subset(ds, cfg, tc, unit_rows, merged, Some(global_w), memo);
                let cand: Vec<u16> = clusters
                    .iter()
                    .enumerate()
                    .filter(|(t, _)| *t != i && *t != j)
                    .map(|(_, &c)| c)
                    .chain(std::iter::once(merged))
                    .collect();
                let val = partition_objective(&cand, memo, obj, n);
                if val < bval {
                    bval = val;
                    bi = i;
                    bj = j;
                    bmask = merged;
                }
            }
        }
        clusters = clusters
            .iter()
            .enumerate()
            .filter(|(t, _)| *t != bi && *t != bj)
            .map(|(_, &c)| c)
            .chain(std::iter::once(bmask))
            .collect();
        if bval < best_obj {
            best_obj = bval;
            best = clusters.clone();
        }
    }
    best
}

type OptOut = (Vec<Row>, Vec<f64>, Vec<(String, Vec<f64>)>);

/// Predict one fold's `test` rows under a selected partition, appending to (eval_rows, p). A deck
/// with no training rows that fold (test-only) is predicted with the global model.
fn opt_predict(
    ds: &Dataset,
    cfg: &Config,
    bases: &OptBases,
    masks: &[u16],
    memo: &std::collections::HashMap<u16, SubsetFit>,
    global_w: &[f64],
    test: &[Row],
    eval_rows: &mut Vec<Row>,
    p: &mut Vec<f64>,
) {
    use std::collections::HashMap;
    let m = bases.unit_rows.len();
    let mut unit_mask = vec![0u16; m];
    for &mask in masks {
        for (u, slot) in unit_mask.iter_mut().enumerate() {
            if mask & (1 << u) != 0 {
                *slot = mask;
            }
        }
    }
    let mut groups: HashMap<u16, Vec<Row>> = HashMap::new();
    for r in test {
        let key = match bases.deck_to_unit.get(&r.partition) {
            Some(&u) => unit_mask[u],
            None => 0u16, // test-only deck → global (real cluster masks are ≥1)
        };
        groups.entry(key).or_default().push(r.clone());
    }
    let mut keys: Vec<u16> = groups.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let group = &groups[&key];
        let w: &[f64] = if key == 0 { global_w } else { &memo[&key].params };
        let tm = Model::build(ds, group, &vec![1.0; group.len()], None, cfg);
        let all: Vec<usize> = (0..tm.rows.len()).collect();
        for (i, pr) in tm.predict(w, &all).into_iter().enumerate() {
            eval_rows.push(tm.rows[i].clone());
            p.push(pr);
        }
    }
}

fn masks_to_params(
    masks: &[u16],
    memo: &std::collections::HashMap<u16, SubsetFit>,
) -> Vec<(String, Vec<f64>)> {
    masks.iter().enumerate().map(|(i, m)| (i.to_string(), memo[m].params.clone())).collect()
}

/// Run the optimal-partition search for one pre-merge metric across all folds → (BIC out, AIC out).
fn optimal_one_metric(ds: &Dataset, cfg: &Config, tc: &TrainConfig, metric: PreMerge) -> (OptOut, OptOut) {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let cov = crate::smart::global();
    let (mut erb, mut pb, mut era, mut pa) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut lb, mut la): (Vec<(String, Vec<f64>)>, Vec<(String, Vec<f64>)>) = (Vec::new(), Vec::new());
    for s in splits {
        let train = &rows[..s.test_start];
        let test = &rows[s.test_start..s.test_end];
        let bases = build_opt_bases(ds, cfg, tc, cov, train, metric);
        let m = bases.unit_rows.len();
        if m == 0 {
            continue;
        }
        let full: u16 = ((1u32 << m) - 1) as u16;
        let mut memo: std::collections::HashMap<u16, SubsetFit> = std::collections::HashMap::new();
        fit_subset(ds, cfg, tc, &bases.unit_rows, full, None, &mut memo);
        let global_w = memo[&full].params.clone();
        let n = memo[&full].n_rows;
        let (bic_masks, aic_masks) = if m <= 6 {
            opt_exhaustive(ds, cfg, tc, &bases.unit_rows, &global_w, &mut memo, n)
        } else {
            let b = opt_greedy(ds, cfg, tc, &bases.unit_rows, &global_w, &mut memo, OptObj::Bic, n);
            let a = opt_greedy(ds, cfg, tc, &bases.unit_rows, &global_w, &mut memo, OptObj::Aic, n);
            (b, a)
        };
        opt_predict(ds, cfg, &bases, &bic_masks, &memo, &global_w, test, &mut erb, &mut pb);
        opt_predict(ds, cfg, &bases, &aic_masks, &memo, &global_w, test, &mut era, &mut pa);
        lb = masks_to_params(&bic_masks, &memo);
        la = masks_to_params(&aic_masks, &memo);
    }
    ((erb, pb, lb), (era, pa, la))
}

/// `--cluster_method optimal --cluster_sweep`: the 4 optimal-partition configs in suffix order
/// (opt-bic-maha, opt-aic-maha, opt-bic-kl, opt-aic-kl). For N≤12 decks the Maha/KL pre-merge never
/// triggers, so those results are identical and computed once.
pub fn process_optimal_sweep(ds: &Dataset, cfg: &Config) -> Vec<(ModelOutput, f64)> {
    use std::time::Instant;
    let tc = train_config();
    let mut decks: Vec<i64> = ds.rows.iter().map(|r| r.partition).collect();
    decks.sort_unstable();
    decks.dedup();
    let both = decks.len() > 12;
    let t0 = Instant::now();
    let (bm, am) = optimal_one_metric(ds, cfg, &tc, PreMerge::Maha);
    let (bk, ak) = if both { optimal_one_metric(ds, cfg, &tc, PreMerge::Kl) } else { (bm.clone(), am.clone()) };
    let t = t0.elapsed().as_secs_f64() / 4.0;
    let mk = |o: OptOut| ModelOutput { eval_rows: o.0, p: o.1, params: Params::Partitioned(o.2) };
    vec![(mk(bm), t), (mk(am), t), (mk(bk), t), (mk(ak), t)]
}

/// Filename suffixes for the optimal-partition sweep, in `process_optimal_sweep` output order.
pub fn opt_sweep_suffixes() -> Vec<String> {
    vec!["opt-bic-maha".into(), "opt-aic-maha".into(), "opt-bic-kl".into(), "opt-aic-kl".into()]
}

#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sym_kl_matches_reference() {
        // The logit closed-form sym_kl must equal the ln-in-loop reference, incl. at clamp extremes.
        let a = [0.9, 0.5, 0.99, 0.2, 0.7, 0.999999, 0.000001, 0.43];
        let b = [0.8, 0.55, 0.6, 0.25, 0.7, 0.5, 0.5, 0.43];
        let (pa, la) = kl_prep(&a);
        let (pb, lb) = kl_prep(&b);
        let got = sym_kl(&pa, &la, &pb, &lb);
        let want = mean_sym_kl(&a, &b);
        assert!((got - want).abs() < 1e-12, "sym_kl {got} vs reference {want}");
        assert_eq!(sym_kl(&pa, &la, &pa, &la), 0.0, "identical vectors → 0");
    }

    #[test]
    fn set_partitions_counts_bell() {
        // Number of set partitions of an m-set = Bell(m).
        let bell = [1usize, 1, 2, 5, 15, 52, 203];
        for (m, &b) in bell.iter().enumerate() {
            assert_eq!(set_partitions(m).len(), b, "Bell({m})");
        }
        // Every partition must cover all m units exactly once (disjoint masks, union = full).
        for m in 1..=6 {
            let full = (1u16 << m) - 1;
            for masks in set_partitions(m) {
                let mut seen = 0u16;
                for &mm in &masks {
                    assert_eq!(seen & mm, 0, "overlapping clusters at m={m}");
                    seen |= mm;
                    assert_ne!(mm, 0, "empty cluster at m={m}");
                }
                assert_eq!(seen, full, "partition does not cover all units at m={m}");
            }
        }
    }

    // Finite-difference (h=1e-6) needs f64; in the default f32 build the difference is dominated
    // by rounding noise, so this math check runs under the `fp64` feature only.
    #[cfg(feature = "fp64")]
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

// ===================== Hyperparameter probe (`--hp_probe`) =====================
//
// Research-only, no metrics. Anki's "Optimize" fits ONE model on ONE training set starting from
// the default parameters — no warm start, no cross-validation folds. So a per-user hyperparameter
// rule is only deployable if it can be decided from that training set alone. This probe measures
// whether such a rule can exist, without committing to one. Per fold, per candidate it records
// eps-clipped BCE **sums** (so folds pool into the user's LogLoss exactly):
//
//   * `test_full` — trained on 100% of the fold's train rows, scored on the fold's test rows.
//                   `min` over candidates = the (optimistically biased) per-user oracle.
//   * `val`       — trained on the first 80% of the train rows, scored on the last 20%. This is
//                   the ONLY selection signal an Anki-side rule could legally use.
//   * `test_val`  — that same 80%-trained model scored on the fold's test rows; prices the
//                   "select and skip the refit" variant against "select and refit on 100%".
//
// Every candidate trains from `INIT_W`, exactly like a real optimize run.

/// `--hp_probe` candidates: `(name, lr, betas, n_epoch, hyper_beta)` around the FSRS-7 default.
/// The default must stay at index 0 — the analysis scripts treat it as the baseline.
///
/// The `hg*` entries are hypergradient descent on the learning rate (see
/// [`crate::train::train_with_init`]) — they cost the same as `default` (one dot product per step)
/// and need NO selection, so they are the only entries here that are free to deploy.
pub const HP_CANDIDATES: &[(&str, f64, (f64, f64), usize, f64)] = &[
    ("default", 0.0118, (0.70, 0.98), 9, 0.0),
    ("ep20", 0.0118, (0.70, 0.98), 20, 0.0),
    ("ep45", 0.0118, (0.70, 0.98), 45, 0.0),
    ("lr_half", 0.0059, (0.70, 0.98), 9, 0.0),
    ("lr_double", 0.0236, (0.70, 0.98), 9, 0.0),
    ("betas_torch", 0.0118, (0.90, 0.999), 9, 0.0),
    ("betas_low", 0.0118, (0.50, 0.95), 9, 0.0),
    ("lr_double_ep20", 0.0236, (0.70, 0.98), 20, 0.0),
    ("hg0.02", 0.0118, (0.70, 0.98), 9, 0.02),
    ("hg0.05", 0.0118, (0.70, 0.98), 9, 0.05),
    ("hg0.15", 0.0118, (0.70, 0.98), 9, 0.15),
    ("hg0.05_ep20", 0.0118, (0.70, 0.98), 20, 0.05),
];

fn hp_train_config(c: &(&str, f64, (f64, f64), usize, f64)) -> TrainConfig {
    TrainConfig {
        lr: c.1,
        betas: c.2,
        n_epoch: c.3,
        batch_size: 512,
        keep_final: true,
        hyper_beta: c.4,
    }
}

/// eps-clipped BCE summed over `rows` (sklearn `log_loss` numerator; see `metrics::log_loss`).
fn hp_bce_sum(rows: &[Row], p: &[f64]) -> f64 {
    let eps = f64::EPSILON;
    rows.iter()
        .zip(p)
        .map(|(r, &pi)| {
            let pc = pi.clamp(eps, 1.0 - eps);
            if r.y == 1 {
                -pc.ln()
            } else {
                -(1.0 - pc).ln()
            }
        })
        .sum()
}

/// Score `w` on an already-built eval model, returning the BCE sum over its rows.
fn hp_score(tm: &Model, w: &[f64]) -> f64 {
    let all: Vec<usize> = (0..tm.rows.len()).collect();
    hp_bce_sum(&tm.rows, &tm.predict(w, &all))
}

/// The `(train, test)` folds the research modes evaluate on — the non-partitioned branch of
/// [`process`], shared so `--hp_probe` and `--hp_features` cannot drift apart.
fn hp_folds(ds: &Dataset, cfg: &Config) -> Vec<(Vec<Row>, Vec<Row>)> {
    let rows = &ds.rows;
    if let Some(eq) = &ds.equalize_splits {
        eq.iter()
            .map(|sp| {
                let train: Vec<Row> = rows[..sp.train_end].to_vec();
                let test: Vec<Row> = sp.test_idx.iter().map(|&i| rows[i].clone()).collect();
                (train, test)
            })
            .collect()
    } else {
        time_series_split(rows.len(), cfg.n_splits)
            .into_iter()
            .map(|s| (rows[..s.test_start].to_vec(), rows[s.test_start..s.test_end].to_vec()))
            .collect()
    }
}

/// `--hp_probe`: the per-user candidate × fold loss table. See the section comment above.
pub fn process_hp_probe(ds: &Dataset, cfg: &Config) -> serde_json::Value {
    let folds = hp_folds(ds, cfg);
    let mut out_folds = Vec::new();
    for (train, test) in folds {
        let test_model = Model::build(ds, &test, &vec![1.0; test.len()], None, cfg);
        // Inner validation split: the last 20% of the train rows, chronologically. Needs at least
        // one row on each side; tiny folds get no validation signal (`val`/`test_val` = null).
        let cut = train.len() * 4 / 5;
        let inner = if cut >= 1 && train.len() - cut >= 1 {
            let itrain: Vec<Row> = train[..cut].to_vec();
            let ival: Vec<Row> = train[cut..].to_vec();
            let ival_model = Model::build(ds, &ival, &vec![1.0; ival.len()], None, cfg);
            Some((itrain, ival_model))
        } else {
            None
        };

        let mut cands = Vec::new();
        for c in HP_CANDIDATES {
            let tc = hp_train_config(c);
            let wf = {
                let weights = recency_weights_fsrs7(train.len(), cfg.use_recency_weighting);
                let m = Model::build(ds, &train, &weights, Some(cfg.max_seq_len), cfg);
                train::train_with_init(&m, &tc, INIT_W.to_vec())
            };
            let (val, test_val) = match &inner {
                Some((itrain, ival_model)) => {
                    let weights = recency_weights_fsrs7(itrain.len(), cfg.use_recency_weighting);
                    let m = Model::build(ds, itrain, &weights, Some(cfg.max_seq_len), cfg);
                    let wv = train::train_with_init(&m, &tc, INIT_W.to_vec());
                    (
                        serde_json::json!(hp_score(ival_model, &wv)),
                        serde_json::json!(hp_score(&test_model, &wv)),
                    )
                }
                None => (serde_json::Value::Null, serde_json::Value::Null),
            };
            cands.push(serde_json::json!({
                "name": c.0,
                "test_full": hp_score(&test_model, &wf),
                "val": val,
                "test_val": test_val,
            }));
        }

        out_folds.push(serde_json::json!({
            "n_train": train.len(),
            "n_test": test.len(),
            "n_val": inner.as_ref().map(|(_, m)| m.rows.len()).unwrap_or(0),
            "cand": cands,
        }));
    }

    serde_json::json!({ "folds": out_folds })
}

/// `--hp_features`: per-fold summary statistics of each fold's TRAINING rows — the only inputs an
/// Anki-side hyperparameter rule could legally read (Anki's "Optimize" sees one training set and
/// nothing else). Pairs with the `--hp_probe` loss table: same users, same folds, same order, so
/// the two files join on `(user, fold index)`.
///
/// No training happens here, so this costs one feature-building pass — seconds, not the probe's 35x.
///
/// Intervals are summarised as `log1p(delta_t)`: it is defined at 0 and stays ~identity for the
/// short same-day intervals `--secs` produces, so they are not blown into large negative logs (the
/// same reason `Dash::from_rows` stores `ln(feat + 1)`).
pub fn process_hp_features(ds: &Dataset, cfg: &Config) -> serde_json::Value {
    let mut out = Vec::new();
    for (train, test) in hp_folds(ds, cfg) {
        let n = train.len();
        let nf = n.max(1) as f64;

        let mut cards: Vec<i64> = train.iter().map(|r| r.card_id).collect();
        cards.sort_unstable();
        cards.dedup();

        // Button shares. `p_again` is 1 - retention by construction (`features::label` sets
        // y = 0 iff rating == 1), so retention is NOT a separate feature here.
        let mut btn = [0usize; 4];
        for r in &train {
            if (1..=4).contains(&r.rating) {
                btn[(r.rating - 1) as usize] += 1;
            }
        }

        let mut ldt: Vec<f64> = train.iter().map(|r| r.delta_t.max(0.0).ln_1p()).collect();
        ldt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if ldt.is_empty() { 0.0 } else { ldt[ldt.len() / 2] };
        let mean = ldt.iter().sum::<f64>() / nf;
        let sd = (ldt.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nf).sqrt();

        let same_day = train.iter().filter(|r| r.elapsed_days == 0).count() as f64 / nf;
        let mean_pos = train.iter().map(|r| r.pos as f64).sum::<f64>() / nf;

        out.push(serde_json::json!({
            "n_train": n,
            "n_test": test.len(),
            "n_cards": cards.len(),
            "reviews_per_card": nf / cards.len().max(1) as f64,
            "batches": n.div_ceil(512),
            "p_again": btn[0] as f64 / nf,
            "p_hard": btn[1] as f64 / nf,
            "p_good": btn[2] as f64 / nf,
            "p_easy": btn[3] as f64 / nf,
            "median_log1p_dt": median,
            "mean_log1p_dt": mean,
            "sd_log1p_dt": sd,
            "same_day_share": same_day,
            "mean_pos": mean_pos,
        }));
    }
    serde_json::json!({ "folds": out })
}

/// `--reopt_growth eps`: retrain from the default parameters whenever the training set has grown
/// by a factor `(1 + eps)`, instead of only at the 5 `TimeSeriesSplit` boundaries. This is an upper
/// bound on what a user could reach by re-optimizing often — every prediction is then made by a
/// model that has seen at least `1/(1+eps)` of the history available to it, versus 50%-83% for the
/// 5-fold schedule.
///
/// **The evaluated row set is unchanged.** `time_series_split` pools test folds covering exactly
/// `rows[eval_start..]` with `eval_start = n - n_splits * (n / (n_splits + 1))`, and the geometric
/// schedule partitions that same range. So `size` is identical per user and in the sum, and the
/// metrics are directly comparable to a normal run (rule #6 stays checkable the cheap way).
///
/// Cost is `~n*(1+eps)/eps` training rows against the 5-fold `2.5n`, so `eps = 0.006` is ~67x a
/// normal run. Training prefixes are passed as SLICES, never cloned — at ~1500 retrains per user
/// the copies would otherwise dominate.
fn process_geometric(ds: &Dataset, cfg: &Config, tc: &TrainConfig) -> ModelOutput {
    let rows = &ds.rows;
    let n = rows.len();
    let test_size = n / (cfg.n_splits + 1);
    let eval_start = n - cfg.n_splits * test_size;
    let growth = 1.0 + cfg.reopt_growth;

    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = INIT_W.to_vec();

    let mut t = eval_start;
    while t < n {
        // Next retrain point: the training set must grow by at least one row, and by at least the
        // requested factor. `min(n)` keeps the last chunk inside the evaluated range.
        let next = (((t as f64) * growth).floor() as usize).max(t + 1).min(n);
        let w = train_weights(ds, &rows[..t], cfg, tc);
        let test = &rows[t..next];
        let tm = Model::build(ds, test, &vec![1.0; test.len()], None, cfg);
        let all: Vec<usize> = (0..tm.rows.len()).collect();
        for (i, pr) in tm.predict(&w, &all).into_iter().enumerate() {
            eval_rows.push(tm.rows[i].clone());
            p.push(pr);
        }
        last_w = w;
        t = next;
    }

    ModelOutput { eval_rows, p, params: Params::Partitioned(vec![("0".to_string(), last_w)]) }
}
