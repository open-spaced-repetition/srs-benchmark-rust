//! FSRS-6-one-step — `models/fsrs_v6_one_step.py`.
//!
//! An *online* variant of FSRS-6: after the standard S0 initialization, it makes a single
//! pass over the training reviews (in `review_th` order) and, for each review, takes ONE
//! SGD step (lr = 1e-4) on a hand-derived gradient that backpropagates through only the
//! most-recent state transition (hence "one-step"). Predictions use stock FSRS-6 with the
//! resulting weights, so the eval row-set — and `size` — is identical to plain FSRS-6.
//!
//! The training-time forward (`step` below) is the model's own simplified FSRS-6 recurrence
//! (S_MIN floor, no s_max cap, `rating >= 3` short-term branch, no failure-stability floor);
//! it must match the Python byte-for-byte because it determines the SGD trajectory. The
//! analytic `backward` is ported line-for-line from the Python.

use super::fsrs_init::fit_s0_from_x0;
use super::fsrs_v6;
use super::{ModelOutput};
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;

const NP: usize = 21;
const S_MIN: f64 = 0.001; // hardcoded in the Python step/init clamps (NOT cfg.s_min)
const LR: f64 = 1e-4;

/// Forgetting curve with trained decay (`decay = -w[20]`): `(1 + factor·t/s)^decay`.
fn forgetting_curve(t: f64, s: f64, w: &[f64]) -> f64 {
    let decay = -w[20];
    let factor = 0.9f64.powf(1.0 / decay) - 1.0;
    (1.0 + factor * t / s).powf(decay)
}

fn init_stability(rating: i64, w: &[f64]) -> f64 {
    S_MIN.max(w[(rating - 1) as usize])
}
fn init_difficulty(rating: i64, w: &[f64]) -> f64 {
    (w[4] - (w[5] * (rating as f64 - 1.0)).exp() + 1.0).clamp(1.0, 10.0)
}
fn next_difficulty(last_d: f64, rating: i64, w: &[f64]) -> f64 {
    let init_d_4 = w[4] - (w[5] * 3.0).exp() + 1.0;
    let delta_d = -w[6] * (rating as f64 - 3.0);
    let linear_damping = delta_d * (10.0 - last_d) / 9.0;
    let d_intermediate = last_d + linear_damping;
    let new_d = w[7] * init_d_4 + (1.0 - w[7]) * d_intermediate;
    new_d.clamp(1.0, 10.0)
}
fn stability_short_term(s: f64, rating: i64, w: &[f64]) -> f64 {
    if s <= 0.0 {
        return S_MIN;
    }
    let sinc = (w[17] * (rating as f64 - 3.0 + w[18])).exp() * s.powf(-w[19]);
    let new_s = s * (if rating >= 3 { sinc.max(1.0) } else { sinc });
    S_MIN.max(new_s)
}
fn stability_after_success(last_s: f64, last_d: f64, last_r: f64, rating: i64, w: &[f64]) -> f64 {
    let hard_penalty = if rating == 2 { w[15] } else { 1.0 };
    let easy_bonus = if rating == 4 { w[16] } else { 1.0 };
    let new_s = last_s
        * (1.0
            + w[8].exp()
                * (11.0 - last_d)
                * last_s.powf(-w[9])
                * (((1.0 - last_r) * w[10]).exp() - 1.0)
                * hard_penalty
                * easy_bonus);
    S_MIN.max(new_s)
}
fn stability_after_failure(last_s: f64, last_d: f64, last_r: f64, w: &[f64]) -> f64 {
    let new_s = w[11]
        * last_d.powf(-w[12])
        * ((last_s + 1.0).powf(w[13]) - 1.0)
        * ((1.0 - last_r) * w[14]).exp();
    S_MIN.max(new_s)
}

/// One recurrence step: returns the new `(stability, difficulty)` from the prior state.
fn step(delta_t: f64, rating: i64, last: Option<(f64, f64)>, w: &[f64]) -> (f64, f64) {
    match last {
        None => (init_stability(rating, w), init_difficulty(rating, w)),
        Some((last_s, last_d)) => {
            if delta_t < 1.0 {
                (stability_short_term(last_s, rating, w), next_difficulty(last_d, rating, w))
            } else {
                let new_d = next_difficulty(last_d, rating, w);
                let r = forgetting_curve(delta_t, last_s, w);
                let new_s = if rating == 1 {
                    stability_after_failure(last_s, new_d, r, w)
                } else {
                    stability_after_success(last_s, new_d, r, rating, w)
                };
                (new_s, new_d)
            }
        }
    }
}

/// State extracted from a forward pass over a card's prior `(delta_t, rating)` history.
struct Fwd {
    new_s: f64,
    new_d: f64,
    last: Option<(f64, f64)>, // (s, d) after the second-to-last prior
    last_delta_t: f64,        // last prior's interval
    last_rating: i64,         // last prior's rating
}

/// Replay the prior history, returning the final two states + the last prior's (Δt, rating).
/// `priors` must be non-empty (callers skip pos==0 rows, which have no priors).
fn forward(prior_dt: &[f64], prior_r: &[i64], w: &[f64]) -> Fwd {
    let mut outputs: Vec<(f64, f64)> = Vec::with_capacity(prior_dt.len());
    let mut last: Option<(f64, f64)> = None;
    for k in 0..prior_dt.len() {
        let cur = step(prior_dt[k], prior_r[k], last, w);
        outputs.push(cur);
        last = Some(cur);
    }
    let n = outputs.len();
    Fwd {
        new_s: outputs[n - 1].0,
        new_d: outputs[n - 1].1,
        last: if n > 1 { Some(outputs[n - 2]) } else { None },
        last_delta_t: prior_dt[n - 1],
        last_rating: prior_r[n - 1],
    }
}

/// One SGD step on the current review `(delta_t, y)` given the forward state `f`. Mutates `w`.
/// Direct port of `FSRS_one_step.backward`.
fn backward(delta_t: f64, y: f64, f: &Fwd, w: &mut [f64]) {
    let mut grad = [0.0f64; NP];
    if f.new_s <= S_MIN {
        return; // no update
    }
    let mut r = forgetting_curve(delta_t, f.new_s, w);
    r = r.clamp(1e-9, 1.0 - 1e-9);
    let dl_dr = (r - y) / (r * (1.0 - r));

    let decay = -w[20];
    let factor = 0.9f64.powf(1.0 / decay) - 1.0;
    let s = f.new_s;
    let dr_ds = decay * (1.0 + factor * delta_t / s).powf(decay - 1.0) * (-factor * delta_t / (s * s));
    let c = dl_dr * dr_ds;
    let rating = f.last_rating;

    match f.last {
        None => {
            grad[(rating - 1) as usize] = c * 100.0;
        }
        Some((last_s, last_d_prev)) => {
            let last_r = forgetting_curve(f.last_delta_t, last_s, w);
            let s = last_s;
            let d = f.new_d;
            let ds_new_d_new;
            if rating == 1 {
                let term1 = d.powf(-w[12]);
                let term3 = ((1.0 - last_r) * w[14]).exp();
                grad[11] = c * (f.new_s / w[11]);
                grad[12] = c * (-f.new_s * d.ln());
                grad[13] = c * (w[11] * term1 * (s + 1.0).powf(w[13]) * (s + 1.0).ln() * term3);
                grad[14] = c * (f.new_s * (1.0 - last_r));
                ds_new_d_new = f.new_s * (-w[12] / d);
            } else {
                let hard_penalty = if rating == 2 { w[15] } else { 1.0 };
                let easy_bonus = if rating == 4 { w[16] } else { 1.0 };
                ds_new_d_new = s
                    * w[8].exp()
                    * (-1.0)
                    * s.powf(-w[9])
                    * (((1.0 - last_r) * w[10]).exp() - 1.0)
                    * hard_penalty
                    * easy_bonus;
                let term_exp_w10 = ((1.0 - last_r) * w[10]).exp();
                let term_s_pow_w9 = s.powf(-w[9]);
                let common_factor =
                    s * w[8].exp() * (11.0 - d) * term_s_pow_w9 * (term_exp_w10 - 1.0);
                grad[8] = c * common_factor * hard_penalty * easy_bonus;
                grad[9] = c * common_factor * (-s.ln()) * hard_penalty * easy_bonus;
                grad[10] = c
                    * s
                    * w[8].exp()
                    * (11.0 - d)
                    * term_s_pow_w9
                    * (term_exp_w10 * (1.0 - last_r))
                    * hard_penalty
                    * easy_bonus;
                if rating == 2 {
                    grad[15] = c * (common_factor * easy_bonus);
                }
                if rating == 4 {
                    grad[16] = c * (common_factor * hard_penalty);
                }
            }
            let last_d = last_d_prev;
            let init_d_4 = w[4] - (w[5] * 3.0).exp() + 1.0;
            let d_intermediate = last_d + (-w[6] * (rating as f64 - 3.0) * (10.0 - last_d) / 9.0);
            grad[4] = c * ds_new_d_new * w[7];
            grad[5] = c * ds_new_d_new * (w[7] * (-(w[5] * 3.0).exp() * 3.0));
            grad[6] = c * ds_new_d_new * ((1.0 - w[7]) * (-(rating as f64 - 3.0) * (10.0 - last_d) / 9.0));
            grad[7] = c * ds_new_d_new * (init_d_4 - d_intermediate);
        }
    }

    // w[20] gradient (always).
    let t = delta_t;
    let s = f.new_s;
    let log_term = (1.0 + factor * t / s).ln();
    let d_factor_d_decay = 0.9f64.powf(1.0 / decay) * 0.9f64.ln() * (-1.0 / (decay * decay));
    let dr_d_decay = r * (log_term + decay * (t / s) * d_factor_d_decay / (1.0 + factor * t / s));
    grad[20] = dl_dr * (-dr_d_decay);

    for i in 0..NP {
        w[i] -= LR * grad[i];
    }
    clamp_weights(w);
}

fn clamp_weights(w: &mut [f64]) {
    for v in w.iter_mut().take(4) {
        *v = v.clamp(S_MIN, 100.0);
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
    w[20] = w[20].clamp(0.1, 0.8);
    // NOTE: w[17], w[18], w[19] are intentionally NOT clamped (matches the Python).
}

/// Train one-step weights on `train`: S0 init, then one online SGD pass over the reviews.
fn train_weights(ds: &Dataset, train: &[Row], cfg: &Config) -> Vec<f64> {
    let mut w = fsrs_v6::INIT_W.to_vec();
    let default_decay = -fsrs_v6::INIT_W[20];
    let s0 = fit_s0_from_x0(
        ds,
        train,
        cfg,
        [fsrs_v6::INIT_W[0], fsrs_v6::INIT_W[1], fsrs_v6::INIT_W[2], fsrs_v6::INIT_W[3]],
        |t, s| {
            let decay = default_decay;
            let factor = 0.9f64.powf(1.0 / decay) - 1.0;
            (1.0 + factor * t / s).powf(decay)
        },
    );
    w[0..4].copy_from_slice(&s0);

    // Online SGD: one pass over training reviews in review_th order (train is already sorted).
    for r in train {
        let delta_t = r.delta_t;
        if delta_t < 1.0 {
            continue;
        }
        let prior_dt = ds.prior_dt_active(r);
        let prior_r = ds.prior_ratings(r);
        if prior_dt.is_empty() {
            continue; // pos==0 row has no priors (Python would error; such users aren't emitted)
        }
        let f = forward(prior_dt, prior_r, &w);
        backward(delta_t, r.y as f64, &f, &mut w);
    }
    w
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = fsrs_v6::INIT_W.to_vec();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            fsrs_v6::INIT_W.to_vec()
        } else {
            train_weights(ds, train, cfg)
        };
        let test = &rows[s.test_start..s.test_end];
        for (i, pr) in fsrs_v6::predict(ds, test, &w, cfg).into_iter().enumerate() {
            eval_rows.push(test[i].clone());
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
