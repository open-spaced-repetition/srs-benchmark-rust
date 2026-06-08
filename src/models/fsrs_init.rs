//! Shared FSRS S0 (initial-stability) initialization — port of
//! `models/fsrs_v4.py::initialize_parameters`, used by FSRS v4/v4.5/v5/v6.
//!
//! For each first rating, fit the initial stability that best explains the observed
//! second-review recall (`i == 2` rows), then fill any missing ratings via the log-linear
//! interpolation table. A robust 1-D golden-section search replaces `scipy.minimize`; since
//! the fit objective is ~unimodal it finds the true minimum (≥ as good as scipy), which —
//! under the one-sided LogLoss rule — can only help.

use std::collections::HashMap;

use crate::config::Config;
use crate::features::{Dataset, Row};

/// Golden-section minimization of a unimodal `f` on `[lo, hi]`.
fn golden<F: Fn(f64) -> f64>(f: &F, lo: f64, hi: f64) -> f64 {
    let gr = (5f64.sqrt() - 1.0) / 2.0; // 0.6180339887...
    let mut a = lo;
    let mut b = hi;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..200 {
        if (b - a).abs() < 1e-11 {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

/// Global golden-section min over `[lo, hi]` (assumes ~unimodality). Used by the Adam-trained
/// FSRS models, which re-optimize S0 afterwards, so the best obtainable S0 is fine.
fn minimize_1d<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64) -> f64 {
    golden(&f, lo, hi)
}

/// Local minimization from `x0` to the local minimum in its basin (mimics `scipy.minimize` /
/// L-BFGS-B started at `x0`). FSRS-6-one-step needs this: its tiny-lr single-pass SGD barely
/// moves S0, so the init must land where scipy lands — which can be a boundary local min, not
/// the global min the golden-section would pick. March downhill from `x0` (growing steps) to
/// bracket the basin's min, then golden-section the bracket; a descent into a bound returns it.
fn minimize_1d_local<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, x0: f64) -> f64 {
    let x0 = x0.clamp(lo, hi);
    let f0 = f(x0);
    let span = hi - lo;
    let eps = (span * 1e-6).max(1e-12);
    let fr = if x0 + eps <= hi { f(x0 + eps) } else { f0 };
    let fl = if x0 - eps >= lo { f(x0 - eps) } else { f0 };
    // Downhill direction; if neither side decreases, x0 is already a local min.
    let dir = if fr < f0 && fr <= fl {
        1.0
    } else if fl < f0 {
        -1.0
    } else {
        return x0;
    };

    let mut step = span * 1e-3;
    let (mut a, mut fa) = (x0, f0);
    let (mut b, mut fb) = ((x0 + dir * step).clamp(lo, hi), 0.0);
    fb = f(b);
    loop {
        if b <= lo || b >= hi {
            // Descended into a bound: it's the basin min if still decreasing, else bracket [a,b].
            if fb <= fa {
                return b;
            }
            break;
        }
        let c = (b + dir * step).clamp(lo, hi);
        let fc = f(c);
        if fc >= fb {
            let (g_lo, g_hi) = if a < c { (a, c) } else { (c, a) };
            return golden(&f, g_lo, g_hi);
        }
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        step *= 1.6;
    }
    let (g_lo, g_hi) = if a < b { (a, b) } else { (b, a) };
    golden(&f, g_lo, g_hi)
}

/// Fit `w[0..4]` (initial stabilities for ratings 1–4). `fc(delta_t, stability)` is the
/// version's forgetting curve (value form). `default_s0` are the model's `init_w[0..4]`.
pub fn fit_s0<FC: Fn(f64, f64) -> f64>(
    ds: &Dataset,
    train: &[Row],
    cfg: &Config,
    default_s0: [f64; 4],
    fc: FC,
) -> [f64; 4] {
    fit_s0_impl(ds, train, cfg, default_s0, fc, false)
}

/// As [`fit_s0`] but minimizing locally from `x0 = init_s0` (mimics `scipy.minimize`). Use
/// this for FSRS-6-one-step, whose SGD doesn't re-optimize S0 (see [`minimize_1d_local`]).
pub fn fit_s0_from_x0<FC: Fn(f64, f64) -> f64>(
    ds: &Dataset,
    train: &[Row],
    cfg: &Config,
    default_s0: [f64; 4],
    fc: FC,
) -> [f64; 4] {
    fit_s0_impl(ds, train, cfg, default_s0, fc, true)
}

fn fit_s0_impl<FC: Fn(f64, f64) -> f64>(
    ds: &Dataset,
    train: &[Row],
    cfg: &Config,
    default_s0: [f64; 4],
    fc: FC,
    from_x0: bool,
) -> [f64; 4] {
    let _ = ds;
    let s_min = cfg.s_min;
    let s_max = cfg.init_s_max;

    // Group i==2 rows by (first_rating, delta_t): sum_y, count.
    let mut groups: HashMap<(i64, u64), (f64, f64)> = HashMap::new();
    let mut sum_all = 0.0;
    let mut n_all = 0.0;
    for r in train {
        sum_all += r.y as f64;
        n_all += 1.0;
        if r.i == 2 {
            let e = groups.entry((r.first_rating, r.delta_t.to_bits())).or_insert((0.0, 0.0));
            e.0 += r.y as f64;
            e.1 += 1.0;
        }
    }
    let average_recall = if n_all > 0.0 { sum_all / n_all } else { 0.0 };

    // Per first rating: fit stability.
    let mut rs: [Option<f64>; 5] = [None; 5];
    let mut rc: [f64; 5] = [0.0; 5];
    for fr in 1..=4i64 {
        let mut gs: Vec<(f64, f64, f64)> = groups
            .iter()
            .filter(|((g_fr, _), _)| *g_fr == fr)
            .map(|((_, dt_bits), (sy, cnt))| {
                let dt = f64::from_bits(*dt_bits);
                let mean = sy / cnt;
                let recall = if cfg.use_secs_intervals {
                    mean
                } else {
                    (mean * cnt + average_recall) / (cnt + 1.0)
                };
                (dt, recall, *cnt)
            })
            .collect();
        if gs.is_empty() {
            continue;
        }
        // Sort by delta_t so the loss summation order is deterministic (HashMap iteration is
        // randomized) — also matches pandas `groupby` key order. Without this the S0 fit, and
        // thus all fit_s0 models (FSRS v4/v4.5/v5/v6), are non-deterministic run-to-run.
        gs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let init_s0 = default_s0[(fr - 1) as usize];
        let secs = cfg.use_secs_intervals;
        let loss = |s: f64| -> f64 {
            let mut ll = 0.0;
            for &(dt, recall, cnt) in &gs {
                let yp = fc(dt, s).clamp(1e-15, 1.0 - 1e-15);
                ll += -(recall * yp.ln() + (1.0 - recall) * (1.0 - yp).ln()) * cnt;
            }
            if secs {
                ll
            } else {
                ll + (s - init_s0).abs() / 16.0
            }
        };
        rs[fr as usize] = Some(if from_x0 {
            minimize_1d_local(loss, s_min, s_max, init_s0)
        } else {
            minimize_1d(loss, s_min, s_max)
        });
        rc[fr as usize] = gs.iter().map(|g| g.2).sum();
    }

    // Consistency: a lower rating shouldn't have higher stability than a higher rating.
    for (small, big) in [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] {
        if let (Some(ss), Some(bs)) = (rs[small], rs[big]) {
            if ss > bs {
                if rc[small] > rc[big] {
                    rs[big] = Some(ss);
                } else {
                    rs[small] = Some(bs);
                }
            }
        }
    }

    let initial = interpolate(&mut rs, default_s0);
    [
        initial[0].clamp(s_min, s_max),
        initial[1].clamp(s_min, s_max),
        initial[2].clamp(s_min, s_max),
        initial[3].clamp(s_min, s_max),
    ]
}

fn interpolate(rs: &mut [Option<f64>; 5], default_s0: [f64; 4]) -> [f64; 4] {
    const W1: f64 = 0.41;
    const W2: f64 = 0.54;
    let known: Vec<usize> = (1..=4).filter(|&i| rs[i].is_some()).collect();
    let pw = f64::powf;
    let g = |rs: &[Option<f64>; 5], i: usize| rs[i].unwrap();

    match known.len() {
        0 => default_s0,
        1 => {
            let r = known[0];
            let factor = g(rs, r) / default_s0[r - 1];
            [
                default_s0[0] * factor,
                default_s0[1] * factor,
                default_s0[2] * factor,
                default_s0[3] * factor,
            ]
        }
        2 => {
            let has = |i: usize| rs[i].is_some();
            if !has(1) && !has(2) {
                rs[2] = Some(pw(g(rs, 3), 1.0 / (1.0 - W2)) * pw(g(rs, 4), 1.0 - 1.0 / (1.0 - W2)));
                rs[1] = Some(pw(g(rs, 2), 1.0 / W1) * pw(g(rs, 3), 1.0 - 1.0 / W1));
            } else if !has(1) && !has(3) {
                rs[3] = Some(pw(g(rs, 2), 1.0 - W2) * pw(g(rs, 4), W2));
                rs[1] = Some(pw(g(rs, 2), 1.0 / W1) * pw(g(rs, 3), 1.0 - 1.0 / W1));
            } else if !has(1) && !has(4) {
                rs[4] = Some(pw(g(rs, 2), 1.0 - 1.0 / W2) * pw(g(rs, 3), 1.0 / W2));
                rs[1] = Some(pw(g(rs, 2), 1.0 / W1) * pw(g(rs, 3), 1.0 - 1.0 / W1));
            } else if !has(2) && !has(3) {
                let denom = W1 + W2 - W1 * W2;
                rs[2] = Some(pw(g(rs, 1), W1 / denom) * pw(g(rs, 4), 1.0 - W1 / denom));
                rs[3] = Some(pw(g(rs, 1), 1.0 - W2 / denom) * pw(g(rs, 4), W2 / denom));
            } else if !has(2) && !has(4) {
                rs[2] = Some(pw(g(rs, 1), W1) * pw(g(rs, 3), 1.0 - W1));
                rs[4] = Some(pw(g(rs, 2), 1.0 - 1.0 / W2) * pw(g(rs, 3), 1.0 / W2));
            } else {
                // !has(3) && !has(4)
                rs[3] = Some(pw(g(rs, 1), 1.0 - 1.0 / (1.0 - W1)) * pw(g(rs, 2), 1.0 / (1.0 - W1)));
                rs[4] = Some(pw(g(rs, 2), 1.0 - 1.0 / W2) * pw(g(rs, 3), 1.0 / W2));
            }
            [g(rs, 1), g(rs, 2), g(rs, 3), g(rs, 4)]
        }
        3 => {
            if rs[1].is_none() {
                rs[1] = Some(pw(g(rs, 2), 1.0 / W1) * pw(g(rs, 3), 1.0 - 1.0 / W1));
            } else if rs[2].is_none() {
                rs[2] = Some(pw(g(rs, 1), W1) * pw(g(rs, 3), 1.0 - W1));
            } else if rs[3].is_none() {
                rs[3] = Some(pw(g(rs, 2), 1.0 - W2) * pw(g(rs, 4), W2));
            } else {
                rs[4] = Some(pw(g(rs, 2), 1.0 - 1.0 / W2) * pw(g(rs, 3), 1.0 / W2));
            }
            [g(rs, 1), g(rs, 2), g(rs, 3), g(rs, 4)]
        }
        _ => [g(rs, 1), g(rs, 2), g(rs, 3), g(rs, 4)],
    }
}
