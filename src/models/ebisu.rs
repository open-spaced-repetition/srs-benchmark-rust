//! Ebisu v2 (untrainable) — `models/ebisu.py` + the `ebisu` Python library (v2 math).
//!
//! For each test row we rebuild a fresh Ebisu model from the row's prior `(Δt, rating)`
//! sequence (`features/memory_engineer.py::EbisuFeatureEngineer`: `zip(t_item[:-1],
//! r_item[:-1])`), then `predictRecall` at the current interval. The benchmark always calls
//! `updateRecall(..., total=1)`, so only the single-quiz path (`_updateRecallSingle`) is
//! ported. With integer `successes ∈ {0,1}` the noisy-quiz parameters collapse to
//! `q1 = 1, q0 = 0`, i.e. `(c, d) = (1, 0)` on success and `(-1, 1)` on failure.
//!
//! This port is well-conditioned (no RNG, no training): the only numeric knobs are `lgamma`
//! precision and the Brent root-finder tolerance, both ~1e-12 — far inside rule #5's 0.0005.

use super::ModelOutput;
use crate::config::Config;
use crate::eval::Params;
use crate::features::Dataset;
use crate::split::time_series_split;

// defaultModel(512, alpha=0.2, beta=0.2) — `models/ebisu.py`.
const ALPHA0: f64 = 0.2;
const BETA0: f64 = 0.2;
const T0: f64 = 512.0;

/// One Ebisu model: a Beta(α, β) prior on recall probability at time `t` since last review.
type Model = (f64, f64, f64);

/// ln Γ(x) via the Lanczos approximation (g=7), accurate to ~1e-15 for the x>0 we use.
/// Equivalent to `scipy.special.gammaln` to well within rule #5 tolerance.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: lnΓ(x) = ln(π / sin(πx)) − lnΓ(1−x).
        let pi = std::f64::consts::PI;
        (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &ci) in C.iter().enumerate().skip(1) {
            a += ci / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// betaln(a, b) = lnΓ(a) + lnΓ(b) − lnΓ(a+b).
#[inline]
fn betaln(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Beta function B(a, b).
#[inline]
fn betafn(a: f64, b: f64) -> f64 {
    betaln(a, b).exp()
}

/// log(b0·e^a0 + b1·e^a1) — `scipy.special.logsumexp` for two terms with signed weights.
#[inline]
fn logsumexp2(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    let amax = a0.max(a1);
    if !amax.is_finite() {
        return amax;
    }
    let s = b0 * (a0 - amax).exp() + b1 * (a1 - amax).exp();
    amax + s.ln()
}

/// `predictRecall(prior, tnow, exact=True)` — expected recall probability now.
#[inline]
fn predict_recall(model: Model, tnow: f64) -> f64 {
    let (a, b, t) = model;
    let dt = tnow / t;
    (betaln(a + dt, b) - betaln(a, b)).exp()
}

/// `mean,var → Beta(α,β)` (`_meanVarToBeta`).
#[inline]
fn mean_var_to_beta(mean: f64, var: f64) -> (f64, f64) {
    let tmp = mean * (1.0 - mean) / var - 1.0;
    (mean * tmp, (1.0 - mean) * tmp)
}

/// Roughly bracket monotonic `f` for positive inputs (`_findBracket`). Returns `[lo, hi]`
/// with `f(hi) < 0 < f(lo)`, or `None` if no sign change is found (Python: AssertionError).
fn find_bracket<F: Fn(f64) -> f64>(f: &F, init: f64) -> Option<(f64, f64)> {
    let (factorhigh, factorlow) = (2.0f64, 0.5f64);
    let mut blow = factorlow * init;
    let mut bhigh = factorhigh * init;
    let mut flow = f(blow);
    let mut fhigh = f(bhigh);
    let mut guard = 0;
    while flow > 0.0 && fhigh > 0.0 {
        blow = bhigh;
        flow = fhigh;
        bhigh *= factorhigh;
        fhigh = f(bhigh);
        guard += 1;
        if guard > 2000 {
            return None;
        }
    }
    while flow < 0.0 && fhigh < 0.0 {
        bhigh = blow;
        fhigh = flow;
        blow *= factorlow;
        flow = f(blow);
        guard += 1;
        if guard > 4000 {
            return None;
        }
    }
    if flow > 0.0 && fhigh < 0.0 {
        Some((blow, bhigh))
    } else {
        None
    }
}

/// Brent's method on a bracketed root (mirrors `scipy.optimize.brentq` defaults).
fn brentq<F: Fn(f64) -> f64>(f: &F, xa: f64, xb: f64) -> f64 {
    let xtol = 2e-12;
    let rtol = 8.881_784_197_001_252e-16; // scipy default = 4*eps
    let maxiter = 100;

    let mut xpre = xa;
    let mut xcur = xb;
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return xpre;
    }
    if fcur == 0.0 {
        return xcur;
    }
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for _ in 0..maxiter {
        if fpre * fcur < 0.0 {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = xcur - xpre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return xcur;
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // secant
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // inverse quadratic
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    xcur
}

/// `_updateRecallSingle(prior, result∈{0,1}, tnow, rebalance=True)`.
/// `use_log` mirrors the library's escalation to the log domain for stability.
fn update_recall_single(prior: Model, result: f64, tnow: f64, use_log: bool) -> Model {
    let (alpha, beta, t) = prior;
    if alpha > 400.0 && beta > 400.0 && !use_log {
        return update_recall_single(prior, result, tnow, true);
    }

    // result ∈ {0,1}: q1 = max(result, 1-result) = 1, q0 = 1 - q1 = 0.
    let z = result > 0.5;
    let (c, d) = if !z { (-1.0, 1.0) } else { (1.0, 0.0) };
    let dt = tnow / t;

    let den = c * betafn(alpha + dt, beta) + if d != 0.0 { d * betafn(alpha, beta) } else { 0.0 };
    // logDen only needed in the log path.
    let log_den = if use_log {
        logsumexp2(betaln(alpha + dt, beta), betaln(alpha, beta), c, d)
    } else {
        0.0
    };

    let moment = |nth: f64, et: f64| -> f64 {
        let mut num = c * betafn(alpha + dt + nth * dt * et, beta);
        if d != 0.0 {
            num += d * betafn(alpha + nth * dt * et, beta);
        }
        num / den
    };
    let log_moment = |nth: f64, et: f64| -> f64 {
        if d != 0.0 {
            logsumexp2(
                betaln(alpha + dt + nth * dt * et, beta),
                betaln(alpha + nth * dt * et, beta),
                c,
                d,
            ) - log_den
        } else {
            c.ln() + betaln(alpha + dt + nth * dt * et, beta) - log_den
        }
    };

    // Rebalance: find et so the 1st moment (mean recall) at tback = et·tnow equals 0.5.
    let target = 0.5f64.ln();
    let et = if use_log {
        let rootfn = |et: f64| log_moment(1.0, et) - target;
        match find_bracket(&rootfn, 1.0 / dt) {
            Some((lo, hi)) => brentq(&rootfn, lo, hi),
            None => {
                // Already in log domain and still no bracket: keep the model unchanged
                // (Python would raise; in practice this never triggers on real data).
                return prior;
            }
        }
    } else {
        let rootfn = |et: f64| moment(1.0, et) - 0.5;
        match find_bracket(&rootfn, 1.0 / dt) {
            Some((lo, hi)) => brentq(&rootfn, lo, hi),
            None => return update_recall_single(prior, result, tnow, true),
        }
    };
    let tback = et * tnow;

    let (mean, second_moment) = if use_log {
        (log_moment(1.0, et).exp(), log_moment(2.0, et).exp())
    } else {
        (moment(1.0, et), moment(2.0, et))
    };
    let var = second_moment - mean * mean;
    let (new_alpha, new_beta) = mean_var_to_beta(mean, var);

    if !(new_alpha > 0.0 && new_beta > 0.0 && new_alpha.is_finite() && new_beta.is_finite())
        && !use_log
    {
        return update_recall_single(prior, result, tnow, true);
    }
    (new_alpha, new_beta, tback)
}

/// Build the Ebisu model for one row from its prior `(Δt, rating)` sequence
/// (`Ebisu.ebisu_v2`): start from the default model, fold in each prior review.
fn ebisu_v2(prior_dt: &[f64], prior_r: &[i64]) -> Model {
    let mut model: Model = (ALPHA0, BETA0, T0);
    for (k, &dt) in prior_dt.iter().enumerate() {
        let result = if prior_r[k] > 1 { 1.0 } else { 0.0 };
        let tnow = dt.max(0.001);
        model = update_recall_single(model, result, tnow, false);
    }
    model
}

/// Ebisu-v2 (untrainable). Per split, predict each test row from its rebuilt model.
pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    for s in splits {
        for r in &rows[s.test_start..s.test_end] {
            let model = ebisu_v2(ds.prior_dt_active(r), ds.prior_ratings(r));
            eval_rows.push(r.clone());
            p.push(predict_recall(model, r.delta_t.max(0.001)));
        }
    }
    ModelOutput {
        eval_rows,
        p,
        params: Params::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lgamma_matches_known_values() {
        // lnΓ(0.5) = ln(√π); lnΓ(1)=0; lnΓ(5)=ln(24); lnΓ(0.2)≈1.5240638224.
        assert!((ln_gamma(0.5) - (std::f64::consts::PI).sqrt().ln()).abs() < 1e-12);
        assert!(ln_gamma(1.0).abs() < 1e-12);
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-10);
        assert!((ln_gamma(0.2) - 1.524_063_822_430_784).abs() < 1e-10);
    }

    #[test]
    fn ebisu_matches_python_reference() {
        // Ground truth from the `ebisu` Python library (v2) on this sequence.
        let dt = [0.001, 2.5, 0.4, 13.2, 40.0, 0.001, 99.0];
        let r = [1i64, 3, 1, 3, 4, 2, 3];
        // Chained lgamma + brentq rounding accumulates to ~1e-8 over 7 updates — far inside
        // rule #5's LogLoss tolerance (the predictions, not the internal α/β, are binding).
        let m = ebisu_v2(&dt, &r);
        assert!((m.0 - 2.888_061_482_614_636_7).abs() < 1e-6, "alpha {}", m.0);
        assert!((m.1 - 2.888_061_482_614_624_3).abs() < 1e-6, "beta {}", m.1);
        assert!((m.2 - 70.490_432_746_497_23).abs() < 1e-5, "t {}", m.2);
        assert!((predict_recall(m, 1.0) - 0.988_918_318_845_524_7).abs() < 1e-7);
        assert!((predict_recall(m, 30.0) - 0.728_877_625_914_047_8).abs() < 1e-7);
        assert!((predict_recall(m, 365.0) - 0.080_440_137_183_413_54).abs() < 1e-7);

        // Long all-success run (exercises repeated rebalancing / halflife growth).
        let dt2 = [0.5f64; 40];
        let r2 = [3i64; 40];
        let m2 = ebisu_v2(&dt2, &r2);
        assert!((m2.0 - 0.186_788_306_092_482_1).abs() < 1e-6, "alpha2 {}", m2.0);
        assert!((m2.2 - 745.636_600_224_456_1).abs() < 1e-4, "t2 {}", m2.2);
        assert!((predict_recall(m2, 1.0) - 0.996_145_752_719_407_8).abs() < 1e-7);
    }

    #[test]
    fn predict_recall_default_model() {
        // Fresh model (0.2,0.2,512): at tnow=t the mean recall is α/(α+β)=0.5.
        let p = predict_recall((0.2, 0.2, 512.0), 512.0);
        assert!((p - 0.5).abs() < 1e-9, "{p}");
        // Short interval → high recall, long interval → low recall (monotone). Beta(0.2,0.2)
        // is fat-tailed, so recall decays slowly: ~0.17 even at ~195 halflives.
        assert!(predict_recall((0.2, 0.2, 512.0), 1.0) > 0.9);
        assert!(predict_recall((0.2, 0.2, 512.0), 100000.0) < 0.2);
    }
}
