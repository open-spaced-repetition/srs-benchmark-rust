//! Hand-written forward + reverse-mode backward (analytic VJP) for the FSRS-6 BCE-loss gradient —
//! a manual reverse-mode of the exact forward in [`super::fsrs_v6::retention`]. Replaces the
//! forward-mode `Dual<21>` autodiff for the *training* gradient: forward-mode costs ~21× the value
//! pass per op (it carries a 21-long gradient array through every op); reverse-mode is ~1 forward +
//! ~1 backward regardless of the parameter count. The prediction path keeps `Dual<0>` (already
//! value-only). f64 throughout (FSRS-6 is an f64 algo — see CLAUDE.md per-algo precision).
//!
//! ⚠ This is a MANUAL VJP of a SPECIFIC forward. If you change the FSRS-6 formulas in `fsrs_v6.rs`
//! you MUST re-derive the matching backward here. The `fsrs6_analytic_grad_matches_forward_mode`
//! test (gated to `--features fp64`) guards against silent drift by comparing to the `Dual<21>`
//! oracle.

const NP: usize = 21;

// Branch tags (which arm of the recurrence a step took).
const SHORT: u8 = 0; // same-day (delta_t < 1)
const SUCCESS: u8 = 1; // delta_t >= 1, rating > 1
const FAIL: u8 = 2; // delta_t >= 1, rating == 1
const INIT: u8 = 3; // k == 0 (first review)

/// Loop-invariant (weight-only) subexpressions, computed once per gradient call.
pub struct WConsts {
    s_min: f64,
    s_max: f64,
    decay: f64,          // -w20
    factor: f64,         // 0.9^(1/decay) - 1
    dfactor_ddecay: f64, // ∂factor/∂decay
    exp_w8: f64,         // exp(w8)        (success-stability scale)
    exp3w5: f64,         // exp(3·w5)      (mean-reversion init_d at rating 4)
    initd4: f64,         // w4 - exp(3·w5) + 1
    msd: f64,            // exp(w17·w18)   (relearn min_s denominator)
}

pub fn wconsts(w: &[f64], s_min: f64, s_max: f64) -> WConsts {
    let decay = -w[20];
    let ln09 = 0.9f64.ln();
    let factor = (ln09 / decay).exp() - 1.0; // 0.9^(1/decay) - 1
    let dfactor_ddecay = (factor + 1.0) * ln09 * (-1.0 / (decay * decay));
    let exp3w5 = (3.0 * w[5]).exp();
    WConsts {
        s_min,
        s_max,
        decay,
        factor,
        dfactor_ddecay,
        exp_w8: w[8].exp(),
        exp3w5,
        initd4: w[4] - exp3w5 + 1.0,
        msd: (w[17] * w[18]).exp(),
    }
}

// ===================== forgetting curve =====================
// fc(t, s) = (1 + factor·t/s)^decay,  factor & decay loop-invariant (decay = -w20).

#[derive(Clone, Copy, Default)]
struct CurveCache {
    out: f64,
    base: f64,
    ln_base: f64,
}

#[inline]
fn curve_fwd(t: f64, s: f64, wc: &WConsts) -> CurveCache {
    let base = 1.0 + wc.factor * t / s;
    let out = base.powf(wc.decay);
    CurveCache { out, base, ln_base: base.ln() }
}

/// VJP of the curve. `t` is data (no grad). Returns g_s; accumulates gw[20] (via decay).
#[inline]
fn curve_bwd(c: &CurveCache, t: f64, s: f64, g_out: f64, wc: &WConsts, gw: &mut [f64]) -> f64 {
    // out = base^decay
    let g_base = g_out * wc.decay * c.out / c.base; // ∂out/∂base = decay·base^(decay-1)
    let g_decay_exp = g_out * c.out * c.ln_base; // explicit exponent
    // base = 1 + factor·(t/s)
    let g_factor = g_base * (t / s);
    let g_s = g_base * wc.factor * (-t / (s * s)); // t/s wrt s
    // factor = 0.9^(1/decay) - 1  →  also depends on decay
    let g_decay = g_decay_exp + g_factor * wc.dfactor_ddecay;
    gw[20] += -g_decay; // decay = -w20
    g_s
}

// ===================== one recurrence step =====================

#[derive(Clone, Copy, Default)]
pub struct StepCache {
    branch: u8,
    s: f64, // input stability (s_{k-1})
    d: f64, // input difficulty (d_{k-1})
    dt: f64,
    rating: f64,
    ns_in_open: bool, // ns inside (s_min, s_max) clamp
    nd_in_open: bool, // nd_pre / initd_pre inside (1, 10) clamp
    // difficulty (k>0)
    delta_d: f64,
    nd0: f64,
    nd_pre: f64, // also holds initd_pre for INIT
    // curve (SUCCESS/FAIL)
    cr: CurveCache,
    r: f64,
    // SHORT
    e1: f64,
    spa: f64,
    a_picks_one: bool,
    // SUCCESS
    spb: f64,
    rrb: f64,
    term: f64,
    hard: f64,
    easy: f64,
    // FAIL
    dp: f64,
    up: f64,
    sp1: f64,
    erc: f64,
    nf: f64,
    min_s: f64,
}

/// One k>0 step: maps (s, d) + (dt, rating) → (s_new, d_new). Returns new state + cache.
fn step_fwd(w: &[f64], s: f64, d: f64, dt: f64, rating: f64, wc: &WConsts) -> ((f64, f64), StepCache) {
    let mut c = StepCache { s, d, dt, rating, ..Default::default() };
    let short_term = dt < 1.0;
    let success = rating > 1.0;
    let ns = if short_term {
        c.branch = SHORT;
        // sinc = exp((w18 + rating-3)·w17) · s^(-w19)
        let e1 = ((w[18] + (rating - 3.0)) * w[17]).exp();
        let spa = s.powf(-w[19]);
        let sinc = e1 * spa;
        c.e1 = e1;
        c.spa = spa;
        let f = if rating >= 2.0 {
            let picks_one = 1.0 > sinc;
            c.a_picks_one = picks_one;
            if picks_one { 1.0 } else { sinc }
        } else {
            sinc
        };
        s * f
    } else {
        let cr = curve_fwd(dt, s, wc);
        let r = cr.out;
        c.cr = cr;
        c.r = r;
        if success {
            c.branch = SUCCESS;
            let hard = if rating == 2.0 { w[15] } else { 1.0 };
            let easy = if rating == 4.0 { w[16] } else { 1.0 };
            let spb = s.powf(-w[9]);
            let rrb = (w[10] * (1.0 - r)).exp() - 1.0;
            let term = wc.exp_w8 * (11.0 - d) * spb * rrb * hard * easy;
            c.spb = spb;
            c.rrb = rrb;
            c.term = term;
            c.hard = hard;
            c.easy = easy;
            s * (1.0 + term)
        } else {
            c.branch = FAIL;
            let dp = d.powf(-w[12]);
            let up = (s + 1.0).powf(w[13]);
            let sp1 = up - 1.0;
            let erc = (w[14] * (1.0 - r)).exp();
            let nf = w[11] * dp * sp1 * erc;
            let min_s = s / wc.msd;
            c.dp = dp;
            c.up = up;
            c.sp1 = sp1;
            c.erc = erc;
            c.nf = nf;
            c.min_s = min_s;
            // Dual::min(nf, min_s): ties → nf (the first arg).
            nf.min(min_s)
        }
    };
    // difficulty update (same for all k>0 branches)
    let delta_d = -w[6] * (rating - 3.0);
    let damp = delta_d * (10.0 - d) / 9.0;
    let nd0 = d + damp;
    let nd_pre = w[7] * wc.initd4 + (1.0 - w[7]) * nd0;
    let nd = nd_pre.clamp(1.0, 10.0);
    c.delta_d = delta_d;
    c.nd0 = nd0;
    c.nd_pre = nd_pre;
    c.ns_in_open = ns >= wc.s_min && ns <= wc.s_max;
    c.nd_in_open = nd_pre >= 1.0 && nd_pre <= 10.0;
    let s_new = ns.clamp(wc.s_min, wc.s_max);
    ((s_new, nd), c)
}

/// VJP of one k>0 step. Given adjoints on the OUTPUT state, returns adjoints on the INPUT state;
/// accumulates parameter grads into `gw`.
fn step_bwd(w: &[f64], c: &StepCache, g_s_new: f64, g_d_new: f64, wc: &WConsts, gw: &mut [f64]) -> (f64, f64) {
    let mut g_s = 0.0;
    let mut g_d = 0.0;
    let mut g_r = 0.0;
    let g_ns = if c.ns_in_open { g_s_new } else { 0.0 };
    let g_nd = g_d_new;

    // ---- difficulty backward ----
    let g_nd_pre = if c.nd_in_open { g_nd } else { 0.0 };
    // nd_pre = w7·initd4 + (1-w7)·nd0
    gw[7] += g_nd_pre * (wc.initd4 - c.nd0);
    let g_initd4 = g_nd_pre * w[7];
    let g_nd0 = g_nd_pre * (1.0 - w[7]);
    // initd4 = w4 - exp(3w5) + 1
    gw[4] += g_initd4;
    gw[5] += g_initd4 * (-wc.exp3w5 * 3.0);
    // nd0 = d + damp ; damp = delta_d·(10-d)/9
    g_d += g_nd0;
    let g_damp = g_nd0;
    let g_delta_d = g_damp * (10.0 - c.d) / 9.0;
    g_d += g_damp * c.delta_d * (-1.0 / 9.0);
    // delta_d = -w6·(rating-3)
    gw[6] += g_delta_d * (-(c.rating - 3.0));

    // ---- stability backward (branch-specific, uses g_ns) ----
    match c.branch {
        SHORT => {
            // ns = s·f ;  f = max(sinc,1)|sinc ;  sinc = e1·spa
            g_s += g_ns * (if c.a_picks_one { 1.0 } else { c.e1 * c.spa });
            let g_f = g_ns * c.s;
            let g_sinc = if c.a_picks_one { 0.0 } else { g_f };
            let g_e1 = g_sinc * c.spa;
            let g_spa = g_sinc * c.e1;
            // e1 = exp((w18 + rating-3)·w17)
            gw[17] += g_e1 * c.e1 * (w[18] + (c.rating - 3.0));
            gw[18] += g_e1 * c.e1 * w[17];
            // spa = s^(-w19)
            g_s += g_spa * (-w[19] * c.spa / c.s);
            gw[19] += g_spa * (-(c.s.ln()) * c.spa);
        }
        SUCCESS => {
            // ns = s·(1+term)
            g_s += g_ns * (1.0 + c.term);
            let g_term = g_ns * c.s;
            // term = exp_w8·(11-d)·spb·rrb·hard·easy
            let b11d = 11.0 - c.d;
            gw[8] += g_term * c.term; // ∂term/∂w8 = term (term ∝ exp(w8))
            let g_b11d = g_term * wc.exp_w8 * c.spb * c.rrb * c.hard * c.easy;
            let g_spb = g_term * wc.exp_w8 * b11d * c.rrb * c.hard * c.easy;
            let g_rrb = g_term * wc.exp_w8 * b11d * c.spb * c.hard * c.easy;
            if c.rating == 2.0 {
                gw[15] += g_term * wc.exp_w8 * b11d * c.spb * c.rrb * c.easy;
            }
            if c.rating == 4.0 {
                gw[16] += g_term * wc.exp_w8 * b11d * c.spb * c.rrb * c.hard;
            }
            g_d += -g_b11d;
            // spb = s^(-w9)
            g_s += g_spb * (-w[9] * c.spb / c.s);
            gw[9] += g_spb * (-(c.s.ln()) * c.spb);
            // rrb = exp(w10·(1-r)) - 1
            let erb = c.rrb + 1.0;
            g_r += g_rrb * (-w[10] * erb);
            gw[10] += g_rrb * ((1.0 - c.r) * erb);
        }
        FAIL => {
            // ns = min(nf, min_s) ; ties → nf
            let (g_nf, g_min_s) = if c.nf <= c.min_s { (g_ns, 0.0) } else { (0.0, g_ns) };
            // min_s = s / msd = s·exp(-w17·w18)
            g_s += g_min_s / wc.msd;
            gw[17] += g_min_s * (-c.min_s * w[18]);
            gw[18] += g_min_s * (-c.min_s * w[17]);
            // nf = w11·dp·sp1·erc
            gw[11] += g_nf * (c.dp * c.sp1 * c.erc);
            let g_dp = g_nf * w[11] * c.sp1 * c.erc;
            let g_sp1 = g_nf * w[11] * c.dp * c.erc;
            let g_erc = g_nf * w[11] * c.dp * c.sp1;
            // dp = d^(-w12)
            g_d += g_dp * (-w[12] * c.dp / c.d);
            gw[12] += g_dp * (-(c.d.ln()) * c.dp);
            // sp1 = (s+1)^w13 - 1
            let g_up = g_sp1;
            g_s += g_up * (w[13] * c.up / (c.s + 1.0));
            gw[13] += g_up * (c.up * (c.s + 1.0).ln());
            // erc = exp(w14·(1-r))
            g_r += g_erc * (-w[14] * c.erc);
            gw[14] += g_erc * ((1.0 - c.r) * c.erc);
        }
        _ => unreachable!(),
    }
    // curve backward (only SUCCESS/FAIL computed/used r)
    if c.branch == SUCCESS || c.branch == FAIL {
        g_s += curve_bwd(&c.cr, c.dt, c.s, g_r, wc, gw);
    }
    (g_s, g_d)
}

/// VJP of the k==0 init. Given adjoints on (s_0, d_0); accumulates gw.
fn init_bwd(w: &[f64], c: &StepCache, g_s0: f64, g_d0: f64, gw: &mut [f64]) {
    let rc = c.rating as usize; // rating 1..4
    // s_0 = clamp(w[rc-1], s_min, s_max)
    let g_ns = if c.ns_in_open { g_s0 } else { 0.0 };
    gw[rc - 1] += g_ns;
    // d_0 = clamp(initd_pre, 1, 10) ; initd_pre = w4 - exp(w5·(rc-1)) + 1
    let g_initd_pre = if c.nd_in_open { g_d0 } else { 0.0 };
    gw[4] += g_initd_pre;
    let rm1 = c.rating - 1.0;
    gw[5] += g_initd_pre * (-(w[5] * rm1).exp() * rm1);
}

/// Forward + reverse-mode backward for ONE row; accumulates d(weight·BCE)/dw into `gw` and returns
/// the prediction `p`. `caches` is a scratch buffer (cleared internally) reused across rows.
#[allow(clippy::too_many_arguments)]
pub fn grad_one(
    w: &[f64],
    prior_dt: &[f64],
    prior_r: &[i64],
    cur_dt: f64,
    y: f64,
    weight: f64,
    wc: &WConsts,
    gw: &mut [f64],
    caches: &mut Vec<StepCache>,
) -> f64 {
    caches.clear();
    let (mut s, mut d) = (0.0f64, 0.0f64);
    for k in 0..prior_r.len() {
        if k == 0 {
            let rating = prior_r[0] as f64;
            let rc = prior_r[0] as usize;
            let ns = w[rc - 1];
            let initd_pre = w[4] - (w[5] * (rating - 1.0)).exp() + 1.0;
            let s0 = ns.clamp(wc.s_min, wc.s_max);
            let d0 = initd_pre.clamp(1.0, 10.0);
            caches.push(StepCache {
                branch: INIT,
                rating,
                nd_pre: initd_pre,
                ns_in_open: ns >= wc.s_min && ns <= wc.s_max,
                nd_in_open: initd_pre >= 1.0 && initd_pre <= 10.0,
                ..Default::default()
            });
            s = s0;
            d = d0;
        } else {
            let ((s_new, d_new), c) = step_fwd(w, s, d, prior_dt[k], prior_r[k] as f64, wc);
            s = s_new;
            d = d_new;
            caches.push(c);
        }
    }
    // final prediction p = fc(cur_dt, s)
    let cr = curve_fwd(cur_dt, s, wc);
    let p = cr.out;
    // d(weight·BCE)/dp = weight·(p-y)/(p(1-p)), denom floored at 1e-12 (matches fsrs_v6::grad)
    let denom = (p * (1.0 - p)).max(1e-12);
    let g_p = weight * (p - y) / denom;
    let mut g_s = curve_bwd(&cr, cur_dt, s, g_p, wc, gw);
    let mut g_d = 0.0;
    for k in (1..caches.len()).rev() {
        let (gs, gd) = step_bwd(w, &caches[k], g_s, g_d, wc, gw);
        g_s = gs;
        g_d = gd;
    }
    init_bwd(w, &caches[0], g_s, g_d, gw);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Dual;
    use crate::models::fsrs_v6::{retention_dual, INIT_W};

    // Forward-mode oracle: value + gradient of fsrs_v6::retention via Dual<NP>.
    fn forward_mode(prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, w: &[f64; NP], s_min: f64, s_max: f64) -> (f64, [f64; NP]) {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w[k]));
        let ret = retention_dual(prior_dt, prior_r, cur_dt, &wd, s_min, s_max);
        (ret.v, ret.g)
    }

    #[cfg(feature = "fp64")] // f64-tight comparison; the analytic path here is f64 anyway
    #[test]
    fn fsrs6_analytic_grad_matches_forward_mode() {
        // Sequences exercising init + all three k>0 branches (short/success/fail) and the clamps.
        let seqs: Vec<(Vec<f64>, Vec<i64>, f64)> = vec![
            (vec![0.0, 9.0, 1.5, 30.0, 0.3, 100.0, 0.02], vec![3, 1, 3, 4, 2, 1, 2], 7.0),
            (vec![0.0, 5.0, 2.0], vec![2, 3, 1], 12.0),
            (vec![0.0, 1.0, 1.0, 1.0, 50.0, 0.5], vec![1, 1, 2, 3, 4, 3], 3.0),
            (vec![0.0, 0.4, 0.6, 20.0], vec![4, 2, 3, 1], 15.0),
        ];
        let (s_min, s_max) = (0.0001, 36500.0);
        // Perturb params off the default so more branches/values are exercised.
        let mut w = INIT_W;
        for (k, wk) in w.iter_mut().enumerate() {
            *wk += 0.01 * (k as f64).sin();
        }
        for (prior_dt, prior_r, cur_dt) in &seqs {
            let wc = wconsts(&w, s_min, s_max);
            let (y, weight) = (1.0, 0.7);
            let mut gw = vec![0.0f64; NP];
            let mut caches = Vec::new();
            let p = grad_one(&w, prior_dt, prior_r, *cur_dt, y, weight, &wc, &mut gw, &mut caches);
            let (p_fwd, g_fwd) = forward_mode(prior_dt, prior_r, *cur_dt, &w, s_min, s_max);
            assert!((p - p_fwd).abs() < 1e-9, "p {p} vs {p_fwd}");
            let denom = (p_fwd * (1.0 - p_fwd)).max(1e-12);
            let dl = weight * (p_fwd - y) / denom;
            for k in 0..NP {
                let expected = dl * g_fwd[k];
                assert!(
                    (gw[k] - expected).abs() < 1e-6 + 1e-6 * expected.abs(),
                    "param {k}: analytic {} vs fwd {}",
                    gw[k],
                    expected
                );
            }
        }
    }
}
