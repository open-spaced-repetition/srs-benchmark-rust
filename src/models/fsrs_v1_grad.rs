//! Hand-written reverse-mode VJP for the FSRS-1 BCE-loss gradient — a manual reverse-mode of the
//! exact forward in [`super::fsrs_v1::retention`]. Replaces forward-mode `Dual<7>` for the training
//! gradient. Prediction keeps `Dual<0>`. f64. No penalty / grad_mask.
//!
//! FSRS-1 (NP=7) is a 3-state recurrence (stability, difficulty, lapse-count). The lapse count `l`
//! is data-driven (constant w.r.t. params, no gradient). Quirks: curve `0.9^(t/s)`; difficulty
//! `nd = relu(d + r + 0.1 - 0.25·2^(rating-1))` reads the curve `r` and floors at 0; success
//! stability reads the NEW `nd` via `(nd+0.1)^w3`; the FAIL branch is `w0·exp(w6·l)` — independent
//! of s/d/r; init difficulty `w1+3-rating` has NO clamp. NP is small, so the reverse-mode win over
//! the (already-vectorized) forward-mode is modest — kept iff the Wilcoxon gate passes.
//!
//! ⚠ Manual VJP of a SPECIFIC forward — change the formulas and re-derive. Guarded by
//! `fsrs1_analytic_grad_matches_forward_mode` (gated `--features fp64`).

const NP: usize = 7;
const LN09: f64 = -0.10536051565782628; // ln(0.9)

const SUCCESS: u8 = 1;
const FAIL: u8 = 2;
const INIT: u8 = 3;

pub struct WConsts {
    s_min: f64,
    s_max: f64,
    exp_w2: f64, // exp(w2)
}

pub fn wconsts(w: &[f64], s_min: f64, s_max: f64) -> WConsts {
    WConsts { s_min, s_max, exp_w2: w[2].exp() }
}

#[inline]
fn curve_fwd(t: f64, s: f64) -> f64 {
    (LN09 * t / s).exp()
}
#[inline]
fn curve_bwd(out: f64, t: f64, s: f64, g_out: f64) -> f64 {
    g_out * out * LN09 * (-t / (s * s))
}

#[derive(Clone, Copy, Default)]
pub struct StepCache {
    branch: u8,
    s: f64,
    d: f64,
    dt: f64,
    rating: f64,
    ns_in_open: bool,
    nd_in_open: bool, // nd_pre >= 0 (clamp_min 0)
    nd: f64,
    r: f64,
    ndp: f64,  // (nd+0.1)^w3   (success)
    sp: f64,   // s^w4          (success)
    er: f64,   // exp(w5·(1-r)) (success)
    term: f64, // success
    eg: f64,   // exp(w6·l)     (fail)
    l: f64,    // lapse count (data; no grad)
}

fn step_fwd(w: &[f64], s: f64, d: f64, l: f64, dt: f64, rating: f64, wc: &WConsts) -> ((f64, f64), StepCache) {
    let mut c = StepCache { s, d, dt, rating, l, ..Default::default() };
    let r = curve_fwd(dt, s);
    c.r = r;
    let pow2 = 2f64.powf(rating - 1.0);
    // nd = relu(d + r + 0.1 - 0.25·2^(rating-1))
    let nd_pre = d + r + (0.1 - 0.25 * pow2);
    let nd = nd_pre.max(0.0);
    c.nd = nd;
    c.nd_in_open = nd_pre >= 0.0;
    let ns = if rating > 1.0 {
        c.branch = SUCCESS;
        // s·(1 + exp_w2·(nd+0.1)^w3·s^w4·(exp(w5·(1-r))-1))
        let ndp = (nd + 0.1).powf(w[3]);
        let sp = s.powf(w[4]);
        let er = (w[5] * (1.0 - r)).exp();
        let term = wc.exp_w2 * ndp * sp * (er - 1.0);
        c.ndp = ndp;
        c.sp = sp;
        c.er = er;
        c.term = term;
        s * (1.0 + term)
    } else {
        c.branch = FAIL;
        let eg = (w[6] * l).exp();
        c.eg = eg;
        w[0] * eg // independent of s, d, r
    };
    c.ns_in_open = ns >= wc.s_min && ns <= wc.s_max;
    ((ns.clamp(wc.s_min, wc.s_max), nd), c)
}

fn step_bwd(w: &[f64], c: &StepCache, g_s_new: f64, g_d_new: f64, wc: &WConsts, gw: &mut [f64]) -> (f64, f64) {
    let mut g_s = 0.0;
    let mut g_r = 0.0;
    let mut g_nd = g_d_new; // d_new = nd
    let g_ns = if c.ns_in_open { g_s_new } else { 0.0 };

    match c.branch {
        SUCCESS => {
            g_s += g_ns * (1.0 + c.term);
            let g_term = g_ns * c.s;
            let rr = c.er - 1.0;
            // term = exp_w2·ndp·sp·rr
            gw[2] += g_term * c.term; // ∂term/∂w2 = term (term ∝ exp(w2))
            let g_ndp = g_term * wc.exp_w2 * c.sp * rr;
            let g_sp = g_term * wc.exp_w2 * c.ndp * rr;
            let g_rr = g_term * wc.exp_w2 * c.ndp * c.sp;
            // ndp = (nd+0.1)^w3
            let base = c.nd + 0.1;
            g_nd += g_ndp * (w[3] * c.ndp / base);
            gw[3] += g_ndp * (c.ndp * base.ln());
            // sp = s^w4
            g_s += g_sp * (w[4] * c.sp / c.s);
            gw[4] += g_sp * (c.sp * c.s.ln());
            // rr = er-1 ; er = exp(w5·(1-r))
            g_r += g_rr * (-w[5] * c.er);
            gw[5] += g_rr * ((1.0 - c.r) * c.er);
        }
        FAIL => {
            // ns = w0·eg ; eg = exp(w6·l)  (l data)
            gw[0] += g_ns * c.eg;
            gw[6] += g_ns * w[0] * c.eg * c.l;
        }
        _ => unreachable!(),
    }
    // difficulty: nd = max(nd_pre, 0) ; nd_pre = d + r + const  (uses r!)
    let g_nd_pre = if c.nd_in_open { g_nd } else { 0.0 };
    let g_d = g_nd_pre; // ∂nd_pre/∂d = 1
    g_r += g_nd_pre; // ∂nd_pre/∂r = 1
    g_s += curve_bwd(c.r, c.dt, c.s, g_r);
    (g_s, g_d)
}

fn init_bwd(c: &StepCache, g_s0: f64, g_d0: f64, gw: &mut [f64]) {
    let pow2 = 2f64.powf(c.rating - 1.0);
    // s_0 = clamp(w0·0.25·2^(rating-1))
    let g_ns = if c.ns_in_open { g_s0 } else { 0.0 };
    gw[0] += g_ns * (0.25 * pow2);
    // d_0 = w1 + 3 - rating  (NO clamp)
    gw[1] += g_d0;
}

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
    let (mut s, mut d, mut l) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..prior_r.len() {
        let rating = prior_r[k] as f64;
        let relu_l = (2.0 - rating).max(0.0);
        if k == 0 {
            let pow2 = 2f64.powf(rating - 1.0);
            let ns = w[0] * (0.25 * pow2);
            // init difficulty has NO clamp → grad always passes
            caches.push(StepCache {
                branch: INIT,
                rating,
                ns_in_open: ns >= wc.s_min && ns <= wc.s_max,
                nd_in_open: true,
                ..Default::default()
            });
            s = ns.clamp(wc.s_min, wc.s_max);
            d = w[1] + 3.0 - rating;
            l = relu_l;
        } else {
            let ((s_new, d_new), c) = step_fwd(w, s, d, l, prior_dt[k], rating, wc);
            s = s_new;
            d = d_new;
            l += relu_l;
            caches.push(c);
        }
    }
    let p = curve_fwd(cur_dt, s);
    let denom = (p * (1.0 - p)).max(1e-12);
    let g_p = weight * (p - y) / denom;
    let mut g_s = curve_bwd(p, cur_dt, s, g_p);
    let mut g_d = 0.0;
    for k in (1..caches.len()).rev() {
        let (gs, gd) = step_bwd(w, &caches[k], g_s, g_d, wc, gw);
        g_s = gs;
        g_d = gd;
    }
    init_bwd(&caches[0], g_s, g_d, gw);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Dual;
    use crate::models::fsrs_v1::{retention_dual, INIT_W};

    fn forward_mode(prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, w: &[f64; NP], s_min: f64, s_max: f64) -> (f64, [f64; NP]) {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w[k]));
        let ret = retention_dual(prior_dt, prior_r, cur_dt, &wd, s_min, s_max);
        (ret.v, ret.g)
    }

    #[cfg(feature = "fp64")]
    #[test]
    fn fsrs1_analytic_grad_matches_forward_mode() {
        let seqs: Vec<(Vec<f64>, Vec<i64>, f64)> = vec![
            (vec![0.0, 9.0, 1.5, 30.0, 100.0, 3.0], vec![3, 1, 3, 4, 2, 1], 7.0),
            (vec![0.0, 5.0, 2.0], vec![2, 3, 1], 12.0),
            (vec![0.0, 1.0, 1.0, 50.0, 5.0], vec![1, 2, 3, 4, 3], 3.0),
            (vec![0.0, 4.0, 6.0, 20.0], vec![4, 2, 3, 1], 15.0),
        ];
        let (s_min, s_max) = (0.0001, 36500.0);
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
