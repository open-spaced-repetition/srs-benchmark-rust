//! Hand-written reverse-mode VJP for the FSRS-3 BCE-loss gradient — a manual reverse-mode of the
//! exact forward in [`super::fsrs_v3::retention`]. Replaces forward-mode `Dual<13>` for the training
//! gradient. Prediction keeps `Dual<0>`. f64 (FSRS-3 is an f64 algo). No penalty / grad_mask.
//!
//! FSRS-3 quirks vs the v4–v6 family: curve `0.9^(t/s)`, difficulty `nd` is computed BEFORE
//! stability and the stability branch reads the NEW `nd` (not the old `d`), success uses `(11-nd)`,
//! fail has no `-1`, additive init. NP=13.
//!
//! ⚠ Manual VJP of a SPECIFIC forward — change the formulas and re-derive. Guarded by
//! `fsrs3_analytic_grad_matches_forward_mode` (gated `--features fp64`).

const NP: usize = 13;
const LN09: f64 = -0.10536051565782628; // ln(0.9)

const SUCCESS: u8 = 1;
const FAIL: u8 = 2;
const INIT: u8 = 3;

pub struct WConsts {
    s_min: f64,
    s_max: f64,
    exp_w6: f64, // exp(w6)
}

pub fn wconsts(w: &[f64], s_min: f64, s_max: f64) -> WConsts {
    WConsts { s_min, s_max, exp_w6: w[6].exp() }
}

// fc(t, s) = 0.9^(t/s) = exp(LN09·t/s)
#[inline]
fn curve_fwd(t: f64, s: f64) -> f64 {
    (LN09 * t / s).exp()
}

#[inline]
fn curve_bwd(out: f64, t: f64, s: f64, g_out: f64) -> f64 {
    // out = exp(LN09·t/s) ; ∂out/∂s = out·LN09·t·(-1/s²)
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
    nd_in_open: bool,
    nd: f64,
    nd0: f64,
    r: f64,
    // success
    sp7: f64,  // s^w7
    er8: f64,  // exp(w8·(1-r))
    term: f64,
    // fail
    ndp: f64,  // nd^w10
    sp11: f64, // s^w11
    er12: f64, // exp(w12·(1-r))
}

fn step_fwd(w: &[f64], s: f64, d: f64, dt: f64, rating: f64, wc: &WConsts) -> ((f64, f64), StepCache) {
    let mut c = StepCache { s, d, dt, rating, ..Default::default() };
    let r = curve_fwd(dt, s);
    c.r = r;
    // difficulty FIRST (does not use r): nd = w5·w2 + (1-w5)·(d + w4·(rating-3)), clamp(1,10)
    let nd0 = d + w[4] * (rating - 3.0);
    let nd_pre = w[5] * w[2] + (1.0 - w[5]) * nd0;
    let nd = nd_pre.clamp(1.0, 10.0);
    c.nd0 = nd0;
    c.nd = nd;
    c.nd_in_open = nd_pre >= 1.0 && nd_pre <= 10.0;
    let ns = if rating > 1.0 {
        c.branch = SUCCESS;
        // s·(1 + exp_w6·(11-nd)·s^w7·(exp(w8·(1-r))-1))
        let sp7 = s.powf(w[7]);
        let er8 = (w[8] * (1.0 - r)).exp();
        let rr = er8 - 1.0;
        let term = wc.exp_w6 * (11.0 - nd) * sp7 * rr;
        c.sp7 = sp7;
        c.er8 = er8;
        c.term = term;
        s * (1.0 + term)
    } else {
        c.branch = FAIL;
        // w9·nd^w10·s^w11·exp(w12·(1-r))
        let ndp = nd.powf(w[10]);
        let sp11 = s.powf(w[11]);
        let er12 = (w[12] * (1.0 - r)).exp();
        c.ndp = ndp;
        c.sp11 = sp11;
        c.er12 = er12;
        w[9] * ndp * sp11 * er12
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
            // ns = s·(1 + term)
            g_s += g_ns * (1.0 + c.term);
            let g_term = g_ns * c.s;
            // term = exp_w6·(11-nd)·sp7·rr ;  rr = er8 - 1
            let b11nd = 11.0 - c.nd;
            let rr = c.er8 - 1.0;
            gw[6] += g_term * c.term; // ∂term/∂w6 = term (term ∝ exp(w6))
            let g_b11nd = g_term * wc.exp_w6 * c.sp7 * rr;
            let g_sp7 = g_term * wc.exp_w6 * b11nd * rr;
            let g_rr = g_term * wc.exp_w6 * b11nd * c.sp7;
            g_nd += -g_b11nd;
            // sp7 = s^w7
            g_s += g_sp7 * (w[7] * c.sp7 / c.s);
            gw[7] += g_sp7 * (c.sp7 * c.s.ln());
            // rr = er8 - 1 ; er8 = exp(w8·(1-r))
            g_r += g_rr * (-w[8] * c.er8);
            gw[8] += g_rr * ((1.0 - c.r) * c.er8);
        }
        FAIL => {
            // ns = w9·ndp·sp11·er12
            gw[9] += g_ns * (c.ndp * c.sp11 * c.er12);
            let g_ndp = g_ns * w[9] * c.sp11 * c.er12;
            let g_sp11 = g_ns * w[9] * c.ndp * c.er12;
            let g_er12 = g_ns * w[9] * c.ndp * c.sp11;
            // ndp = nd^w10
            g_nd += g_ndp * (w[10] * c.ndp / c.nd);
            gw[10] += g_ndp * (c.ndp * c.nd.ln());
            // sp11 = s^w11
            g_s += g_sp11 * (w[11] * c.sp11 / c.s);
            gw[11] += g_sp11 * (c.sp11 * c.s.ln());
            // er12 = exp(w12·(1-r))
            g_r += g_er12 * (-w[12] * c.er12);
            gw[12] += g_er12 * ((1.0 - c.r) * c.er12);
        }
        _ => unreachable!(),
    }
    // difficulty backward: nd = clamp(nd_pre,1,10) ; nd_pre = w5·w2 + (1-w5)·nd0 ; nd0 = d + w4·(rating-3)
    let g_nd_pre = if c.nd_in_open { g_nd } else { 0.0 };
    gw[5] += g_nd_pre * (w[2] - c.nd0);
    gw[2] += g_nd_pre * w[5];
    let g_nd0 = g_nd_pre * (1.0 - w[5]);
    let g_d = g_nd0; // ∂nd0/∂d = 1
    gw[4] += g_nd0 * (c.rating - 3.0);
    // curve backward
    g_s += curve_bwd(c.r, c.dt, c.s, g_r);
    (g_s, g_d)
}

fn init_bwd(w: &[f64], c: &StepCache, g_s0: f64, g_d0: f64, gw: &mut [f64]) {
    // s_0 = clamp(w0 + w1·(rating-1), s_min, s_max)
    let g_ns = if c.ns_in_open { g_s0 } else { 0.0 };
    gw[0] += g_ns;
    gw[1] += g_ns * (c.rating - 1.0);
    // d_0 = clamp(w2 + w3·(rating-3), 1, 10)
    let g_initd = if c.nd_in_open { g_d0 } else { 0.0 };
    gw[2] += g_initd;
    gw[3] += g_initd * (c.rating - 3.0);
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
    let (mut s, mut d) = (0.0f64, 0.0f64);
    for k in 0..prior_r.len() {
        if k == 0 {
            let rating = prior_r[0] as f64;
            let ns = w[0] + w[1] * (rating - 1.0);
            let initd_pre = w[2] + w[3] * (rating - 3.0);
            caches.push(StepCache {
                branch: INIT,
                rating,
                ns_in_open: ns >= wc.s_min && ns <= wc.s_max,
                nd_in_open: initd_pre >= 1.0 && initd_pre <= 10.0,
                ..Default::default()
            });
            s = ns.clamp(wc.s_min, wc.s_max);
            d = initd_pre.clamp(1.0, 10.0);
        } else {
            let ((s_new, d_new), c) = step_fwd(w, s, d, prior_dt[k], prior_r[k] as f64, wc);
            s = s_new;
            d = d_new;
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
    init_bwd(w, &caches[0], g_s, g_d, gw);
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Dual;
    use crate::models::fsrs_v3::{retention_dual, INIT_W};

    fn forward_mode(prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, w: &[f64; NP], s_min: f64, s_max: f64) -> (f64, [f64; NP]) {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w[k]));
        let ret = retention_dual(prior_dt, prior_r, cur_dt, &wd, s_min, s_max);
        (ret.v, ret.g)
    }

    #[cfg(feature = "fp64")]
    #[test]
    fn fsrs3_analytic_grad_matches_forward_mode() {
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
