//! Hand-written reverse-mode VJP for the FSRS-4.5 BCE-loss gradient — a manual reverse-mode of the
//! exact forward in [`super::fsrs_v4dot5::retention`]. Replaces forward-mode `Dual<17>` for the
//! *training* gradient. Prediction keeps `Dual<0>`. f64 (FSRS-4.5 is an f64 algo).
//!
//! FSRS-4.5: power curve `(1+factor·t/s)^-0.5` (fixed decay), NO short-term branch, success same as
//! the FSRS-5/6 family, after-failure `min(nf, old_s)`, LINEAR difficulty reverting to `w4`
//! (`nd0 = d - w6·(rating-3)`), linear init `nd = clamp(w4 - w5·(rating-3), 1, 10)`.
//!
//! ⚠ Manual VJP of a SPECIFIC forward — change the formulas and re-derive. Guarded by
//! `fsrs45_analytic_grad_matches_forward_mode` (gated `--features fp64`).

const NP: usize = 17;
const DECAY: f64 = -0.5;

const SUCCESS: u8 = 1;
const FAIL: u8 = 2;
const INIT: u8 = 3;

pub struct WConsts {
    s_min: f64,
    s_max: f64,
    factor: f64, // 0.9^(1/DECAY) - 1
    exp_w8: f64,
}

pub fn wconsts(w: &[f64], s_min: f64, s_max: f64) -> WConsts {
    WConsts { s_min, s_max, factor: 0.9f64.powf(1.0 / DECAY) - 1.0, exp_w8: w[8].exp() }
}

#[derive(Clone, Copy, Default)]
struct CurveCache {
    out: f64,
    base: f64,
}

#[inline]
fn curve_fwd(t: f64, s: f64, wc: &WConsts) -> CurveCache {
    let base = 1.0 + wc.factor * t / s;
    CurveCache { out: base.powf(DECAY), base }
}

#[inline]
fn curve_bwd(c: &CurveCache, t: f64, s: f64, g_out: f64, wc: &WConsts) -> f64 {
    let g_base = g_out * DECAY * c.out / c.base;
    g_base * wc.factor * (-t / (s * s))
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
    nd0: f64,
    cr: CurveCache,
    r: f64,
    spb: f64,
    rrb: f64,
    term: f64,
    hard: f64,
    easy: f64,
    dp: f64,
    up: f64,
    sp1: f64,
    erc: f64,
    nf: f64,
}

fn step_fwd(w: &[f64], s: f64, d: f64, dt: f64, rating: f64, wc: &WConsts) -> ((f64, f64), StepCache) {
    let mut c = StepCache { s, d, dt, rating, ..Default::default() };
    let cr = curve_fwd(dt, s, wc);
    let r = cr.out;
    c.cr = cr;
    c.r = r;
    let ns = if rating > 1.0 {
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
        c.dp = dp;
        c.up = up;
        c.sp1 = sp1;
        c.erc = erc;
        c.nf = nf;
        nf.min(s) // cap at OLD stability
    };
    // linear difficulty reverting to w4
    let nd0 = d - w[6] * (rating - 3.0);
    let nd_pre = w[7] * w[4] + (1.0 - w[7]) * nd0;
    let nd = nd_pre.clamp(1.0, 10.0);
    c.nd0 = nd0;
    c.ns_in_open = ns >= wc.s_min && ns <= wc.s_max;
    c.nd_in_open = nd_pre >= 1.0 && nd_pre <= 10.0;
    ((ns.clamp(wc.s_min, wc.s_max), nd), c)
}

fn step_bwd(w: &[f64], c: &StepCache, g_s_new: f64, g_d_new: f64, wc: &WConsts, gw: &mut [f64]) -> (f64, f64) {
    let mut g_s = 0.0;
    let mut g_d = 0.0;
    let mut g_r = 0.0;
    let g_ns = if c.ns_in_open { g_s_new } else { 0.0 };
    let g_nd = g_d_new;

    // difficulty: nd_pre = w7·w4 + (1-w7)·nd0 ;  nd0 = d - w6·(rating-3)
    let g_nd_pre = if c.nd_in_open { g_nd } else { 0.0 };
    gw[7] += g_nd_pre * (w[4] - c.nd0);
    gw[4] += g_nd_pre * w[7];
    let g_nd0 = g_nd_pre * (1.0 - w[7]);
    g_d += g_nd0;
    gw[6] += g_nd0 * (-(c.rating - 3.0));

    match c.branch {
        SUCCESS => {
            g_s += g_ns * (1.0 + c.term);
            let g_term = g_ns * c.s;
            let b11d = 11.0 - c.d;
            gw[8] += g_term * c.term;
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
            g_s += g_spb * (-w[9] * c.spb / c.s);
            gw[9] += g_spb * (-(c.s.ln()) * c.spb);
            let erb = c.rrb + 1.0;
            g_r += g_rrb * (-w[10] * erb);
            gw[10] += g_rrb * ((1.0 - c.r) * erb);
        }
        FAIL => {
            // ns = min(nf, s) ; ties → nf
            let (g_nf, g_s_cap) = if c.nf <= c.s { (g_ns, 0.0) } else { (0.0, g_ns) };
            g_s += g_s_cap;
            gw[11] += g_nf * (c.dp * c.sp1 * c.erc);
            let g_dp = g_nf * w[11] * c.sp1 * c.erc;
            let g_sp1 = g_nf * w[11] * c.dp * c.erc;
            let g_erc = g_nf * w[11] * c.dp * c.sp1;
            g_d += g_dp * (-w[12] * c.dp / c.d);
            gw[12] += g_dp * (-(c.d.ln()) * c.dp);
            let g_up = g_sp1;
            g_s += g_up * (w[13] * c.up / (c.s + 1.0));
            gw[13] += g_up * (c.up * (c.s + 1.0).ln());
            g_r += g_erc * (-w[14] * c.erc);
            gw[14] += g_erc * ((1.0 - c.r) * c.erc);
        }
        _ => unreachable!(),
    }
    g_s += curve_bwd(&c.cr, c.dt, c.s, g_r, wc);
    (g_s, g_d)
}

fn init_bwd(w: &[f64], c: &StepCache, g_s0: f64, g_d0: f64, gw: &mut [f64]) {
    let rc = c.rating as usize;
    let g_ns = if c.ns_in_open { g_s0 } else { 0.0 };
    gw[rc - 1] += g_ns;
    // initd_pre = w4 - w5·(rating-3)
    let g_initd_pre = if c.nd_in_open { g_d0 } else { 0.0 };
    gw[4] += g_initd_pre;
    gw[5] += g_initd_pre * (-(c.rating - 3.0));
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
            let rc = prior_r[0] as usize;
            let ns = w[rc - 1];
            let initd_pre = w[4] - w[5] * (rating - 3.0);
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
    let cr = curve_fwd(cur_dt, s, wc);
    let p = cr.out;
    let denom = (p * (1.0 - p)).max(1e-12);
    let g_p = weight * (p - y) / denom;
    let mut g_s = curve_bwd(&cr, cur_dt, s, g_p, wc);
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
    use crate::models::fsrs_v4dot5::{retention_dual, INIT_W_DAYS};

    fn forward_mode(prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, w: &[f64; NP], s_min: f64, s_max: f64) -> (f64, [f64; NP]) {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w[k]));
        let ret = retention_dual(prior_dt, prior_r, cur_dt, &wd, s_min, s_max);
        (ret.v, ret.g)
    }

    #[cfg(feature = "fp64")]
    #[test]
    fn fsrs45_analytic_grad_matches_forward_mode() {
        let seqs: Vec<(Vec<f64>, Vec<i64>, f64)> = vec![
            (vec![0.0, 9.0, 1.5, 30.0, 100.0, 3.0], vec![3, 1, 3, 4, 2, 1], 7.0),
            (vec![0.0, 5.0, 2.0], vec![2, 3, 1], 12.0),
            (vec![0.0, 1.0, 1.0, 50.0, 5.0], vec![1, 2, 3, 4, 3], 3.0),
            (vec![0.0, 4.0, 6.0, 20.0], vec![4, 2, 3, 1], 15.0),
        ];
        let (s_min, s_max) = (0.0001, 36500.0);
        let mut w = INIT_W_DAYS;
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
