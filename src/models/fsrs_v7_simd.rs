//! f64×4 SIMD (AVX2) port of the FSRS-7 gradient — vectorizes the per-prefix recurrence
//! forward+backward across **4 rows per lane**. The math is identical to the scalar
//! [`super::fsrs_v7_grad`]; only the data type changes (`f64` → `wide::f64x4`), `if` → lane
//! `blend`, and `clamp` → `fast_max/fast_min`. Transcendentals use `wide`'s built-in f64×4
//! `exp`/`ln` (Cephes, ~1 ulp), so the result is NOT bit-identical to scalar libm but agrees to
//! ~1e-14 — far inside the ±0.0005 gate, and the **batching is unchanged** (same per-prefix
//! items, same order), so the training trajectory is preserved up to that tiny FP difference.
//!
//! The 4 lanes are 4 distinct per-prefix rows of (usually) different prefix length. The recurrence
//! runs `max_pos` steps; a lane whose prefix is shorter gets `rating==0` padding steps that FREEZE
//! its state (and pass the adjoint straight through in the backward), so its final-state prediction
//! is exactly its own state-after-prefix. Dummy/short lanes carry `weight==0` ⇒ zero gradient.
//! Rows with `pos==0` (empty prefix) are handled by the scalar path (caller), never here.

use wide::{f64x4, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt};

use super::fsrs_v7::NP;
use super::fsrs_v7_grad::WConsts;
use crate::features::{Dataset, Row};

const S_MAX: f64 = 36500.0;
const D_MIN: f64 = 1.0;
const D_MAX: f64 = 10.0;
const MIN_R: f64 = 1e-5;
const MAX_R: f64 = 1.0 - 1e-5;

#[inline(always)]
fn sp(x: f64) -> f64x4 {
    f64x4::from(x)
}
#[inline(always)]
fn clamp4(x: f64x4, lo: f64, hi: f64) -> f64x4 {
    x.fast_max(sp(lo)).fast_min(sp(hi))
}

// ===================== forgetting curve =====================

struct Curve4 {
    out: f64x4,
    a: f64x4,
    bv: f64x4,
    decay1: f64x4,
    factor1: f64x4,
    b1: f64x4,
    r1: f64x4,
    q1: f64x4,
    e1: f64x4,
    m1: f64x4,
    p35: f64x4,
    b2: f64x4,
    r2: f64x4,
    ex34: f64x4,
    weight1: f64x4,
    weight2: f64x4,
    wsum: f64x4,
    ret: f64x4,
    p31: f64x4,
    se: f64x4,
    ln_sf: f64x4,
    ln_b1: f64x4,
    ln_b2: f64x4,
    ln_s: f64x4,
}

#[allow(clippy::too_many_arguments)]
fn curve4_fwd(w: &WLanes, t: f64x4, s: f64x4, sf: f64x4, d: f64x4, ln_s: f64x4, ln_sf: f64x4) -> Curve4 {
    let t = t.fast_max(sp(0.0));
    let a = t / sf;
    let bv = t / s;
    let p35 = ((w.w[33] - sp(0.3)) * ln_sf).exp();
    let m1 = w.w[23] * p35;
    let dm1 = clamp4(m1, 0.01, 0.95);
    let decay1 = -dm1;
    let q1 = w.ln_w25 / decay1;
    let e1 = q1.fast_min(sp(60.0)).exp();
    let factor1 = e1 - sp(1.0);
    let b1 = a * factor1 + sp(1.0);
    let ln_b1 = b1.ln();
    let r1 = (decay1 * ln_b1).exp();
    let ex34 = ((d - sp(5.0)) * (w.w[32] - sp(0.3))).exp();
    // decay2/inv2/p28/factor2 are loop-invariant ⇒ taken from `w` (hoisted in WLanes::new).
    let b2 = bv * w.factor2 * ex34 + sp(1.0);
    let ln_b2 = b2.ln();
    let r2 = (w.decay2 * ln_b2).exp();
    let p31 = ((-w.w[29]) * ln_sf).exp();
    let weight1 = w.w[27] * p31;
    // se = s32·ex33 = exp(w30·ln_s + (w31−0.5)·(d−5)) — ONE exp instead of two (iter 6 fusion;
    // ~1-ulp reassociation, well inside the gate). weight2 = w28·se.
    let se = (w.w[30] * ln_s + (d - sp(5.0)) * (w.w[31] - sp(0.5))).exp();
    let weight2 = w.w[28] * se;
    let wsum = weight1 + weight2;
    let num = weight1 * r1 + weight2 * r2;
    let ret = num / wsum;
    let out = ret * sp(1.0 - 2e-5) + sp(1e-5);
    Curve4 {
        out, a, bv, decay1, factor1, b1, r1, q1, e1, m1, p35, b2, r2,
        ex34, weight1, weight2, wsum, ret, p31, se, ln_sf, ln_b1, ln_b2, ln_s,
    }
}

#[allow(clippy::too_many_arguments)]
fn curve4_bwd(w: &WLanes, c: &Curve4, t: f64x4, s: f64x4, sf: f64x4, d: f64x4, g_out: f64x4, g_r1_extra: f64x4, gw: &mut [f64x4; NP]) -> (f64x4, f64x4, f64x4) {
    let z = sp(0.0);
    let t = t.fast_max(z);
    let g_ret = g_out * sp(1.0 - 2e-5);
    let g_num = g_ret / c.wsum;
    let g_wsum = -g_ret * c.ret / c.wsum;
    let g_weight1 = g_num * c.r1 + g_wsum;
    let g_weight2 = g_num * c.r2 + g_wsum;
    let g_r1 = g_num * c.weight1 + g_r1_extra;
    let g_r2 = g_num * c.weight2;
    // weight2 = w28·se ; se = exp(w30·ln_s + (w31−0.5)·(d−5)). g_se folds in the shared ×se factor.
    gw[28] += g_weight2 * c.se;
    let g_se = g_weight2 * w.w[28] * c.se;
    let mut g_d = g_se * (w.w[31] - sp(0.5));
    gw[31] += g_se * (d - sp(5.0));
    let mut g_s = g_se * w.w[30] / s;
    gw[30] += g_se * c.ln_s;
    gw[27] += g_weight1 * c.p31;
    let g_p31 = g_weight1 * w.w[27];
    let mut g_sf = g_p31 * (-w.w[29]) * (c.p31 / sf);
    gw[29] += g_p31 * (-(c.p31 * c.ln_sf));
    let g_b2 = g_r2 * w.decay2 * (c.r2 / c.b2);
    let mut g_decay2 = g_r2 * c.r2 * c.ln_b2;
    let g_bv = g_b2 * w.factor2 * c.ex34;
    let g_factor2 = g_b2 * c.bv * c.ex34;
    let g_ex34 = g_b2 * c.bv * w.factor2;
    g_d += g_ex34 * c.ex34 * (w.w[32] - sp(0.3));
    gw[32] += g_ex34 * c.ex34 * (d - sp(5.0));
    let g_p28 = g_factor2;
    gw[26] += g_p28 * w.inv2 * (w.p28 / w.w[26]);
    let g_inv2 = g_p28 * w.p28 * w.ln_w26;
    g_decay2 += g_inv2 * (-sp(1.0) / (w.decay2 * w.decay2));
    let g_dm2 = -g_decay2;
    let g_m2 = (w.m2.cmp_gt(sp(0.01)) & w.m2.cmp_lt(sp(0.95))).blend(g_dm2, z);
    gw[24] += g_m2;
    g_s += g_bv * (-t / (s * s));
    let g_b1 = g_r1 * c.decay1 * (c.r1 / c.b1);
    let mut g_decay1 = g_r1 * c.r1 * c.ln_b1;
    let g_a = g_b1 * c.factor1;
    let g_factor1 = g_b1 * c.a;
    let g_e1 = g_factor1;
    let g_q1c = g_e1 * c.e1;
    let g_q1 = c.q1.cmp_lt(sp(60.0)).blend(g_q1c, z);
    let g_lw25 = g_q1 / c.decay1;
    g_decay1 += g_q1 * (-w.ln_w25 / (c.decay1 * c.decay1));
    gw[25] += g_lw25 / w.w[25];
    let g_dm1 = -g_decay1;
    let g_m1 = (c.m1.cmp_gt(sp(0.01)) & c.m1.cmp_lt(sp(0.95))).blend(g_dm1, z);
    gw[23] += g_m1 * c.p35;
    let g_p35 = g_m1 * w.w[23];
    g_sf += g_p35 * (w.w[33] - sp(0.3)) * (c.p35 / sf);
    gw[33] += g_p35 * c.p35 * c.ln_sf;
    g_sf += g_a * (-t / (sf * sf));
    (g_s, g_sf, g_d)
}

// ===================== stability after review =====================

struct Stab4 {
    out: f64x4,
    nsf_fail: f64x4,
    pls: f64x4,
    sinc: f64x4,
    ls_sinc: f64x4,
    aa: f64x4,
    bb: f64x4,
    cc: f64x4,
    expr: f64x4,
    qbase: f64x4,
    rexp: f64x4,
    hard: f64x4,
    easy: f64x4,
    ln_ls: f64x4,
    ln_ls1: f64x4,
}

#[allow(clippy::too_many_arguments)]
fn stab4_fwd(w: &WLanes, last_s: f64x4, last_d: f64x4, r: f64x4, rating: f64x4, start: usize, aa: f64x4, ln_ls: f64x4) -> Stab4 {
    let one = sp(1.0);
    let hard = rating.cmp_eq(sp(2.0)).blend(w.w[start + 6], one);
    let easy = rating.cmp_eq(sp(4.0)).blend(w.w[start + 7], one);
    let ln_ls1 = (last_s + one).ln();
    let qbase = (w.w[start + 4] * ln_ls1).exp();
    let rexp = ((one - r) * w.w[start + 5]).exp();
    let nsf_fail = w.w[start + 3] * (qbase - one) * rexp;
    let pls = last_s.fast_min(nsf_fail);
    let bb = sp(11.0) - last_d;
    let cc = ((-w.w[start + 1]) * ln_ls).exp();
    let expr = ((one - r) * w.w[start + 2]).exp();
    let sinc = aa * bb * cc * (expr - one) * hard * easy + one;
    let ls_sinc = last_s * sinc;
    let nss = pls.fast_max(ls_sinc);
    let out = rating.cmp_gt(one).blend(nss, pls);
    Stab4 { out, nsf_fail, pls, sinc, ls_sinc, aa, bb, cc, expr, qbase, rexp, hard, easy, ln_ls, ln_ls1 }
}

#[allow(clippy::too_many_arguments)]
fn stab4_bwd(w: &WLanes, c: &Stab4, last_s: f64x4, r: f64x4, rating: f64x4, start: usize, g_out: f64x4, gw: &mut [f64x4; NP]) -> (f64x4, f64x4, f64x4) {
    let one = sp(1.0);
    let z = sp(0.0);
    let gt1 = rating.cmp_gt(one);
    let g_nss = gt1.blend(g_out, z);
    let g_pls_direct = gt1.blend(z, g_out);
    let g_pls_from_nss = c.pls.cmp_ge(c.ls_sinc).blend(g_nss, z);
    let g_ls_sinc = c.ls_sinc.cmp_gt(c.pls).blend(g_nss, z);
    let mut g_last_s = g_ls_sinc * c.sinc;
    let g_sinc = g_ls_sinc * last_s;
    let g_pls = g_pls_direct + g_pls_from_nss;
    g_last_s += last_s.cmp_le(c.nsf_fail).blend(g_pls, z);
    let g_nsf_fail = c.nsf_fail.cmp_lt(last_s).blend(g_pls, z);
    let em1 = c.expr - one;
    let g_prod = g_sinc;
    let prod = c.aa * c.bb * c.cc * em1 * c.hard * c.easy;
    gw[start] += g_prod * prod;
    let g_bb = g_prod * (c.aa * c.cc * em1 * c.hard * c.easy);
    let g_cc = g_prod * (c.aa * c.bb * em1 * c.hard * c.easy);
    let g_em1 = g_prod * (c.aa * c.bb * c.cc * c.hard * c.easy);
    gw[start + 6] += rating.cmp_eq(sp(2.0)).blend(g_prod * (c.aa * c.bb * c.cc * em1 * c.easy), z);
    gw[start + 7] += rating.cmp_eq(sp(4.0)).blend(g_prod * (c.aa * c.bb * c.cc * em1 * c.hard), z);
    let g_last_d = g_bb * (-one);
    g_last_s += g_cc * (-w.w[start + 1]) * (c.cc / last_s);
    gw[start + 1] += g_cc * (-(c.cc * c.ln_ls));
    let mut g_r = g_em1 * c.expr * (-w.w[start + 2]);
    gw[start + 2] += g_em1 * c.expr * (one - r);
    let q = c.qbase - one;
    gw[start + 3] += g_nsf_fail * (q * c.rexp);
    let g_q = g_nsf_fail * w.w[start + 3] * c.rexp;
    let g_rexp = g_nsf_fail * w.w[start + 3] * q;
    g_last_s += g_q * w.w[start + 4] * (c.qbase / (last_s + one));
    gw[start + 4] += g_q * c.qbase * c.ln_ls1;
    g_r += g_rexp * c.rexp * (-w.w[start + 5]);
    gw[start + 5] += g_rexp * c.rexp * (one - r);
    (g_last_s, g_last_d, g_r)
}

// ===================== next difficulty =====================

fn next_d4_fwd(w: &WLanes, last_d: f64x4, rating: f64x4, r: f64x4) -> (f64x4, f64x4, f64x4) {
    let delta_d_base = -w.w[6] * (rating - sp(3.0));
    let is_lapse = rating.cmp_eq(sp(1.0));
    let delta_d = is_lapse.blend(delta_d_base * (r + sp(0.1)), delta_d_base);
    let new_d = last_d + (sp(10.0) - last_d) * delta_d / sp(9.0);
    let out_pre = sp(0.01) * w.init + sp(0.99) * new_d;
    (clamp4(out_pre, D_MIN, D_MAX), out_pre, delta_d)
}

#[allow(clippy::too_many_arguments)]
fn next_d4_bwd(w: &WLanes, out_pre: f64x4, delta_d: f64x4, last_d: f64x4, rating: f64x4, r: f64x4, g_out: f64x4, gw: &mut [f64x4; NP]) -> (f64x4, f64x4) {
    let z = sp(0.0);
    let in_range = out_pre.cmp_gt(sp(D_MIN)) & out_pre.cmp_lt(sp(D_MAX));
    let g_out_pre = in_range.blend(g_out, z);
    let g_init = g_out_pre * sp(0.01);
    let g_new_d = g_out_pre * sp(0.99);
    gw[4] += g_init;
    gw[5] += g_init * (-w.exp3w5 * sp(3.0));
    let g_last_d = g_new_d * (sp(1.0) - delta_d / sp(9.0));
    let g_delta_d = g_new_d * (sp(10.0) - last_d) / sp(9.0);
    let rm3 = rating - sp(3.0);
    let is_lapse = rating.cmp_eq(sp(1.0));
    // lapse: delta_d_eff = (-w6*(rating-3))*(r+0.1) ; else delta_d = -w6*(rating-3)
    gw[6] += is_lapse.blend(g_delta_d * (-rm3) * (r + sp(0.1)), g_delta_d * (-rm3));
    let g_r = is_lapse.blend(g_delta_d * (-w.w[6] * rm3), z);
    (g_last_d, g_r)
}

// ===================== one recurrence step =====================

struct Step4 {
    s0: f64x4,
    d0: f64x4,
    sf0: f64x4,
    last_s: f64x4,
    last_d: f64x4,
    last_sf: f64x4,
    dt: f64x4,
    rating: f64x4,
    nth0: bool,
    curve: Curve4,
    slow: Stab4,
    fast: Stab4,
    nd_out_pre: f64x4,
    nd_delta_d: f64x4,
    ns3: f64x4,
    nsf3: f64x4,
}

fn step4_fwd(w: &WLanes, delta_t: f64x4, rating: f64x4, state: (f64x4, f64x4, f64x4), nth0: bool, s_min: f64) -> ((f64x4, f64x4, f64x4), Step4) {
    let (s0, d0, sf0) = state;
    let last_s = clamp4(s0, s_min, S_MAX);
    let last_d = clamp4(d0, D_MIN, D_MAX);
    let last_sf = clamp4(sf0, s_min, S_MAX);
    let dt = delta_t.fast_max(sp(0.0));
    let ln_last_s = last_s.ln();
    let ln_last_sf = last_sf.ln();
    let curve = curve4_fwd(w, dt, last_s, last_sf, last_d, ln_last_s, ln_last_sf);
    let r = curve.out;
    let r1 = curve.r1;
    let slow = stab4_fwd(w, last_s, last_d, r, rating, 7, w.aa7, ln_last_s);
    let fast = stab4_fwd(w, last_sf, last_d, r1, rating, 15, w.aa16, ln_last_sf);
    let (nd1, nd_out_pre, nd_delta_d) = next_d4_fwd(w, last_d, rating, r);
    // post-lapse short reset: on a lapse cap s_short at 0.8 * post-lapse s_long.
    let is_lapse = rating.cmp_eq(sp(1.0));
    let nsf_pre = is_lapse.blend(fast.out.fast_min(sp(0.8) * slow.out), fast.out);
    let mut ns = slow.out;
    let mut nsf = nsf_pre;
    let mut nd = nd1;
    // init override (first review of every lane's prefix; at t==0 all lanes have s0==0).
    if nth0 {
        let init_mask = s0.cmp_eq(sp(0.0));
        let rc = clamp4(rating, 1.0, 4.0);
        let init_s = w.init_s_by_rating(rc);
        let init_d = clamp4(w.w[4] - (w.w[5] * (rc - sp(1.0))).exp() + sp(1.0), D_MIN, D_MAX);
        ns = init_mask.blend(init_s, ns);
        nsf = init_mask.blend(sp(0.8) * init_s, nsf);
        nd = init_mask.blend(init_d, nd);
    }
    // padding: rating==0 lanes keep their previous state (frozen).
    let pad = rating.cmp_eq(sp(0.0));
    ns = pad.blend(last_s, ns);
    nsf = pad.blend(last_sf, nsf);
    nd = pad.blend(last_d, nd);
    let ns3 = ns;
    let nsf3 = nsf;
    let out = (clamp4(ns, s_min, S_MAX), nd, clamp4(nsf, s_min, S_MAX));
    let cache = Step4 {
        s0, d0, sf0, last_s, last_d, last_sf, dt, rating, nth0, curve, slow, fast, nd_out_pre, nd_delta_d, ns3, nsf3,
    };
    (out, cache)
}

fn step4_bwd(w: &WLanes, c: &Step4, g_out: (f64x4, f64x4, f64x4), gw: &mut [f64x4; NP], s_min: f64) -> (f64x4, f64x4, f64x4) {
    let z = sp(0.0);
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    let g_ns3 = (c.ns3.cmp_gt(sp(s_min)) & c.ns3.cmp_lt(sp(S_MAX))).blend(g_ns_out, z);
    let g_nsf3 = (c.nsf3.cmp_gt(sp(s_min)) & c.nsf3.cmp_lt(sp(S_MAX))).blend(g_nsf_out, z);
    let g_nd3 = g_nd_out;
    // padding (rating==0): output state = input state (passthrough); stab/curve/next_d get 0.
    let pad = c.rating.cmp_eq(z);
    let g_ns2 = pad.blend(z, g_ns3);
    let g_nsf2 = pad.blend(z, g_nsf3);
    let g_nd2 = pad.blend(z, g_nd3);
    let g_last_s_extra = pad.blend(g_ns3, z);
    let g_last_sf_extra = pad.blend(g_nsf3, z);
    let g_last_d_extra = pad.blend(g_nd3, z);
    // init override (t==0): route to init params, zero the downstream.
    let (g_ns1, g_nsf1, g_nd1) = if c.nth0 {
        let init_mask = c.s0.cmp_eq(z);
        let rc = clamp4(c.rating, 1.0, 4.0);
        // init_s = w[rc-1]; nsf = 0.8*init_s -> accumulate to the per-rating init weight.
        w.accum_init_s_grad(rc, init_mask.blend(g_ns2 + g_nsf2 * sp(0.8), z), gw);
        let id_pre = w.w[4] - (w.w[5] * (rc - sp(1.0))).exp() + sp(1.0);
        let id_in = id_pre.cmp_gt(sp(D_MIN)) & id_pre.cmp_lt(sp(D_MAX));
        let g_id = (init_mask & id_in).blend(g_nd2, z);
        gw[4] += g_id;
        gw[5] += g_id * (-((w.w[5] * (rc - sp(1.0))).exp()) * (rc - sp(1.0)));
        (init_mask.blend(z, g_ns2), init_mask.blend(z, g_nsf2), init_mask.blend(z, g_nd2))
    } else {
        (g_ns2, g_nsf2, g_nd2)
    };
    // post-lapse min routing.
    let is_lapse = c.rating.cmp_eq(sp(1.0));
    let lapse_fast = c.fast.out.cmp_le(sp(0.8) * c.slow.out);
    let g_fast_out = is_lapse.blend(lapse_fast.blend(g_nsf1, z), g_nsf1);
    let g_slow_from_relearn = is_lapse.blend(lapse_fast.blend(z, g_nsf1 * sp(0.8)), z);
    let (g_ls_a, g_ld_a, g_r_long) = stab4_bwd(w, &c.slow, c.last_s, c.curve.out, c.rating, 7, g_ns1 + g_slow_from_relearn, gw);
    let (g_lsf_b, g_ld_b, g_r1_short) = stab4_bwd(w, &c.fast, c.last_sf, c.curve.r1, c.rating, 15, g_fast_out, gw);
    let (g_ld_c, g_r_nextd) = next_d4_bwd(w, c.nd_out_pre, c.nd_delta_d, c.last_d, c.rating, c.curve.out, g_nd1, gw);
    let (g_ls_d, g_lsf_d, g_ld_d) = curve4_bwd(w, &c.curve, c.dt, c.last_s, c.last_sf, c.last_d, g_r_long + g_r_nextd, g_r1_short, gw);
    let g_last_s = g_ls_a + g_ls_d + g_last_s_extra;
    let g_last_sf = g_lsf_b + g_lsf_d + g_last_sf_extra;
    let g_last_d = g_ld_a + g_ld_b + g_ld_c + g_ld_d + g_last_d_extra;
    let g_s0 = (c.s0.cmp_gt(sp(s_min)) & c.s0.cmp_lt(sp(S_MAX))).blend(g_last_s, z);
    let g_d0 = (c.d0.cmp_gt(sp(D_MIN)) & c.d0.cmp_lt(sp(D_MAX))).blend(g_last_d, z);
    let g_sf0 = (c.sf0.cmp_gt(sp(s_min)) & c.sf0.cmp_lt(sp(S_MAX))).blend(g_last_sf, z);
    (g_s0, g_d0, g_sf0)
}

// ===================== per-lane weights (splatted) =====================

/// The 34 params + hoisted loop-invariants, each splatted to all 4 lanes (params are shared
/// across lanes — only the per-row data differs).
pub struct WLanes {
    w: [f64x4; NP],
    ln_w25: f64x4,
    ln_w26: f64x4,
    aa7: f64x4,
    aa16: f64x4,
    init: f64x4,
    exp3w5: f64x4,
    // Long-term decay block — depends only on w24/w26 (NOT per-step state), so hoisted out of the
    // per-timestep curve. Computed once via the same f64×4 ops the per-step path used ⇒ byte-identical.
    m2: f64x4,
    decay2: f64x4,
    inv2: f64x4,
    p28: f64x4,
    factor2: f64x4,
}

impl WLanes {
    pub fn new(params: &[f64], wc: &WConsts) -> Self {
        let ln_w26 = sp(wc.ln_w26());
        let m2 = sp(params[24]);
        let decay2 = -clamp4(m2, 0.01, 0.95);
        let inv2 = sp(1.0) / decay2;
        let p28 = (inv2 * ln_w26).exp();
        let factor2 = p28 - sp(1.0);
        WLanes {
            w: std::array::from_fn(|k| sp(params[k])),
            ln_w25: sp(wc.ln_w25()),
            ln_w26,
            aa7: sp(wc.aa7()),
            aa16: sp(wc.aa16()),
            init: sp(wc.init()),
            exp3w5: sp(wc.exp3w5()),
            m2,
            decay2,
            inv2,
            p28,
            factor2,
        }
    }
    /// `init_s = w[rc-1]` selected per lane by the (clamped) rating `rc ∈ {1,2,3,4}`.
    #[inline]
    fn init_s_by_rating(&self, rc: f64x4) -> f64x4 {
        let mut out = self.w[0];
        out = rc.cmp_eq(sp(2.0)).blend(self.w[1], out);
        out = rc.cmp_eq(sp(3.0)).blend(self.w[2], out);
        out = rc.cmp_eq(sp(4.0)).blend(self.w[3], out);
        out
    }
    /// Accumulate `g` into the per-rating init weight gw[rc-1] (lane-selected).
    #[inline]
    fn accum_init_s_grad(&self, rc: f64x4, g: f64x4, gw: &mut [f64x4; NP]) {
        let z = sp(0.0);
        gw[0] += rc.cmp_eq(sp(1.0)).blend(g, z);
        gw[1] += rc.cmp_eq(sp(2.0)).blend(g, z);
        gw[2] += rc.cmp_eq(sp(3.0)).blend(g, z);
        gw[3] += rc.cmp_eq(sp(4.0)).blend(g, z);
    }
}

// ===================== group-of-4 gradient driver =====================

/// One per-prefix row's data needed to build a lane.
struct LaneRow<'a> {
    prior_dt: &'a [f64],
    prior_r: &'a [i64],
    cur_dt: f64,
    y: f64,
    weight: f64,
}

/// Accumulate d(Σ weight·BCE)/dw for up to 4 per-prefix rows (one per lane) into the scalar `gw`.
/// `s_min` is the stability floor. Rows must have `pos ≥ 1` (the recurrence runs ≥1 step).
fn grad_group(wl: &WLanes, lanes: &[LaneRow], s_min: f64, gw: &mut [f64], caches: &mut Vec<Step4>) {
    caches.clear();
    // max prefix length over the (≤4) real lanes.
    let max_steps = lanes.iter().map(|l| l.prior_r.len()).max().unwrap_or(0);
    debug_assert!(max_steps >= 1);
    // Final-prediction lane arrays (dummy lanes default to weight 0 ⇒ no gradient).
    let mut cur_dt_a = [1.0f64; 4];
    let mut y_a = [0.0f64; 4];
    let mut w_a = [0.0f64; 4];
    for (l, lane) in lanes.iter().enumerate() {
        cur_dt_a[l] = lane.cur_dt;
        y_a[l] = lane.y;
        w_a[l] = lane.weight;
    }
    let cur_dt = f64x4::from(cur_dt_a);
    let y = f64x4::from(y_a);
    let weight = f64x4::from(w_a);

    // forward — build each step's dt / rating (rating==0 = padding) inline, no per-step alloc.
    let mut state = (sp(0.0), sp(0.0), sp(0.0));
    for t in 0..max_steps {
        let mut dt_a = [0.0f64; 4];
        let mut rt_a = [0.0f64; 4];
        for (l, lane) in lanes.iter().enumerate() {
            if t < lane.prior_r.len() {
                dt_a[l] = lane.prior_dt[t];
                rt_a[l] = lane.prior_r[t] as f64;
            }
        }
        let (ns, cache) = step4_fwd(wl, f64x4::from(dt_a), f64x4::from(rt_a), state, t == 0, s_min);
        state = ns;
        caches.push(cache);
    }
    let (s, d, sf) = state;
    let fc = curve4_fwd(wl, cur_dt, s, sf, d, s.ln(), sf.ln());
    let r_raw = fc.out;
    let r = clamp4(r_raw, MIN_R, MAX_R);
    let g_r = weight * (r - y) / (r * (sp(1.0) - r));
    let in_range = r_raw.cmp_gt(sp(MIN_R)) & r_raw.cmp_lt(sp(MAX_R));
    let g_rraw = in_range.blend(g_r, sp(0.0));

    let mut gw4: [f64x4; NP] = [sp(0.0); NP];
    let (mut g_s, mut g_sf, mut g_d) = curve4_bwd(wl, &fc, cur_dt, s, sf, d, g_rraw, sp(0.0), &mut gw4);
    for t in (0..max_steps).rev() {
        let (gs, gd, gsf) = step4_bwd(wl, &caches[t], (g_s, g_d, g_sf), &mut gw4, s_min);
        g_s = gs;
        g_d = gd;
        g_sf = gsf;
    }
    for k in 0..NP {
        gw[k] += gw4[k].reduce_add();
    }
}

/// SIMD gradient of `Σ_i weight_i·BCE(p_i, y_i)` over the rows in `idx` (FSRS-7's per-batch grad).
/// Rows with `pos==0` (empty prefix) are returned in `scalar_fallback` for the caller to handle
/// via the scalar path; everything else is done 4-wide. Accumulates into `gw` (length NP).
pub fn grad_simd(ds: &Dataset, rows: &[Row], weights: &[f64], idx: &[usize], wl: &WLanes, s_min: f64, gw: &mut [f64], scalar_fallback: &mut Vec<usize>) {
    // Sort the batch by prefix length (pos) so each group of 4 has near-equal length (minimal
    // padding). pos==0 rows split off to the scalar path.
    let mut order: Vec<usize> = Vec::with_capacity(idx.len());
    for &i in idx {
        if rows[i].pos == 0 {
            scalar_fallback.push(i);
        } else {
            order.push(i);
        }
    }
    order.sort_by_key(|&i| rows[i].pos);
    // Scratch buffers reused across all groups (one alloc per `grad_simd` call, not per group).
    let mut caches: Vec<Step4> = Vec::new();
    let mut lanes: Vec<LaneRow> = Vec::with_capacity(4);
    let mut g = 0usize;
    while g < order.len() {
        let end = (g + 4).min(order.len());
        lanes.clear();
        for &i in &order[g..end] {
            let row = &rows[i];
            lanes.push(LaneRow {
                prior_dt: ds.prior_dt_active(row),
                prior_r: ds.prior_ratings(row),
                cur_dt: row.delta_t,
                y: row.y as f64,
                weight: weights[i],
            });
        }
        grad_group(wl, &lanes, s_min, gw, &mut caches);
        g = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::fsrs_v7::INIT_W;
    use crate::models::fsrs_v7_grad::{grad_one, wconsts, StepCacheBox};

    #[test]
    fn simd_grad_matches_scalar_grad() {
        // Several rows of varying prefix length -> a single group of 4 + tail.
        let card_dt: Vec<f64> = vec![0.0, 0.3, 9.0, 1.5, 30.0, 0.02, 100.0, 5.0, 2.0, 14.0];
        let card_r: Vec<i64> = vec![3, 1, 3, 4, 2, 1, 3, 2, 4, 3];
        let positions = [1usize, 2, 4, 5, 7, 9];
        let ys = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let ws = [0.5, 1.0, 0.8, 1.2, 0.3, 0.9];
        let s_min = 0.0001;
        let mut w = INIT_W;
        for (k, wk) in w.iter_mut().enumerate() {
            *wk += 0.013 * ((k as f64 + 1.0).cos());
        }
        let wc = wconsts(&w, s_min);
        let wl = WLanes::new(&w, &wc);

        // scalar reference: sum of grad_one over the rows.
        let mut g_scalar = vec![0.0f64; NP];
        let mut caches: Vec<StepCacheBox> = Vec::new();
        for (j, &pos) in positions.iter().enumerate() {
            grad_one(&w, &card_dt[..pos], &card_r[..pos], card_dt[pos], ys[j], ws[j], &wc, &mut g_scalar, &mut caches);
        }

        // SIMD: build LaneRow list and run group-of-4 driver directly.
        let mut g_simd = vec![0.0f64; NP];
        let lane_rows: Vec<LaneRow> = positions
            .iter()
            .enumerate()
            .map(|(j, &pos)| LaneRow {
                prior_dt: &card_dt[..pos],
                prior_r: &card_r[..pos],
                cur_dt: card_dt[pos],
                y: ys[j],
                weight: ws[j],
            })
            .collect();
        let mut caches: Vec<Step4> = Vec::new();
        let mut g0 = 0;
        while g0 < lane_rows.len() {
            let end = (g0 + 4).min(lane_rows.len());
            grad_group(&wl, &lane_rows[g0..end], s_min, &mut g_simd, &mut caches);
            g0 = end;
        }

        for k in 0..NP {
            let diff = (g_simd[k] - g_scalar[k]).abs();
            let tol = 1e-7 + 1e-7 * g_scalar[k].abs();
            assert!(diff < tol, "param {k}: simd {} vs scalar {} (diff {diff:e})", g_simd[k], g_scalar[k]);
        }
    }
}
