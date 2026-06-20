//! Hand-written forward + reverse-mode backward (analytic VJP) for the FSRS-7 BCE loss — an
//! f64 port of `fsrs-rs-speed-autoresearch/fsrs-rs/src/analytic.rs` (scalar path). Replaces the
//! forward-mode `Dual<34>` autodiff in [`super::fsrs_v7`]: forward-mode costs ~P× (P=34) the
//! value pass on every op; reverse-mode costs ~1 forward + ~1 backward regardless of P, so the
//! gradient is ~10× cheaper. Same per-prefix-item batching as before ⇒ the training trajectory
//! is unchanged (only the gradient's FP reduction order differs, ~1e-12).
//!
//! The forward (`*_fwd`) mirrors `fsrs_v7`'s math EXACTLY and stashes the intermediates each
//! step's backward needs; the backward (`*_bwd`) is the reverse-mode adjoint of that same
//! forward (single source of truth). `predict_one` is the value-only forward (for prediction);
//! `grad_one` does forward+backward for one row, accumulating d(weight·BCE)/dw into `gw`.

use crate::autodiff::round_scalar as r;

const S_MAX: f64 = 36500.0;
const D_MIN: f64 = 1.0;
const D_MAX: f64 = 10.0;
const MIN_R: f64 = 1e-5;
const MAX_R: f64 = 1.0 - 1e-5;

#[inline(always)]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Loop-invariant (weight-only) subexpressions, computed once per gradient/predict call instead
/// of once per timestep. `s_min` is threaded here too (the per-config stability floor).
pub struct WConsts {
    s_min: f64,
    ln_w25: f64, // ln(base1 = w[25])
    ln_w26: f64, // ln(base2 = w[26])
    aa7: f64,    // exp(w[7]-1.5)   long-stab sinc base (start=7)
    aa16: f64,   // exp(w[15]-1.5)  short-stab sinc base (start=15)
    init: f64,   // w4 - exp(3*w5) + 1   (mean-reversion init_d at rating 4)
    exp3w5: f64, // exp(3*w5)            (next_d backward: d(init)/d(w5))
}

pub fn wconsts(w: &[f64], s_min: f64) -> WConsts {
    WConsts {
        s_min,
        ln_w25: r(w[25].ln()),
        ln_w26: r(w[26].ln()),
        aa7: r(r(w[7] - 1.5).exp()),
        aa16: r(r(w[15] - 1.5).exp()),
        init: r(r(w[4] - r(r(w[5] * 3.0).exp())) + 1.0),
        exp3w5: r(r(w[5] * 3.0).exp()),
    }
}

// Accessors for the SIMD path (`fsrs_v7_simd`), which splats these into f64x4 lanes.
impl WConsts {
    pub fn ln_w25(&self) -> f64 {
        self.ln_w25
    }
    pub fn ln_w26(&self) -> f64 {
        self.ln_w26
    }
    pub fn aa7(&self) -> f64 {
        self.aa7
    }
    pub fn aa16(&self) -> f64 {
        self.aa16
    }
    pub fn init(&self) -> f64 {
        self.init
    }
    pub fn exp3w5(&self) -> f64 {
        self.exp3w5
    }
}

// ===================== forgetting curve =====================

struct CurveCache {
    out: f64,
    a: f64,
    bv: f64,
    decay1: f64,
    factor1: f64,
    b1: f64,
    r1: f64,
    q1: f64,
    e1: f64,
    m1: f64,
    p35: f64,
    decay2: f64,
    factor2: f64,
    b2: f64,
    r2: f64,
    inv2: f64,
    p28: f64,
    m2: f64,
    ex34: f64,
    weight1: f64,
    weight2: f64,
    wsum: f64,
    ret: f64,
    p31: f64,
    s32: f64,
    ex33: f64,
    ln_sf: f64,
    ln_b1: f64,
    ln_b2: f64,
    ln_s: f64,
    ln_w25: f64,
    ln_w26: f64,
}

#[allow(clippy::too_many_arguments)]
fn curve_fwd(w: &[f64], t: f64, s: f64, sf: f64, d: f64, ln_w25: f64, ln_w26: f64, ln_s: f64, ln_sf: f64) -> CurveCache {
    let t = t.max(0.0);
    let a = r(t / sf);
    let bv = r(t / s);
    // short-trace recall r1 (driven by s_short=sf):
    let p35 = r(r(r(w[33] - 0.3) * ln_sf).exp()); // sf^(w33-0.3)
    let m1 = r(w[23] * p35);
    let dm1 = clamp(m1, 0.01, 0.95);
    let decay1 = -dm1;
    let q1 = r(ln_w25 / decay1);
    let e1 = r(q1.min(60.0).exp());
    let factor1 = r(e1 - 1.0);
    let b1 = r(r(a * factor1) + 1.0);
    let ln_b1 = r(b1.ln());
    let r1 = r(r(decay1 * ln_b1).exp()); // b1^decay1
    // long-trace recall r2 (D scales the horizontal time-scale; decay2 not d-modulated):
    let ex34 = r(r(r(d - 5.0) * r(w[32] - 0.3)).exp());
    let m2 = w[24];
    let dm2 = clamp(m2, 0.01, 0.95);
    let decay2 = -dm2;
    let inv2 = r(1.0 / decay2);
    let p28 = r(r(inv2 * ln_w26).exp()); // base2^inv2
    let factor2 = r(p28 - 1.0);
    let b2 = r(r(r(bv * factor2) * ex34) + 1.0);
    let ln_b2 = r(b2.ln());
    let r2 = r(r(decay2 * ln_b2).exp()); // b2^decay2
    // mixture weights (t-independent; weight2 D-modulated):
    let p31 = r(r((-w[29]) * ln_sf).exp()); // sf^(-w29)
    let weight1 = r(w[27] * p31);
    let s32 = r(r(w[30] * ln_s).exp()); // s^w30
    let ex33 = r(r(r(d - 5.0) * r(w[31] - 0.5)).exp());
    let weight2 = r(r(w[28] * s32) * ex33);
    let wsum = r(weight1 + weight2);
    let num = r(r(weight1 * r1) + r(weight2 * r2));
    let ret = r(num / wsum);
    let out = r(r(ret * (1.0 - 2e-5)) + 1e-5);
    CurveCache {
        out, a, bv, decay1, factor1, b1, r1, q1, e1, m1, p35, decay2, factor2, b2, r2, inv2,
        p28, m2, ex34, weight1, weight2, wsum, ret, p31, s32, ex33, ln_sf, ln_b1, ln_b2, ln_s,
        ln_w25, ln_w26,
    }
}

/// VJP of the curve. `t` is data (no grad). Returns adjoints (g_s, g_sf, g_d); accumulates gw.
/// `g_r1_extra` is the adjoint flowing into r1 from the short-trace stability (which reads r1
/// rather than the mixed retention).
#[allow(clippy::too_many_arguments)]
fn curve_bwd(w: &[f64], c: &CurveCache, t: f64, s: f64, sf: f64, d: f64, g_out: f64, g_r1_extra: f64, gw: &mut [f64]) -> (f64, f64, f64) {
    let t = t.max(0.0);
    let g_ret = r(g_out * (1.0 - 2e-5));
    let g_num = r(g_ret / c.wsum);
    let g_wsum = r(r(-g_ret * c.ret) / c.wsum);
    let g_weight1 = r(r(g_num * c.r1) + g_wsum);
    let g_weight2 = r(r(g_num * c.r2) + g_wsum);
    let g_r1 = r(r(g_num * c.weight1) + g_r1_extra);
    let g_r2 = r(g_num * c.weight2);
    // weight2 = w28 * s32 * ex33
    gw[28] = r(gw[28] + r(r(g_weight2 * c.s32) * c.ex33));
    let g_s32 = r(r(g_weight2 * w[28]) * c.ex33);
    let g_ex33 = r(r(g_weight2 * w[28]) * c.s32);
    let mut g_d = r(r(g_ex33 * c.ex33) * r(w[31] - 0.5)); // ex33 = exp((d-5)*(w31-0.5))
    gw[31] = r(gw[31] + r(r(g_ex33 * c.ex33) * r(d - 5.0)));
    let mut g_s = r(r(g_s32 * w[30]) * r(c.s32 / s)); // s32 = s^w30
    gw[30] = r(gw[30] + r(r(g_s32 * c.s32) * c.ln_s));
    // weight1 = w27 * p31 ; p31 = sf^(-w29)
    gw[27] = r(gw[27] + r(g_weight1 * c.p31));
    let g_p31 = r(g_weight1 * w[27]);
    let mut g_sf = r(r(g_p31 * (-w[29])) * r(c.p31 / sf));
    gw[29] = r(gw[29] + r(g_p31 * (-r(c.p31 * c.ln_sf))));
    // r2 = b2^decay2 ; b2 = bv*factor2*ex34 + 1
    let g_b2 = r(r(g_r2 * c.decay2) * r(c.r2 / c.b2));
    let mut g_decay2 = r(r(g_r2 * c.r2) * c.ln_b2);
    let g_bv = r(r(g_b2 * c.factor2) * c.ex34);
    let g_factor2 = r(r(g_b2 * c.bv) * c.ex34);
    let g_ex34 = r(r(g_b2 * c.bv) * c.factor2);
    g_d = r(g_d + r(r(g_ex34 * c.ex34) * r(w[32] - 0.3))); // ex34 = exp((d-5)*(w32-0.3))
    gw[32] = r(gw[32] + r(r(g_ex34 * c.ex34) * r(d - 5.0)));
    let g_p28 = g_factor2; // factor2 = p28 - 1
    gw[26] = r(gw[26] + r(r(g_p28 * c.inv2) * r(c.p28 / w[26]))); // p28 = base2^inv2
    let g_inv2 = r(r(g_p28 * c.p28) * c.ln_w26);
    g_decay2 = r(g_decay2 + r(g_inv2 * r(-1.0 / r(c.decay2 * c.decay2)))); // inv2 = 1/decay2
    let g_dm2 = -g_decay2;
    let g_m2 = if c.m2 > 0.01 && c.m2 < 0.95 { g_dm2 } else { 0.0 };
    gw[24] = r(gw[24] + g_m2); // m2 = w24 directly
    g_s = r(g_s + r(g_bv * r(-t / r(s * s)))); // bv = t/s
    // r1 = b1^decay1 ; b1 = a*factor1 + 1
    let g_b1 = r(r(g_r1 * c.decay1) * r(c.r1 / c.b1));
    let mut g_decay1 = r(r(g_r1 * c.r1) * c.ln_b1);
    let g_a = r(g_b1 * c.factor1);
    let g_factor1 = r(g_b1 * c.a);
    let g_e1 = g_factor1; // factor1 = e1 - 1
    let g_q1c = r(g_e1 * c.e1); // e1 = exp(min(q1,60))
    let g_q1 = if c.q1 < 60.0 { g_q1c } else { 0.0 };
    // q1 = ln(base1) / decay1
    let lw25 = c.ln_w25;
    let g_lw25 = r(g_q1 / c.decay1);
    g_decay1 = r(g_decay1 + r(g_q1 * r(-lw25 / r(c.decay1 * c.decay1))));
    gw[25] = r(gw[25] + r(g_lw25 / w[25])); // ln(base1) = ln(w25)
    let g_dm1 = -g_decay1;
    let g_m1 = if c.m1 > 0.01 && c.m1 < 0.95 { g_dm1 } else { 0.0 };
    gw[23] = r(gw[23] + r(g_m1 * c.p35)); // m1 = w23 * p35
    let g_p35 = r(g_m1 * w[23]);
    g_sf = r(g_sf + r(r(g_p35 * r(w[33] - 0.3)) * r(c.p35 / sf))); // p35 = sf^(w33-0.3)
    gw[33] = r(gw[33] + r(r(g_p35 * c.p35) * c.ln_sf));
    g_sf = r(g_sf + r(g_a * r(-t / r(sf * sf)))); // a = t/sf
    (g_s, g_sf, g_d)
}

// ===================== stability after review =====================

struct StabCache {
    out: f64,
    nsf_fail: f64,
    pls: f64,
    sinc: f64,
    ls_sinc: f64,
    aa: f64,
    bb: f64,
    cc: f64,
    expr: f64,
    qbase: f64,
    rexp: f64,
    hard: f64,
    easy: f64,
    ln_ls: f64,
    ln_ls1: f64,
}

#[allow(clippy::too_many_arguments)]
fn stab_fwd(w: &[f64], last_s: f64, last_d: f64, r_in: f64, rating: f64, start: usize, aa: f64, ln_ls: f64) -> StabCache {
    let hard = if rating == 2.0 { w[start + 6] } else { 1.0 };
    let easy = if rating == 4.0 { w[start + 7] } else { 1.0 };
    let ln_ls1 = r((last_s + 1.0).ln());
    let qbase = r(r(w[start + 4] * ln_ls1).exp()); // (last_s+1)^w[start+4]
    let rexp = r(r(r(1.0 - r_in) * w[start + 5]).exp()); // exp((1-r)*w[start+5])
    let nsf_fail = r(r(w[start + 3] * r(qbase - 1.0)) * rexp);
    let pls = last_s.min(nsf_fail);
    let bb = r(11.0 - last_d);
    let cc = r(r((-w[start + 1]) * ln_ls).exp()); // last_s^(-w[start+1])
    let expr = r(r(r(1.0 - r_in) * w[start + 2]).exp());
    let sinc = r(r(r(r(r(r(aa * bb) * cc) * r(expr - 1.0)) * hard) * easy) + 1.0);
    let ls_sinc = r(last_s * sinc);
    let nss = pls.max(ls_sinc);
    let out = if rating > 1.0 { nss } else { pls };
    StabCache { out, nsf_fail, pls, sinc, ls_sinc, aa, bb, cc, expr, qbase, rexp, hard, easy, ln_ls, ln_ls1 }
}

/// VJP of stability. Returns (g_last_s, g_last_d, g_r); accumulates gw[start..start+8].
#[allow(clippy::too_many_arguments)]
fn stab_bwd(w: &[f64], c: &StabCache, last_s: f64, r_in: f64, rating: f64, start: usize, g_out: f64, gw: &mut [f64]) -> (f64, f64, f64) {
    let (g_nss, g_pls_direct) = if rating > 1.0 { (g_out, 0.0) } else { (0.0, g_out) };
    let g_pls_from_nss = if c.pls >= c.ls_sinc { g_nss } else { 0.0 };
    let g_ls_sinc = if c.ls_sinc > c.pls { g_nss } else { 0.0 };
    let mut g_last_s = r(g_ls_sinc * c.sinc);
    let g_sinc = r(g_ls_sinc * last_s);
    let g_pls = r(g_pls_direct + g_pls_from_nss);
    g_last_s = r(g_last_s + if last_s <= c.nsf_fail { g_pls } else { 0.0 });
    let g_nsf_fail = if c.nsf_fail < last_s { g_pls } else { 0.0 };
    // sinc = aa*bb*cc*(expr-1)*hard*easy + 1
    let em1 = r(c.expr - 1.0);
    let g_prod = g_sinc;
    let prod = r(r(r(r(r(c.aa * c.bb) * c.cc) * em1) * c.hard) * c.easy);
    gw[start] = r(gw[start] + r(g_prod * prod)); // aa = exp(w[start]-1.5)
    let g_bb = r(g_prod * r(r(r(r(c.aa * c.cc) * em1) * c.hard) * c.easy));
    let g_cc = r(g_prod * r(r(r(r(c.aa * c.bb) * em1) * c.hard) * c.easy));
    let g_em1 = r(g_prod * r(r(r(r(c.aa * c.bb) * c.cc) * c.hard) * c.easy));
    if rating == 2.0 {
        gw[start + 6] = r(gw[start + 6] + r(g_prod * r(r(r(r(c.aa * c.bb) * c.cc) * em1) * c.easy)));
    }
    if rating == 4.0 {
        gw[start + 7] = r(gw[start + 7] + r(g_prod * r(r(r(r(c.aa * c.bb) * c.cc) * em1) * c.hard)));
    }
    let g_last_d = -g_bb; // bb = 11 - last_d
    g_last_s = r(g_last_s + r(r(g_cc * (-w[start + 1])) * r(c.cc / last_s)));
    gw[start + 1] = r(gw[start + 1] + r(g_cc * (-r(c.cc * c.ln_ls))));
    // expr = exp((1-r)*w[start+2])
    let mut g_r = r(r(g_em1 * c.expr) * (-w[start + 2]));
    gw[start + 2] = r(gw[start + 2] + r(r(g_em1 * c.expr) * r(1.0 - r_in)));
    // nsf_fail = w[start+3] * (qbase-1) * rexp
    let q = r(c.qbase - 1.0);
    gw[start + 3] = r(gw[start + 3] + r(g_nsf_fail * r(q * c.rexp)));
    let g_q = r(r(g_nsf_fail * w[start + 3]) * c.rexp);
    let g_rexp = r(r(g_nsf_fail * w[start + 3]) * q);
    g_last_s = r(g_last_s + r(r(g_q * w[start + 4]) * r(c.qbase / (last_s + 1.0)))); // qbase = (last_s+1)^w[start+4]
    gw[start + 4] = r(gw[start + 4] + r(r(g_q * c.qbase) * c.ln_ls1));
    g_r = r(g_r + r(r(g_rexp * c.rexp) * (-w[start + 5]))); // rexp = exp((1-r)*w[start+5])
    gw[start + 5] = r(gw[start + 5] + r(r(g_rexp * c.rexp) * r(1.0 - r_in)));
    (g_last_s, g_last_d, g_r)
}

// ===================== next difficulty =====================

fn next_d_fwd(w: &[f64], last_d: f64, rating: f64, r_in: f64, init: f64) -> (f64, f64, f64) {
    let delta_d_base = r(-w[6] * r(rating - 3.0));
    let delta_d = if rating == 1.0 { r(delta_d_base * r(r_in + 0.1)) } else { delta_d_base };
    let new_d = r(last_d + r(r(r(10.0 - last_d) * delta_d) / 9.0));
    let out_pre = r(r(0.01 * init) + r(0.99 * new_d));
    (clamp(out_pre, D_MIN, D_MAX), out_pre, delta_d)
}

/// VJP of next_difficulty. Returns (g_last_d, g_r); accumulates gw[4], gw[5], gw[6].
#[allow(clippy::too_many_arguments)]
fn next_d_bwd(w: &[f64], out_pre: f64, delta_d: f64, last_d: f64, rating: f64, r_in: f64, g_out: f64, gw: &mut [f64], exp3w5: f64) -> (f64, f64) {
    let g_out_pre = if out_pre > D_MIN && out_pre < D_MAX { g_out } else { 0.0 };
    let g_init = r(g_out_pre * 0.01);
    let g_new_d = r(g_out_pre * 0.99);
    gw[4] = r(gw[4] + g_init);
    gw[5] = r(gw[5] + r(g_init * r(-exp3w5 * 3.0))); // init = w4 - exp(3 w5) + 1
    let g_last_d = r(g_new_d * r(1.0 - r(delta_d / 9.0)));
    let g_delta_d = r(r(g_new_d * r(10.0 - last_d)) / 9.0);
    let rm3 = r(rating - 3.0);
    let mut g_r = 0.0;
    if rating == 1.0 {
        gw[6] = r(gw[6] + r(r(g_delta_d * (-rm3)) * r(r_in + 0.1)));
        g_r = r(g_delta_d * r(-w[6] * rm3));
    } else {
        gw[6] = r(gw[6] + r(g_delta_d * (-rm3)));
    }
    (g_last_d, g_r)
}

// ===================== one recurrence step =====================

/// Per-step forward intermediates the backward pass replays. Public (so callers can hold a
/// reusable `Vec<StepCache>` scratch buffer) but with private fields.
pub struct StepCache {
    s0: f64,
    d0: f64,
    sf0: f64,
    last_s: f64,
    last_d: f64,
    last_sf: f64,
    dt: f64,
    rating: f64,
    nth0: bool,
    curve: CurveCache,
    slow: StabCache,
    fast: StabCache,
    nd_out_pre: f64,
    nd_delta_d: f64,
    ns3: f64,
    nsf3: f64,
}

fn step_fwd(w: &[f64], delta_t: f64, rating: f64, state: (f64, f64, f64), nth0: bool, wc: &WConsts) -> ((f64, f64, f64), StepCache) {
    let (s0, d0, sf0) = state;
    let last_s = clamp(s0, wc.s_min, S_MAX);
    let last_d = clamp(d0, D_MIN, D_MAX);
    let last_sf = clamp(sf0, wc.s_min, S_MAX);
    let dt = delta_t.max(0.0);
    let ln_last_s = r(last_s.ln());
    let ln_last_sf = r(last_sf.ln());
    let curve = curve_fwd(w, dt, last_s, last_sf, last_d, wc.ln_w25, wc.ln_w26, ln_last_s, ln_last_sf);
    let rec = curve.out;
    let r1 = curve.r1; // short component recall — drives the short-trace stability
    let slow = stab_fwd(w, last_s, last_d, rec, rating, 7, wc.aa7, ln_last_s);
    let fast = stab_fwd(w, last_sf, last_d, r1, rating, 15, wc.aa16, ln_last_sf);
    let (nd1, nd_out_pre, nd_delta_d) = next_d_fwd(w, last_d, rating, rec, wc.init);
    // post-lapse short reset: on a lapse cap s_short at 0.8 * post-lapse s_long.
    let nsf_pre = if rating == 1.0 { fast.out.min(r(0.8 * slow.out)) } else { fast.out };
    let (mut ns, mut nsf, mut nd) = (slow.out, nsf_pre, nd1);
    if nth0 && s0 == 0.0 {
        let rc = clamp(rating, 1.0, 4.0);
        let init_s = w[(rc as usize) - 1];
        let init_d = clamp(r(r(w[4] - r(r(w[5] * r(rc - 1.0)).exp())) + 1.0), D_MIN, D_MAX);
        ns = init_s;
        nsf = r(0.8 * init_s);
        nd = init_d;
    }
    let ns3 = ns;
    let nsf3 = nsf;
    let out = (clamp(ns, wc.s_min, S_MAX), nd, clamp(nsf, wc.s_min, S_MAX));
    let cache = StepCache {
        s0, d0, sf0, last_s, last_d, last_sf, dt, rating, nth0, curve, slow, fast, nd_out_pre, nd_delta_d, ns3, nsf3,
    };
    (out, cache)
}

/// VJP of one step. Given adjoints on the OUTPUT state, returns adjoints on the INPUT state.
fn step_bwd(w: &[f64], c: &StepCache, g_out: (f64, f64, f64), gw: &mut [f64], wc: &WConsts) -> (f64, f64, f64) {
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    // final state clamps
    let g_ns3 = if c.ns3 > wc.s_min && c.ns3 < S_MAX { g_ns_out } else { 0.0 };
    let g_nsf3 = if c.nsf3 > wc.s_min && c.nsf3 < S_MAX { g_nsf_out } else { 0.0 };
    let g_nd3 = g_nd_out;
    // init override (first review)
    let (g_ns1, g_nsf1, g_nd1) = if c.nth0 && c.s0 == 0.0 {
        let rc = clamp(c.rating, 1.0, 4.0);
        gw[(rc as usize) - 1] = r(gw[(rc as usize) - 1] + r(g_ns3 + r(g_nsf3 * 0.8))); // init_s = w[rc-1]; nsf = 0.8*init_s
        let id_pre = r(r(w[4] - r(r(w[5] * r(rc - 1.0)).exp())) + 1.0);
        if id_pre > D_MIN && id_pre < D_MAX {
            gw[4] = r(gw[4] + g_nd3);
            gw[5] = r(gw[5] + r(g_nd3 * r(-r(r(w[5] * r(rc - 1.0)).exp()) * r(rc - 1.0))));
        }
        (0.0, 0.0, 0.0)
    } else {
        (g_ns3, g_nsf3, g_nd3)
    };
    // post-lapse min routing: nsf_pre = (rating==1)? min(fast.out, 0.8*slow.out) : fast.out.
    let (g_fast_out, g_slow_from_relearn) = if c.rating == 1.0 {
        if c.fast.out <= r(0.8 * c.slow.out) {
            (g_nsf1, 0.0)
        } else {
            (0.0, r(g_nsf1 * 0.8))
        }
    } else {
        (g_nsf1, 0.0)
    };
    // long stab reads mixed retention curve.out; short stab reads r1 = curve.r1.
    let (g_ls_a, g_ld_a, g_r_long) = stab_bwd(w, &c.slow, c.last_s, c.curve.out, c.rating, 7, r(g_ns1 + g_slow_from_relearn), gw);
    let (g_lsf_b, g_ld_b, g_r1_short) = stab_bwd(w, &c.fast, c.last_sf, c.curve.r1, c.rating, 15, g_fast_out, gw);
    let (g_ld_c, g_r_nextd) = next_d_bwd(w, c.nd_out_pre, c.nd_delta_d, c.last_d, c.rating, c.curve.out, g_nd1, gw, wc.exp3w5);
    // curve.out adjoint = long-stab r + next_d r; curve.r1 adjoint = short-stab r.
    let (g_ls_d, g_lsf_d, g_ld_d) = curve_bwd(w, &c.curve, c.dt, c.last_s, c.last_sf, c.last_d, r(g_r_long + g_r_nextd), g_r1_short, gw);
    let g_last_s = r(g_ls_a + g_ls_d);
    let g_last_sf = r(g_lsf_b + g_lsf_d);
    let g_last_d = r(r(r(g_ld_a + g_ld_b) + g_ld_c) + g_ld_d);
    // input state clamps
    let g_s0 = if c.s0 > wc.s_min && c.s0 < S_MAX { g_last_s } else { 0.0 };
    let g_d0 = if c.d0 > D_MIN && c.d0 < D_MAX { g_last_d } else { 0.0 };
    let g_sf0 = if c.sf0 > wc.s_min && c.sf0 < S_MAX { g_last_sf } else { 0.0 };
    (g_s0, g_d0, g_sf0)
}

// ===================== public entry points =====================

/// Run the dual-stability recurrence over `prior_dt`/`prior_r`, then predict at `cur_dt`.
/// Value-only (no gradient) — mirrors `fsrs_v7::retention` with `Dual<0>`.
pub fn predict_one(w: &[f64], prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, wc: &WConsts) -> f64 {
    let mut state = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..prior_r.len() {
        let dt = prior_dt[k];
        state = step_fwd(w, dt, prior_r[k] as f64, state, k == 0, wc).0;
    }
    let (s, d, sf) = state;
    curve_fwd(w, cur_dt, s, sf, d, wc.ln_w25, wc.ln_w26, r(s.ln()), r(sf.ln())).out
}

/// Windowed O(C) prediction: replay a card's review sequence ONCE, emitting a prediction at each
/// requested position. Bit-identical to calling [`predict_one`] per position (same `step_fwd`
/// recurrence + the same separate final `curve_fwd`), but O(C) per card instead of O(C²) over the
/// card's rows. `positions` MUST be sorted ascending; `cur_dts[j]` is `positions[j]`'s row
/// `delta_t`; the prediction for `positions[j]` is written to `out[j]`.
pub fn predict_card(w: &[f64], card_dt: &[f64], card_r: &[i64], positions: &[usize], cur_dts: &[f64], wc: &WConsts, out: &mut [f64]) {
    if positions.is_empty() {
        return;
    }
    let max_pos = positions[positions.len() - 1];
    let mut state = (0.0f64, 0.0f64, 0.0f64); // (s_long, d, s_short) — matches step_fwd output order
    let mut req = 0;
    for t in 0..=max_pos {
        // Emit predictions whose position is the current (pre-step) state = state-after-(0..t-1).
        while req < positions.len() && positions[req] == t {
            let (s, d, sf) = state;
            out[req] = curve_fwd(w, cur_dts[req], s, sf, d, wc.ln_w25, wc.ln_w26, r(s.ln()), r(sf.ln())).out;
            req += 1;
        }
        if t < max_pos {
            state = step_fwd(w, card_dt[t], card_r[t] as f64, state, t == 0, wc).0;
        }
    }
}

/// Forward + reverse-mode backward for ONE row; accumulates d(weight·BCE)/dw into `gw` and
/// returns the prediction `p`. `caches` is a scratch buffer (cleared internally) reused across
/// rows to avoid per-row allocation.
#[allow(clippy::too_many_arguments)]
pub fn grad_one(w: &[f64], prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, y: f64, weight: f64, wc: &WConsts, gw: &mut [f64], caches: &mut Vec<StepCacheBox>) -> f64 {
    caches.clear();
    let mut state = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..prior_r.len() {
        let dt = prior_dt[k];
        let (ns, cache) = step_fwd(w, dt, prior_r[k] as f64, state, k == 0, wc);
        state = ns;
        caches.push(cache);
    }
    let (s, d, sf) = state;
    let fc = curve_fwd(w, cur_dt, s, sf, d, wc.ln_w25, wc.ln_w26, r(s.ln()), r(sf.ln()));
    let r_raw = fc.out;
    let rec = clamp(r_raw, MIN_R, MAX_R);
    // d(weight·BCE)/dr = weight·(r-y)/(r(1-r))
    let g_r = r(r(weight * r(rec - y)) / r(rec * r(1.0 - rec)));
    let g_rraw = if r_raw > MIN_R && r_raw < MAX_R { g_r } else { 0.0 };
    let (mut g_s, mut g_sf, mut g_d) = curve_bwd(w, &fc, cur_dt, s, sf, d, g_rraw, 0.0, gw);
    for k in (0..caches.len()).rev() {
        let (gs0, gd0, gsf0) = step_bwd(w, &caches[k], (g_s, g_d, g_sf), gw, wc);
        g_s = gs0;
        g_d = gd0;
        g_sf = gsf0;
    }
    r_raw
}

/// Public alias so callers (and the scratch buffer's type) don't need the private `StepCache`.
pub type StepCacheBox = StepCache;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Dual;
    use crate::models::fsrs_v7::{self, INIT_W, NP};

    // Forward-mode oracle: value + gradient of fsrs_v7::retention via Dual<NP>.
    fn forward_mode(prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, w: &[f64; NP], s_min: f64) -> (f64, [f64; NP]) {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w[k]));
        let ret = fsrs_v7::retention_dual(prior_dt, prior_r, cur_dt, &wd, s_min);
        (ret.v, ret.g)
    }

    #[test]
    fn analytic_predict_matches_forward_mode() {
        let prior_dt: Vec<f64> = vec![0.0, 0.3, 9.0, 1.5, 30.0, 0.02, 100.0, 5.0];
        let prior_r: Vec<i64> = vec![3, 1, 3, 4, 2, 1, 3, 2];
        let (cur_dt, s_min) = (7.0, 0.0001);
        let wc = wconsts(&INIT_W, s_min);
        let p_analytic = predict_one(&INIT_W, &prior_dt, &prior_r, cur_dt, &wc);
        let (p_fwd, _) = forward_mode(&prior_dt, &prior_r, cur_dt, &INIT_W, s_min);
        // f64-tight when `fp64`; otherwise both are f32 (different op orders) ⇒ ~f32 ulp.
        let tol = if cfg!(feature = "fp64") { 1e-9 } else { 1e-5 };
        assert!((p_analytic - p_fwd).abs() < tol, "predict: analytic {p_analytic} vs fwd {p_fwd}");
    }

    #[test]
    fn analytic_grad_matches_forward_mode() {
        // A few synthetic sequences + perturbed params; check grad of weight·BCE at y=1.
        let seqs: Vec<(Vec<f64>, Vec<i64>, f64)> = vec![
            (vec![0.0, 0.3, 9.0, 1.5, 30.0, 0.02, 100.0], vec![3, 1, 3, 4, 2, 1, 3], 7.0),
            (vec![0.0, 5.0, 2.0], vec![2, 3, 1], 12.0),
            (vec![0.0, 1.0, 1.0, 1.0, 50.0, 0.5], vec![1, 1, 2, 3, 4, 3], 3.0),
        ];
        let s_min = 0.0001;
        // Perturb params a bit so we're not at the exact default (exercises more branches).
        let mut w = INIT_W;
        for (k, wk) in w.iter_mut().enumerate() {
            *wk += 0.01 * (k as f64).sin();
        }
        for (prior_dt, prior_r, cur_dt) in &seqs {
            let wc = wconsts(&w, s_min);
            let y = 1.0;
            let weight = 0.7;
            let mut gw = vec![0.0f64; NP];
            let mut caches = Vec::new();
            let p = grad_one(&w, prior_dt, prior_r, *cur_dt, y, weight, &wc, &mut gw, &mut caches);
            let (p_fwd, g_fwd) = forward_mode(prior_dt, prior_r, *cur_dt, &w, s_min);
            // f64-tight when `fp64`; otherwise both paths are f32 (different op orders / Cephes
            // vs libm) ⇒ agreement only at ~f32 ulp accumulated over the recurrence.
            let (p_tol, g_rel) = if cfg!(feature = "fp64") { (1e-9, 1e-6) } else { (1e-5, 5e-3) };
            assert!((p - p_fwd).abs() < p_tol, "p {p} vs {p_fwd}");
            let dl = weight * (p_fwd - y) / (p_fwd * (1.0 - p_fwd));
            for k in 0..NP {
                let expected = dl * g_fwd[k];
                assert!(
                    (gw[k] - expected).abs() < g_rel + g_rel * expected.abs(),
                    "param {k}: analytic {} vs fwd {}",
                    gw[k],
                    expected
                );
            }
        }
    }
}
