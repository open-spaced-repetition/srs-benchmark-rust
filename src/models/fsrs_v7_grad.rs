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
        ln_w25: w[25].ln(),
        ln_w26: w[26].ln(),
        aa7: (w[7] - 1.5).exp(),
        aa16: (w[15] - 1.5).exp(),
        init: w[4] - (w[5] * 3.0).exp() + 1.0,
        exp3w5: (w[5] * 3.0).exp(),
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
    let a = t / sf;
    let bv = t / s;
    // short-trace recall r1 (driven by s_short=sf):
    let p35 = ((w[33] - 0.3) * ln_sf).exp(); // sf^(w33-0.3)
    let m1 = w[23] * p35;
    let dm1 = clamp(m1, 0.01, 0.95);
    let decay1 = -dm1;
    let q1 = ln_w25 / decay1;
    let e1 = q1.min(60.0).exp();
    let factor1 = e1 - 1.0;
    let b1 = a * factor1 + 1.0;
    let ln_b1 = b1.ln();
    let r1 = (decay1 * ln_b1).exp(); // b1^decay1
    // long-trace recall r2 (D scales the horizontal time-scale; decay2 not d-modulated):
    let ex34 = ((d - 5.0) * (w[32] - 0.3)).exp();
    let m2 = w[24];
    let dm2 = clamp(m2, 0.01, 0.95);
    let decay2 = -dm2;
    let inv2 = 1.0 / decay2;
    let p28 = (inv2 * ln_w26).exp(); // base2^inv2
    let factor2 = p28 - 1.0;
    let b2 = bv * factor2 * ex34 + 1.0;
    let ln_b2 = b2.ln();
    let r2 = (decay2 * ln_b2).exp(); // b2^decay2
    // mixture weights (t-independent; weight2 D-modulated):
    let p31 = ((-w[29]) * ln_sf).exp(); // sf^(-w29)
    let weight1 = w[27] * p31;
    let s32 = (w[30] * ln_s).exp(); // s^w30
    let ex33 = ((d - 5.0) * (w[31] - 0.5)).exp();
    let weight2 = w[28] * s32 * ex33;
    let wsum = weight1 + weight2;
    let num = weight1 * r1 + weight2 * r2;
    let ret = num / wsum;
    let out = ret * (1.0 - 2e-5) + 1e-5;
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
    let g_ret = g_out * (1.0 - 2e-5);
    let g_num = g_ret / c.wsum;
    let g_wsum = -g_ret * c.ret / c.wsum;
    let g_weight1 = g_num * c.r1 + g_wsum;
    let g_weight2 = g_num * c.r2 + g_wsum;
    let g_r1 = g_num * c.weight1 + g_r1_extra;
    let g_r2 = g_num * c.weight2;
    // weight2 = w28 * s32 * ex33
    gw[28] += g_weight2 * c.s32 * c.ex33;
    let g_s32 = g_weight2 * w[28] * c.ex33;
    let g_ex33 = g_weight2 * w[28] * c.s32;
    let mut g_d = g_ex33 * c.ex33 * (w[31] - 0.5); // ex33 = exp((d-5)*(w31-0.5))
    gw[31] += g_ex33 * c.ex33 * (d - 5.0);
    let mut g_s = g_s32 * w[30] * (c.s32 / s); // s32 = s^w30
    gw[30] += g_s32 * c.s32 * c.ln_s;
    // weight1 = w27 * p31 ; p31 = sf^(-w29)
    gw[27] += g_weight1 * c.p31;
    let g_p31 = g_weight1 * w[27];
    let mut g_sf = g_p31 * (-w[29]) * (c.p31 / sf);
    gw[29] += g_p31 * (-(c.p31 * c.ln_sf));
    // r2 = b2^decay2 ; b2 = bv*factor2*ex34 + 1
    let g_b2 = g_r2 * c.decay2 * (c.r2 / c.b2);
    let mut g_decay2 = g_r2 * c.r2 * c.ln_b2;
    let g_bv = g_b2 * c.factor2 * c.ex34;
    let g_factor2 = g_b2 * c.bv * c.ex34;
    let g_ex34 = g_b2 * c.bv * c.factor2;
    g_d += g_ex34 * c.ex34 * (w[32] - 0.3); // ex34 = exp((d-5)*(w32-0.3))
    gw[32] += g_ex34 * c.ex34 * (d - 5.0);
    let g_p28 = g_factor2; // factor2 = p28 - 1
    gw[26] += g_p28 * c.inv2 * (c.p28 / w[26]); // p28 = base2^inv2
    let g_inv2 = g_p28 * c.p28 * c.ln_w26;
    g_decay2 += g_inv2 * (-1.0 / (c.decay2 * c.decay2)); // inv2 = 1/decay2
    let g_dm2 = -g_decay2;
    let g_m2 = if c.m2 > 0.01 && c.m2 < 0.95 { g_dm2 } else { 0.0 };
    gw[24] += g_m2; // m2 = w24 directly
    g_s += g_bv * (-t / (s * s)); // bv = t/s
    // r1 = b1^decay1 ; b1 = a*factor1 + 1
    let g_b1 = g_r1 * c.decay1 * (c.r1 / c.b1);
    let mut g_decay1 = g_r1 * c.r1 * c.ln_b1;
    let g_a = g_b1 * c.factor1;
    let g_factor1 = g_b1 * c.a;
    let g_e1 = g_factor1; // factor1 = e1 - 1
    let g_q1c = g_e1 * c.e1; // e1 = exp(min(q1,60))
    let g_q1 = if c.q1 < 60.0 { g_q1c } else { 0.0 };
    // q1 = ln(base1) / decay1
    let lw25 = c.ln_w25;
    let g_lw25 = g_q1 / c.decay1;
    g_decay1 += g_q1 * (-lw25 / (c.decay1 * c.decay1));
    gw[25] += g_lw25 / w[25]; // ln(base1) = ln(w25)
    let g_dm1 = -g_decay1;
    let g_m1 = if c.m1 > 0.01 && c.m1 < 0.95 { g_dm1 } else { 0.0 };
    gw[23] += g_m1 * c.p35; // m1 = w23 * p35
    let g_p35 = g_m1 * w[23];
    g_sf += g_p35 * (w[33] - 0.3) * (c.p35 / sf); // p35 = sf^(w33-0.3)
    gw[33] += g_p35 * c.p35 * c.ln_sf;
    g_sf += g_a * (-t / (sf * sf)); // a = t/sf
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
fn stab_fwd(w: &[f64], last_s: f64, last_d: f64, r: f64, rating: f64, start: usize, aa: f64, ln_ls: f64) -> StabCache {
    let hard = if rating == 2.0 { w[start + 6] } else { 1.0 };
    let easy = if rating == 4.0 { w[start + 7] } else { 1.0 };
    let ln_ls1 = (last_s + 1.0).ln();
    let qbase = (w[start + 4] * ln_ls1).exp(); // (last_s+1)^w[start+4]
    let rexp = ((1.0 - r) * w[start + 5]).exp(); // exp((1-r)*w[start+5])
    let nsf_fail = w[start + 3] * (qbase - 1.0) * rexp;
    let pls = last_s.min(nsf_fail);
    let bb = 11.0 - last_d;
    let cc = ((-w[start + 1]) * ln_ls).exp(); // last_s^(-w[start+1])
    let expr = ((1.0 - r) * w[start + 2]).exp();
    let sinc = aa * bb * cc * (expr - 1.0) * hard * easy + 1.0;
    let ls_sinc = last_s * sinc;
    let nss = pls.max(ls_sinc);
    let out = if rating > 1.0 { nss } else { pls };
    StabCache { out, nsf_fail, pls, sinc, ls_sinc, aa, bb, cc, expr, qbase, rexp, hard, easy, ln_ls, ln_ls1 }
}

/// VJP of stability. Returns (g_last_s, g_last_d, g_r); accumulates gw[start..start+8].
#[allow(clippy::too_many_arguments)]
fn stab_bwd(w: &[f64], c: &StabCache, last_s: f64, r: f64, rating: f64, start: usize, g_out: f64, gw: &mut [f64]) -> (f64, f64, f64) {
    let (g_nss, g_pls_direct) = if rating > 1.0 { (g_out, 0.0) } else { (0.0, g_out) };
    let g_pls_from_nss = if c.pls >= c.ls_sinc { g_nss } else { 0.0 };
    let g_ls_sinc = if c.ls_sinc > c.pls { g_nss } else { 0.0 };
    let mut g_last_s = g_ls_sinc * c.sinc;
    let g_sinc = g_ls_sinc * last_s;
    let g_pls = g_pls_direct + g_pls_from_nss;
    g_last_s += if last_s <= c.nsf_fail { g_pls } else { 0.0 };
    let g_nsf_fail = if c.nsf_fail < last_s { g_pls } else { 0.0 };
    // sinc = aa*bb*cc*(expr-1)*hard*easy + 1
    let em1 = c.expr - 1.0;
    let g_prod = g_sinc;
    let prod = c.aa * c.bb * c.cc * em1 * c.hard * c.easy;
    gw[start] += g_prod * prod; // aa = exp(w[start]-1.5)
    let g_bb = g_prod * (c.aa * c.cc * em1 * c.hard * c.easy);
    let g_cc = g_prod * (c.aa * c.bb * em1 * c.hard * c.easy);
    let g_em1 = g_prod * (c.aa * c.bb * c.cc * c.hard * c.easy);
    if rating == 2.0 {
        gw[start + 6] += g_prod * (c.aa * c.bb * c.cc * em1 * c.easy);
    }
    if rating == 4.0 {
        gw[start + 7] += g_prod * (c.aa * c.bb * c.cc * em1 * c.hard);
    }
    let g_last_d = g_bb * (-1.0); // bb = 11 - last_d
    g_last_s += g_cc * (-w[start + 1]) * (c.cc / last_s);
    gw[start + 1] += g_cc * (-(c.cc * c.ln_ls));
    // expr = exp((1-r)*w[start+2])
    let mut g_r = g_em1 * c.expr * (-w[start + 2]);
    gw[start + 2] += g_em1 * c.expr * (1.0 - r);
    // nsf_fail = w[start+3] * (qbase-1) * rexp
    let q = c.qbase - 1.0;
    gw[start + 3] += g_nsf_fail * (q * c.rexp);
    let g_q = g_nsf_fail * w[start + 3] * c.rexp;
    let g_rexp = g_nsf_fail * w[start + 3] * q;
    g_last_s += g_q * w[start + 4] * (c.qbase / (last_s + 1.0)); // qbase = (last_s+1)^w[start+4]
    gw[start + 4] += g_q * c.qbase * c.ln_ls1;
    g_r += g_rexp * c.rexp * (-w[start + 5]); // rexp = exp((1-r)*w[start+5])
    gw[start + 5] += g_rexp * c.rexp * (1.0 - r);
    (g_last_s, g_last_d, g_r)
}

// ===================== next difficulty =====================

fn next_d_fwd(w: &[f64], last_d: f64, rating: f64, r: f64, init: f64) -> (f64, f64, f64) {
    let delta_d_base = -w[6] * (rating - 3.0);
    let delta_d = if rating == 1.0 { delta_d_base * (r + 0.1) } else { delta_d_base };
    let new_d = last_d + (10.0 - last_d) * delta_d / 9.0;
    let out_pre = 0.01 * init + 0.99 * new_d;
    (clamp(out_pre, D_MIN, D_MAX), out_pre, delta_d)
}

/// VJP of next_difficulty. Returns (g_last_d, g_r); accumulates gw[4], gw[5], gw[6].
#[allow(clippy::too_many_arguments)]
fn next_d_bwd(w: &[f64], out_pre: f64, delta_d: f64, last_d: f64, rating: f64, r: f64, g_out: f64, gw: &mut [f64], exp3w5: f64) -> (f64, f64) {
    let g_out_pre = if out_pre > D_MIN && out_pre < D_MAX { g_out } else { 0.0 };
    let g_init = g_out_pre * 0.01;
    let g_new_d = g_out_pre * 0.99;
    gw[4] += g_init;
    gw[5] += g_init * (-exp3w5 * 3.0); // init = w4 - exp(3 w5) + 1
    let g_last_d = g_new_d * (1.0 - delta_d / 9.0);
    let g_delta_d = g_new_d * (10.0 - last_d) / 9.0;
    let rm3 = rating - 3.0;
    let mut g_r = 0.0;
    if rating == 1.0 {
        gw[6] += g_delta_d * (-rm3) * (r + 0.1);
        g_r = g_delta_d * (-w[6] * rm3);
    } else {
        gw[6] += g_delta_d * (-rm3);
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
    let ln_last_s = last_s.ln();
    let ln_last_sf = last_sf.ln();
    let curve = curve_fwd(w, dt, last_s, last_sf, last_d, wc.ln_w25, wc.ln_w26, ln_last_s, ln_last_sf);
    let r = curve.out;
    let r1 = curve.r1; // short component recall — drives the short-trace stability
    let slow = stab_fwd(w, last_s, last_d, r, rating, 7, wc.aa7, ln_last_s);
    let fast = stab_fwd(w, last_sf, last_d, r1, rating, 15, wc.aa16, ln_last_sf);
    let (nd1, nd_out_pre, nd_delta_d) = next_d_fwd(w, last_d, rating, r, wc.init);
    // post-lapse short reset: on a lapse cap s_short at 0.8 * post-lapse s_long.
    let nsf_pre = if rating == 1.0 { fast.out.min(0.8 * slow.out) } else { fast.out };
    let (mut ns, mut nsf, mut nd) = (slow.out, nsf_pre, nd1);
    if nth0 && s0 == 0.0 {
        let rc = clamp(rating, 1.0, 4.0);
        let init_s = w[(rc as usize) - 1];
        let init_d = clamp(w[4] - (w[5] * (rc - 1.0)).exp() + 1.0, D_MIN, D_MAX);
        ns = init_s;
        nsf = 0.8 * init_s;
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
        gw[(rc as usize) - 1] += g_ns3 + g_nsf3 * 0.8; // init_s = w[rc-1]; nsf = 0.8*init_s
        let id_pre = w[4] - (w[5] * (rc - 1.0)).exp() + 1.0;
        if id_pre > D_MIN && id_pre < D_MAX {
            gw[4] += g_nd3;
            gw[5] += g_nd3 * (-((w[5] * (rc - 1.0)).exp()) * (rc - 1.0));
        }
        (0.0, 0.0, 0.0)
    } else {
        (g_ns3, g_nsf3, g_nd3)
    };
    // post-lapse min routing: nsf_pre = (rating==1)? min(fast.out, 0.8*slow.out) : fast.out.
    let (g_fast_out, g_slow_from_relearn) = if c.rating == 1.0 {
        if c.fast.out <= 0.8 * c.slow.out {
            (g_nsf1, 0.0)
        } else {
            (0.0, g_nsf1 * 0.8)
        }
    } else {
        (g_nsf1, 0.0)
    };
    // long stab reads mixed retention curve.out; short stab reads r1 = curve.r1.
    let (g_ls_a, g_ld_a, g_r_long) = stab_bwd(w, &c.slow, c.last_s, c.curve.out, c.rating, 7, g_ns1 + g_slow_from_relearn, gw);
    let (g_lsf_b, g_ld_b, g_r1_short) = stab_bwd(w, &c.fast, c.last_sf, c.curve.r1, c.rating, 15, g_fast_out, gw);
    let (g_ld_c, g_r_nextd) = next_d_bwd(w, c.nd_out_pre, c.nd_delta_d, c.last_d, c.rating, c.curve.out, g_nd1, gw, wc.exp3w5);
    // curve.out adjoint = long-stab r + next_d r; curve.r1 adjoint = short-stab r.
    let (g_ls_d, g_lsf_d, g_ld_d) = curve_bwd(w, &c.curve, c.dt, c.last_s, c.last_sf, c.last_d, g_r_long + g_r_nextd, g_r1_short, gw);
    let g_last_s = g_ls_a + g_ls_d;
    let g_last_sf = g_lsf_b + g_lsf_d;
    let g_last_d = g_ld_a + g_ld_b + g_ld_c + g_ld_d;
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
    curve_fwd(w, cur_dt, s, sf, d, wc.ln_w25, wc.ln_w26, s.ln(), sf.ln()).out
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
            out[req] = curve_fwd(w, cur_dts[req], s, sf, d, wc.ln_w25, wc.ln_w26, s.ln(), sf.ln()).out;
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
    let fc = curve_fwd(w, cur_dt, s, sf, d, wc.ln_w25, wc.ln_w26, s.ln(), sf.ln());
    let r_raw = fc.out;
    let r = clamp(r_raw, MIN_R, MAX_R);
    // d(weight·BCE)/dr = weight·(r-y)/(r(1-r))
    let g_r = weight * (r - y) / (r * (1.0 - r));
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
        assert!((p_analytic - p_fwd).abs() < 1e-9, "predict: analytic {p_analytic} vs fwd {p_fwd}");
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
            assert!((p - p_fwd).abs() < 1e-9, "p {p} vs {p_fwd}");
            let dl = weight * (p_fwd - y) / (p_fwd * (1.0 - p_fwd));
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
