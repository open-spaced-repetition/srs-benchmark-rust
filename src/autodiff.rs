//! Minimal forward-mode autodiff (dual numbers) for the FSRS recurrence.
//!
//! Each `Dual<P>` carries a value plus its gradient w.r.t. the `P` model parameters, as a
//! fixed-size array (no heap allocation in the hot path). The FSRS `step`/`forgetting_curve`
//! are written once over `Dual`s and we read the parameter gradient straight out — no
//! hand-derived per-version Jacobians. Op gradients match torch's conventions (clamp passes
//! through in range / zero outside; min/max/where select).
//!
//! Processing is per-row (scalar): `where`/`min`/`max` become ordinary `if`s.

/// Per-algo precision flag (set once in `run.rs` before processing). `true` ⇒ `round_scalar`
/// rounds to f32 (analytic / reverse-mode-gradient algos: HLR, DASH, LogReg, FSRS-7, …, matching
/// torch's f32). `false` ⇒ no-op f64 (forward-mode-`Dual` algos: FSRS v1–v6, v4.5, ACT-R, Anki,
/// DASH[ACT-R], SM2-trainable). Those models' forward-mode gradient is only a faithful proxy for
/// torch's f32 *reverse-mode* in f64 (it matched upstream in the all-f64 era); a *f32* forward-mode
/// gradient rounds differently and diverges badly on chaotic `--secs` training. Default `true`
/// (the common case + unit tests). Forced f64 (no-op) under the opt-in `fp64` feature.
pub(crate) static ROUND_F32: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Round to f32 when the current algo is f32-precision (see `ROUND_F32`); else a no-op (f64). For
/// IEEE elementary ops, computing in f64 then rounding to f32 yields exactly the f32 result, so
/// this faithfully emulates f32 for the analytic algos that need it to match the f32 references.
#[inline(always)]
pub(crate) fn round_scalar(x: f64) -> f64 {
    #[cfg(not(feature = "fp64"))]
    {
        if ROUND_F32.load(std::sync::atomic::Ordering::Relaxed) {
            x as f32 as f64
        } else {
            x
        }
    }
    #[cfg(feature = "fp64")]
    {
        x
    }
}

#[derive(Clone, Copy)]
pub struct Dual<const P: usize> {
    pub v: f64,
    pub g: [f64; P],
}

impl<const P: usize> Dual<P> {
    /// A constant (zero gradient).
    pub fn c(v: f64) -> Self {
        Dual { v, g: [0.0; P] }
    }
    /// The k-th parameter with value `v` (gradient = unit vector e_k).
    pub fn param(k: usize, v: f64) -> Self {
        let mut g = [0.0; P];
        g[k] = 1.0;
        Dual { v, g }
    }

    // Plain f64 throughout (no `round_scalar`). Forward-mode `Dual` is used in production ONLY by
    // the f64 algos (FSRS v1–v6, v4.5, ACT-R, Anki, DASH[ACT-R], SM2-trainable) — torch's f32
    // *reverse-mode* gradient is only proxied faithfully by a *f64* forward-mode pass (a f32
    // forward-mode gradient diverges badly on chaotic `--secs` training), so `ROUND_F32` is always
    // false here and `r()` was always a no-op. Dropping it is therefore bit-identical AND lets the
    // const-`P` gradient loops auto-vectorize (the per-element atomic-load+branch had blocked SIMD).
    pub fn add(self, o: Self) -> Self {
        let mut g = self.g;
        for k in 0..P {
            g[k] += o.g[k];
        }
        Dual { v: self.v + o.v, g }
    }
    pub fn sub(self, o: Self) -> Self {
        let mut g = self.g;
        for k in 0..P {
            g[k] -= o.g[k];
        }
        Dual { v: self.v - o.v, g }
    }
    pub fn mul(self, o: Self) -> Self {
        let mut g = [0.0; P];
        for k in 0..P {
            g[k] = self.v * o.g[k] + o.v * self.g[k];
        }
        Dual { v: self.v * o.v, g }
    }
    pub fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let v = self.v * inv;
        let mut g = [0.0; P];
        for k in 0..P {
            // d(a/b) = (a'·b - a·b') / b^2 = a'/b - v·b'/b
            g[k] = (self.g[k] - v * o.g[k]) * inv;
        }
        Dual { v, g }
    }

    pub fn add_c(self, c: f64) -> Self {
        Dual { v: self.v + c, g: self.g }
    }
    pub fn mul_c(self, c: f64) -> Self {
        let mut g = self.g;
        for k in 0..P {
            g[k] *= c;
        }
        Dual { v: self.v * c, g }
    }
    /// `c - self`.
    pub fn c_sub(self, c: f64) -> Self {
        let mut g = self.g;
        for k in 0..P {
            g[k] = -g[k];
        }
        Dual { v: c - self.v, g }
    }
    pub fn neg(self) -> Self {
        self.mul_c(-1.0)
    }

    pub fn exp(self) -> Self {
        let v = self.v.exp();
        let mut g = [0.0; P];
        for k in 0..P {
            g[k] = v * self.g[k];
        }
        Dual { v, g }
    }
    pub fn ln(self) -> Self {
        let inv = 1.0 / self.v;
        let mut g = [0.0; P];
        for k in 0..P {
            g[k] = self.g[k] * inv;
        }
        Dual { v: self.v.ln(), g }
    }
    /// `self ^ c` for a constant exponent.
    pub fn powf_c(self, c: f64) -> Self {
        let v = self.v.powf(c);
        let d = c * self.v.powf(c - 1.0); // dv/dself
        let mut g = [0.0; P];
        for k in 0..P {
            g[k] = d * self.g[k];
        }
        Dual { v, g }
    }
    /// `self ^ exp` where the exponent is also a dual. Requires `self.v > 0`.
    pub fn powd(self, e: Self) -> Self {
        let v = self.v.powf(e.v); // value kept exact (matches torch's `pow`); predict is unchanged
        // dv/dself = e·self^(e-1) = e·v/self — reuse `v` instead of a SECOND `powf`, which
        // halved this op's transcendental count (was 2·powf + ln; now 1·powf + ln + a div).
        // For P=0 (predict) `da`/`de`/`ln` are dead and elide, leaving just the one `powf`.
        let da = e.v * v / self.v; // dv/dself
        let de = v * self.v.ln(); // dv/dexp
        let mut g = [0.0; P];
        for k in 0..P {
            g[k] = da * self.g[k] + de * e.g[k];
        }
        Dual { v, g }
    }

    pub fn clamp(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            Dual::c(lo)
        } else if self.v > hi {
            Dual::c(hi)
        } else {
            self
        }
    }
    pub fn clamp_min(self, lo: f64) -> Self {
        if self.v < lo {
            Dual::c(lo)
        } else {
            self
        }
    }
    pub fn clamp_max(self, hi: f64) -> Self {
        if self.v > hi {
            Dual::c(hi)
        } else {
            self
        }
    }
    /// `min(self, o)` — gradient routed to the smaller operand (ties → self).
    pub fn min(self, o: Self) -> Self {
        if self.v <= o.v {
            self
        } else {
            o
        }
    }
    /// `max(self, o)` — gradient routed to the larger operand (ties → self).
    pub fn max(self, o: Self) -> Self {
        if self.v >= o.v {
            self
        } else {
            o
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "fp64")] // finite-diff (h=1e-6) needs f64; f32 rounding noise dominates
    #[test]
    fn dual_grad_matches_finite_difference() {
        // f(w) = clamp( exp(w0) * (w1+1)^w2 / (w0+2), 0.01, 100 )  +  min(w1, w2*3)
        const P: usize = 3;
        fn f_dual(w: [f64; P]) -> Dual<P> {
            let w0 = Dual::<P>::param(0, w[0]);
            let w1 = Dual::<P>::param(1, w[1]);
            let w2 = Dual::<P>::param(2, w[2]);
            let a = w0
                .exp()
                .mul(w1.add_c(1.0).powd(w2))
                .div(w0.add_c(2.0))
                .clamp(0.01, 100.0);
            a.add(w1.min(w2.mul_c(3.0)))
        }
        fn f_val(w: [f64; P]) -> f64 {
            let a = (w[0].exp() * (w[1] + 1.0).powf(w[2]) / (w[0] + 2.0)).clamp(0.01, 100.0);
            a + w[1].min(w[2] * 3.0)
        }
        let w = [0.7, 1.3, 0.5];
        let d = f_dual(w);
        assert!((d.v - f_val(w)).abs() < 1e-12);
        let h = 1e-6;
        for k in 0..P {
            let mut wp = w;
            let mut wm = w;
            wp[k] += h;
            wm[k] -= h;
            let num = (f_val(wp) - f_val(wm)) / (2.0 * h);
            assert!(
                (num - d.g[k]).abs() < 1e-5,
                "param {k}: dual {} vs numeric {}",
                d.g[k],
                num
            );
        }
    }
}
