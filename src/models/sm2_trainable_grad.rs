//! Hand-written reverse-mode VJP for the SM2-trainable BCE-loss gradient — a manual reverse-mode of
//! the exact forward in [`super::sm2_trainable::retention`]. Replaces forward-mode `Dual<6>`. f64.
//!
//! SM2-trainable (NP=6): a 2-state machine (interval `ivl`, ease-factor `ef`; `reps` is data). Per
//! prior rating: `new_ivl = {reps==1: w0, reps==2: w1, else: ivl·ef}`,
//! `new_ef = ef - w3·(rating+1-w4)² + w5`, clamped; `ef` starts at `w2`. Curve `0.9^(t/s)`.
//!
//! ⚠ Manual VJP of a SPECIFIC forward — change the formula and re-derive. Guarded by
//! `sm2_analytic_grad_matches_forward_mode` (gated `--features fp64`).

const NP: usize = 6;
const LN09: f64 = -0.10536051565782628; // ln(0.9)

pub struct WConsts {
    s_min: f64,
    s_max: f64,
}

pub fn wconsts(s_min: f64, s_max: f64) -> WConsts {
    WConsts { s_min, s_max }
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
    branch: u8, // 0 = w0, 1 = w1, 2 = ivl·ef
    ivl: f64,   // input interval
    ef: f64,    // input ease-factor
    diff: f64,  // (rating+1) - w4
    ivl_in_open: bool,
    ef_in_open: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn grad_one(
    w: &[f64],
    prior_r: &[i64],
    cur_dt: f64,
    y: f64,
    weight: f64,
    wc: &WConsts,
    gw: &mut [f64],
    caches: &mut Vec<StepCache>,
) -> f64 {
    caches.clear();
    let mut ivl = 0.0f64;
    let mut ef = w[2]; // initial ease-factor
    let mut reps: i64 = 0;
    for &rating in prior_r {
        let new_reps = if rating > 1 { reps + 1 } else { 1 };
        let (branch, new_ivl_pre) = if new_reps == 1 {
            (0u8, w[0])
        } else if new_reps == 2 {
            (1u8, w[1])
        } else {
            (2u8, ivl * ef)
        };
        let diff = (rating + 1) as f64 - w[4];
        let new_ef_pre = ef - w[3] * diff * diff + w[5];
        caches.push(StepCache {
            branch,
            ivl,
            ef,
            diff,
            ivl_in_open: new_ivl_pre >= wc.s_min && new_ivl_pre <= wc.s_max,
            ef_in_open: new_ef_pre >= 1.3 && new_ef_pre <= 10.0,
        });
        ivl = new_ivl_pre.clamp(wc.s_min, wc.s_max);
        ef = new_ef_pre.clamp(1.3, 10.0);
        reps = new_reps;
    }
    let p = curve_fwd(cur_dt, ivl);
    let denom = (p * (1.0 - p)).max(1e-12);
    let g_p = weight * (p - y) / denom;
    let mut g_ivl = curve_bwd(p, cur_dt, ivl, g_p);
    let mut g_ef = 0.0;
    for c in caches.iter().rev() {
        let g_new_ivl = if c.ivl_in_open { g_ivl } else { 0.0 };
        let g_new_ef = if c.ef_in_open { g_ef } else { 0.0 };
        let mut gi = 0.0;
        let mut ge = 0.0;
        match c.branch {
            0 => gw[0] += g_new_ivl,
            1 => gw[1] += g_new_ivl,
            _ => {
                // new_ivl = ivl·ef
                gi += g_new_ivl * c.ef;
                ge += g_new_ivl * c.ivl;
            }
        }
        // new_ef = ef - w3·diff² + w5 ; diff = (rating+1) - w4
        ge += g_new_ef;
        gw[3] += g_new_ef * (-c.diff * c.diff);
        gw[4] += g_new_ef * (2.0 * w[3] * c.diff); // ∂(-w3·diff²)/∂w4 = 2·w3·diff
        gw[5] += g_new_ef;
        g_ivl = gi;
        g_ef = ge;
    }
    // initial ef = w2 (initial ivl = 0 is constant)
    gw[2] += g_ef;
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Dual;
    use crate::models::sm2_trainable::{retention_dual, INIT_W};

    fn forward_mode(prior_r: &[i64], cur_dt: f64, w: &[f64; NP], s_min: f64, s_max: f64) -> (f64, [f64; NP]) {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w[k]));
        let ret = retention_dual(prior_r, cur_dt, &wd, s_min, s_max);
        (ret.v, ret.g)
    }

    #[cfg(feature = "fp64")]
    #[test]
    fn sm2_analytic_grad_matches_forward_mode() {
        let cases: Vec<(Vec<i64>, f64)> = vec![
            (vec![3, 1, 3, 4, 2, 3, 3], 7.0),
            (vec![1, 1, 2, 3], 12.0),
            (vec![4, 4, 4, 2, 1, 3], 3.0),
        ];
        let (s_min, s_max) = (0.0001, 36500.0);
        let mut w = INIT_W;
        for (k, wk) in w.iter_mut().enumerate() {
            *wk += 0.01 * (k as f64).sin();
        }
        for (prior_r, cur_dt) in &cases {
            let wc = wconsts(s_min, s_max);
            let (y, weight) = (1.0, 0.7);
            let mut gw = vec![0.0f64; NP];
            let mut caches = Vec::new();
            let p = grad_one(&w, prior_r, *cur_dt, y, weight, &wc, &mut gw, &mut caches);
            let (p_fwd, g_fwd) = forward_mode(prior_r, *cur_dt, &w, s_min, s_max);
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
