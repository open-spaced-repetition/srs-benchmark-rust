//! Hand-written analytic gradient for the DASH[ACT-R] BCE-loss gradient — a closed-form VJP of the
//! exact forward in [`super::dash_act_r::retention`]. Replaces forward-mode `Dual<5>`. f64.
//!
//! DASH[ACT-R] is STATIC (not a recurrence): `p = sigmoid(w0·log(1 + max(Σ_k term_k, 0)) + w4)`,
//! with `term_k = t_k^(-w1)·(r_k>1 ? w3 : w2)` over prior reviews with time-to-now `t_k > 0.1`. So
//! the gradient is one O(n) pass: the sigmoid+BCE seed simplifies to `g_inner = weight·(p-y)`.
//!
//! ⚠ Manual VJP of a SPECIFIC forward — change the formula and re-derive. Guarded by
//! `dash_actr_analytic_grad_matches_forward_mode` (gated `--features fp64`).

const NP: usize = 5;

/// Forward + analytic backward for ONE row; accumulates d(weight·BCE)/dw into `gw`, returns `p`.
/// `intervals` = `intervals_from_second(row)` (its reverse cumsum is the time-to-now per review).
#[allow(clippy::too_many_arguments)]
pub fn grad_one(w: &[f64], prior_ratings: &[i64], intervals: &[f64], y: f64, weight: f64, gw: &mut [f64]) -> f64 {
    let n = intervals.len();
    // time-to-now = reverse cumulative sum of intervals
    let mut ttn = vec![0.0f64; n];
    let mut acc = 0.0;
    for k in (0..n).rev() {
        acc += intervals[k];
        ttn[k] = acc;
    }
    // forward sum + per-parameter sub-sums (all over the t>0.1 terms)
    let mut sum = 0.0;
    let mut dsum_dw1 = 0.0; // Σ term_k·(-ln t_k)
    let mut dsum_dw2 = 0.0; // Σ_{r<=1} t_k^-w1
    let mut dsum_dw3 = 0.0; // Σ_{r>1}  t_k^-w1
    for k in 0..n {
        let t = ttn[k];
        if t > 0.1 {
            let tp = t.powf(-w[1]); // t^-w1
            let mult = if prior_ratings[k] > 1 { w[3] } else { w[2] };
            let term = tp * mult;
            sum += term;
            dsum_dw1 += term * (-t.ln());
            if prior_ratings[k] > 1 {
                dsum_dw3 += tp;
            } else {
                dsum_dw2 += tp;
            }
        }
    }
    let sp = sum.max(0.0); // max(sum, 0)
    let l = (1.0 + sp).ln(); // log(1 + sp)
    let inner = w[0] * l + w[4];
    let p = 1.0 / (1.0 + (-inner).exp()); // sigmoid

    // d(weight·BCE)/dinner = weight·(p-y)/(p(1-p)) · p(1-p) = weight·(p-y)
    let g_inner = weight * (p - y);
    gw[0] += g_inner * l;
    gw[4] += g_inner;
    let g_l = g_inner * w[0];
    let g_sp = g_l / (1.0 + sp); // l = log(1+sp)
    let g_sum = if sum > 0.0 { g_sp } else { 0.0 }; // sp = max(sum,0)
    gw[1] += g_sum * dsum_dw1;
    gw[2] += g_sum * dsum_dw2;
    gw[3] += g_sum * dsum_dw3;
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Dual;
    use crate::models::dash_act_r::{retention_dual, INIT_W};

    fn forward_mode(prior_r: &[i64], intervals: &[f64], w: &[f64; NP]) -> (f64, [f64; NP]) {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, w[k]));
        let ret = retention_dual(prior_r, intervals, &wd);
        (ret.v, ret.g)
    }

    #[cfg(feature = "fp64")]
    #[test]
    fn dash_actr_analytic_grad_matches_forward_mode() {
        let cases: Vec<(Vec<i64>, Vec<f64>)> = vec![
            (vec![3, 1, 4, 2, 3], vec![2.0, 9.0, 0.05, 1.5, 30.0]),
            (vec![1, 1, 2], vec![0.5, 5.0, 12.0]),
            (vec![4, 3, 2, 1, 3, 4], vec![1.0, 0.05, 3.0, 20.0, 0.2, 7.0]),
        ];
        let mut w = INIT_W;
        for (k, wk) in w.iter_mut().enumerate() {
            *wk += 0.01 * (k as f64).sin();
        }
        for (prior_r, intervals) in &cases {
            let (y, weight) = (1.0, 0.7);
            let mut gw = vec![0.0f64; NP];
            let p = grad_one(&w, prior_r, intervals, y, weight, &mut gw);
            let (p_fwd, g_fwd) = forward_mode(prior_r, intervals, &w);
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
