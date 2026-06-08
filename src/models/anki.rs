//! Anki scheduler (trainable) — `models/anki.py`. A 7-param Adam-trained interval/ease state
//! machine emulating Anki's SM-2-derived scheduler, with a fixed `0.9^(t/s)` forgetting
//! curve. Uses the same BaseModel training hyperparameters as FSRS (lr=4e-2, n_epoch=5),
//! so `TrainConfig::default()` applies.
//!
//! `step` branches on rating (data) and on `days_late < 0` (param-dependent via `ivl`); under
//! forward-mode autodiff each branch's dual is carried for the value-selected path, matching
//! torch's `where`/`max`/`leaky_relu` gradient routing.

use super::{recency_weights, ModelOutput};
use crate::autodiff::Dual;
use crate::config::Config;
use crate::eval::Params;
use crate::features::{Dataset, Row};
use crate::split::time_series_split;
use crate::train::{self, BatchModel, TrainConfig};

const NP: usize = 7;
// graduating ivl, easy ivl, starting ease, easy bonus, hard ivl, new ivl, ivl multiplier
const INIT_W: [f64; NP] = [1.0, 4.0, 2.5, 1.3, 1.2, 0.0, 1.0];

#[inline]
fn fc<const P: usize>(t: f64, s: Dual<P>) -> Dual<P> {
    Dual::<P>::c(0.9).powd(Dual::<P>::c(t).div(s))
}

/// `leaky_relu(x)` with torch's default negative slope 0.01.
#[inline]
fn leaky_relu<const P: usize>(x: Dual<P>) -> Dual<P> {
    if x.v >= 0.0 {
        x
    } else {
        x.mul_c(0.01)
    }
}

/// `passing_early_review_intervals` — `elapsed = ivl + days_late = delta_t` (a constant w.r.t.
/// params: the `ivl` terms cancel, matching torch's zero gradient there).
fn passing_early<const P: usize>(rating: i64, ease: Dual<P>, ivl: Dual<P>, elapsed: f64, w: &[Dual<P>; NP]) -> Dual<P> {
    match rating {
        2 => Dual::<P>::c(elapsed).mul(w[4]).max(ivl.mul(w[4]).mul_c(0.5)),
        3 => Dual::<P>::c(elapsed).mul(ease).max(ivl),
        _ => {
            // rating == 4: max(elapsed*ease, ivl) * (w3 - (w3-1)/2) = ... * (w3+1)/2
            let base = Dual::<P>::c(elapsed).mul(ease).max(ivl);
            base.mul(w[3].add_c(1.0).mul_c(0.5))
        }
    }
}

/// `passing_nonearly_review_intervals`.
fn passing_nonearly<const P: usize>(rating: i64, ease: Dual<P>, ivl: Dual<P>, days_late: Dual<P>, w: &[Dual<P>; NP]) -> Dual<P> {
    match rating {
        2 => ivl.mul(w[4]),
        3 => ivl.add(days_late.mul_c(0.5)).mul(ease),
        _ => ivl.add(days_late).mul(ease).mul(w[3]), // rating == 4
    }
}

fn retention<const P: usize>(prior_dt: &[f64], prior_r: &[i64], cur_dt: f64, w: &[Dual<P>; NP], s_min: f64, s_max: f64) -> Dual<P> {
    let mut ivl = Dual::<P>::c(0.0);
    let mut ease = Dual::<P>::c(0.0);
    for k in 0..prior_r.len() {
        let rating = prior_r[k];
        let (new_ivl, new_ease) = if k == 0 {
            // first learn: ivl = (rating<4 ? w0 : w1), ease = w2
            let nivl = if rating < 4 { w[0] } else { w[1] };
            (nivl, w[2])
        } else {
            let dt = prior_dt[k];
            let days_late = Dual::<P>::c(dt).sub(ivl); // delta_t - ivl
            let nivl = if rating == 1 {
                ivl.mul(w[5])
            } else {
                let body = if days_late.v < 0.0 {
                    passing_early(rating, ease, ivl, dt, w)
                } else {
                    passing_nonearly(rating, ease, ivl, days_late, w)
                };
                body.mul(w[6])
            };
            let nease = match rating {
                1 => ease.add_c(-0.2),
                2 => ease.add_c(-0.15),
                4 => ease.add_c(0.15),
                _ => ease, // rating == 3 unchanged
            };
            (nivl, nease)
        };
        ease = new_ease.clamp(1.3, 5.5);
        // new_ivl = max(leaky_relu(new_ivl - 1) + 1, new_ivl).clamp(s_min, s_max)
        let soft = leaky_relu(new_ivl.add_c(-1.0)).add_c(1.0);
        ivl = soft.max(new_ivl).clamp(s_min, s_max);
    }
    fc(cur_dt, ivl)
}

struct Anki<'a> {
    ds: &'a Dataset,
    rows: Vec<Row>,
    weights: Vec<f64>,
    s_min: f64,
    s_max: f64,
}

impl<'a> Anki<'a> {
    fn build(ds: &'a Dataset, rows: &[Row], weights: &[f64], max_seq_len: Option<usize>, cfg: &Config) -> Self {
        let mut out_rows = Vec::with_capacity(rows.len());
        let mut out_w = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            if let Some(m) = max_seq_len {
                if r.pos as usize > m {
                    continue;
                }
            }
            out_rows.push(r.clone());
            out_w.push(weights[i]);
        }
        Anki { ds, rows: out_rows, weights: out_w, s_min: cfg.s_min, s_max: cfg.s_max }
    }
    fn ret<const P: usize>(&self, w: &[Dual<P>; NP], row: &Row) -> Dual<P> {
        retention(self.ds.prior_dt_active(row), self.ds.prior_ratings(row), row.delta_t, w, self.s_min, self.s_max)
    }
}

impl BatchModel for Anki<'_> {
    fn n_params(&self) -> usize {
        NP
    }
    fn init_params(&self) -> Vec<f64> {
        INIT_W.to_vec()
    }
    fn n_rows(&self) -> usize {
        self.rows.len()
    }
    fn seq_len(&self, row: usize) -> usize {
        self.rows[row].pos as usize
    }
    fn y(&self, row: usize) -> f64 {
        self.rows[row].y as f64
    }
    fn weight(&self, row: usize) -> f64 {
        self.weights[row]
    }
    fn predict(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(params[k]));
        idx.iter().map(|&i| self.ret(&wd, &self.rows[i]).v).collect()
    }
    fn grad(&self, params: &[f64], idx: &[usize]) -> Vec<f64> {
        let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, params[k]));
        let mut g = vec![0.0f64; NP];
        for &i in idx {
            let ret = self.ret(&wd, &self.rows[i]);
            let p = ret.v;
            let denom = (p * (1.0 - p)).max(1e-12);
            let dl = self.weights[i] * (p - self.rows[i].y as f64) / denom;
            for k in 0..NP {
                g[k] += dl * ret.g[k];
            }
        }
        g
    }
    fn clip_params(&self, w: &mut [f64]) {
        // AnkiParameterClipper (limits from Anki 24.11).
        w[0] = w[0].clamp(1.0, 9999.0);
        w[1] = w[1].clamp(1.0, 9999.0);
        w[2] = w[2].clamp(1.31, 5.0);
        w[3] = w[3].clamp(1.0, 5.0);
        w[4] = w[4].clamp(0.5, 1.3);
        w[5] = w[5].clamp(0.0, 1.0);
        w[6] = w[6].clamp(0.5, 2.0);
    }
}

pub fn process(ds: &Dataset, cfg: &Config) -> ModelOutput {
    let rows = &ds.rows;
    let splits = time_series_split(rows.len(), cfg.n_splits);
    let tc = TrainConfig::default();
    let mut eval_rows = Vec::new();
    let mut p = Vec::new();
    let mut last_w = INIT_W.to_vec();

    for s in splits {
        let train = &rows[..s.test_start];
        let w = if cfg.default_params {
            INIT_W.to_vec()
        } else {
            let weights = recency_weights(train.len(), cfg.use_recency_weighting);
            let model = Anki::build(ds, train, &weights, Some(cfg.max_seq_len), cfg);
            train::train(&model, &tc)
        };
        let test = &rows[s.test_start..s.test_end];
        let tm = Anki::build(ds, test, &vec![1.0; test.len()], None, cfg);
        let all: Vec<usize> = (0..tm.rows.len()).collect();
        for (i, pr) in tm.predict(&w, &all).into_iter().enumerate() {
            eval_rows.push(tm.rows[i].clone());
            p.push(pr);
        }
        last_w = w;
    }

    ModelOutput {
        eval_rows,
        p,
        params: Params::Partitioned(vec![("0".to_string(), last_w)]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anki_grad_matches_finite_difference() {
        let prior_dt = [0.0, 2.0, 9.0, 1.5, 30.0];
        let prior_r = [3i64, 2, 3, 4, 1];
        let (cur_dt, s_min, s_max) = (7.0, 0.0001, 36500.0);
        let grad = {
            let wd: [Dual<NP>; NP] = std::array::from_fn(|k| Dual::param(k, INIT_W[k]));
            retention(&prior_dt, &prior_r, cur_dt, &wd, s_min, s_max).g
        };
        let val = |w: [f64; NP]| {
            let wd: [Dual<0>; NP] = std::array::from_fn(|k| Dual::c(w[k]));
            retention(&prior_dt, &prior_r, cur_dt, &wd, s_min, s_max).v
        };
        let h = 1e-6;
        for k in 0..NP {
            let mut wp = INIT_W;
            let mut wm = INIT_W;
            wp[k] += h;
            wm[k] -= h;
            let num = (val(wp) - val(wm)) / (2.0 * h);
            assert!((num - grad[k]).abs() < 1e-4, "param {k}: {} vs {}", grad[k], num);
        }
    }
}
