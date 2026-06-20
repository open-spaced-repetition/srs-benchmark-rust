//! Smart-preset covariance reference (step 1–2 of the pipeline, consumed at benchmark time).
//!
//! Loads the precomputed MinCovDet whitening (from `_smart/fit_covariance.py`, fit on the global
//! FSRS-7 `--short --secs --recency` params of all 10k users) and whitens a deck's FSRS-7 param
//! vector so that Euclidean distance in the whitened space equals Mahalanobis distance in
//! log-param space — the distance metric the clustering (`crate::cluster`) groups decks by.

use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Deserialize)]
struct CovJson {
    center: Vec<f64>,
    whitening: Vec<Vec<f64>>,
    log_param_idxs: Vec<usize>,
    log_eps: f64,
}

/// Whitening reference: `z = (log_transform(x) - center) @ whitening.T`.
pub struct Cov {
    center: Vec<f64>,
    whitening: Vec<Vec<f64>>,
    log_idxs: Vec<usize>,
    log_eps: f64,
}

static COV: OnceLock<Cov> = OnceLock::new();

/// Load the covariance JSON and install it as the process-global reference. Called once from
/// `run::run` (before the parallel loop) when `--partitions smart`, so a missing/bad file fails
/// fast instead of panicking inside a worker.
pub fn load_global(path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read smart covariance {path}: {e} (run _smart/fit_covariance.py first)"))?;
    let j: CovJson = serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))?;
    let d = j.center.len();
    if j.whitening.len() != d || j.whitening.iter().any(|r| r.len() != d) {
        return Err(format!("{path}: whitening must be {d}x{d}"));
    }
    let cov = Cov { center: j.center, whitening: j.whitening, log_idxs: j.log_param_idxs, log_eps: j.log_eps };
    COV.set(cov).map_err(|_| "smart covariance already loaded".to_string())
}

/// The process-global covariance (must have been installed via [`load_global`]).
pub fn global() -> &'static Cov {
    COV.get().expect("smart covariance not loaded (call smart::load_global first)")
}

impl Cov {
    pub fn dim(&self) -> usize {
        self.center.len()
    }

    /// Whiten one FSRS-7 param vector: log-transform the scale params, then `(x-center) @ W.T`.
    pub fn whiten(&self, params: &[f64]) -> Vec<f64> {
        let mut x = params.to_vec();
        for &i in &self.log_idxs {
            x[i] = (x[i] + self.log_eps).ln();
        }
        let d = self.center.len();
        let mut z = vec![0.0f64; d];
        for (r, zr) in z.iter_mut().enumerate() {
            let mut s = 0.0;
            for c in 0..d {
                s += (x[c] - self.center[c]) * self.whitening[r][c];
            }
            *zr = s;
        }
        z
    }
}
