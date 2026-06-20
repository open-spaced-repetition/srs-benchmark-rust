"""Step 1 of smart-preset assignment: fit the robust covariance reference.

Loads every user's GLOBAL FSRS-7 params (parameters['0']) from the
FSRS-7-short-secs-recency benchmark over all 10k users, log-transforms params
[0,1,2,3], fits MinCovDet(random_state=43), and stores center + precision +
whitening matrix to JSON for the Rust benchmark to consume.

This mirrors `fit_precision_min_cov_det` in
"FSRS smart preset assignment (simple).py" EXACTLY (random_state=43, precision
= pinv(cov), whitening = diag(sqrt(eigvals(precision))) @ eigvecs.T, no
shrinkage).

Usage:
    python _smart/fit_covariance.py [recency.jsonl] [out.json]
"""
import json
import sys
import numpy as np
from sklearn.covariance import MinCovDet

LOG_PARAM_IDXS = [0, 1, 2, 3]
LOG_EPS = 1e-12

src = sys.argv[1] if len(sys.argv) > 1 else "result/FSRS-7-short-secs-recency.jsonl"
out = sys.argv[2] if len(sys.argv) > 2 else "_smart/smart_preset_cov.json"

params = []
for line in open(src, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    params.append(obj["parameters"]["0"])
params = np.asarray(params, dtype=float)
print(f"loaded {params.shape[0]} param vectors, dim {params.shape[1]}")

# Log-transform the scale-like params (simple prototype: log(x + eps), error if <=0).
pt = params.copy()
for idx in LOG_PARAM_IDXS:
    if np.any(pt[:, idx] <= 0):
        bad = pt[pt[:, idx] <= 0, idx][:10]
        raise ValueError(f"non-positive at param {idx}: {bad}")
    pt[:, idx] = np.log(pt[:, idx] + LOG_EPS)

mcd = MinCovDet(random_state=43).fit(pt)
center = np.asarray(mcd.location_, dtype=float)
cov = np.asarray(mcd.covariance_, dtype=float)
precision = np.linalg.pinv(cov)

evals, evecs = np.linalg.eigh(precision)
evals = np.maximum(evals, 0.0)
whitening = np.diag(np.sqrt(evals)) @ evecs.T  # z = (x-center) @ whitening.T

print(f"cov det: {np.linalg.det(cov):.6e}")
json.dump(
    {
        "center": center.tolist(),
        "precision": precision.tolist(),
        "whitening": whitening.tolist(),
        "covariance": cov.tolist(),
        "n_users": int(params.shape[0]),
        "dim": int(params.shape[1]),
        "log_param_idxs": LOG_PARAM_IDXS,
        "log_eps": LOG_EPS,
        "random_state": 43,
    },
    open(out, "w"),
)
print(f"wrote {out}")
