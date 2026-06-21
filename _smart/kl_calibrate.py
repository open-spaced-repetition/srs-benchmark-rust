"""Calibrate the 6 hierarchical thresholds for the KL-divergence smart-preset metric.

The Mahalanobis thresholds (1.5..12) are on a different scale, so the KL thresholds must be picked
from the observed pairwise-KL distribution. Generate the dump by running the KL smart path with the
SMART_KL_DUMP env var set, e.g.:

    SMART_KL_DUMP=_smart/kl_dist_dump.txt \
      target/release/script.exe --algo FSRS-7 --short --secs --partitions smart \
      --cluster_distance kl --cluster_method ward --cluster_threshold 0.05 \
      --data C:/Users/Andrew/anki-revlogs-10k --processes 1 --max-user-id 100

Each line of the dump is one off-diagonal deck-pair KL distance (mean over the user's rows of the
symmetric Bernoulli KL between the two decks' recall predictions). This prints the distribution and a
geometric ladder of 6 thresholds spanning ~p10 (fine / many clusters) to above the max (≈1 cluster =
per-user global), which is what goes into KL_SWEEP_THRESHOLDS in src/models/fsrs_v7.rs.
"""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "_smart/kl_dist_dump.txt"
d = np.loadtxt(path)
d = d[np.isfinite(d)]
print(f"n={len(d)}  min={d.min():.5f}  max={d.max():.4f}  mean={d.mean():.4f}")
for q in (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9):
    print(f"  p{q:>5}: {np.percentile(d, q):.5f}")

# A geometric ladder from ~p10 to a bit above the max distance (guaranteed full collapse).
lo = max(np.percentile(d, 10), 1e-4)
hi = d.max() * 1.4
ladder = np.geomspace(lo, hi, 6)
# Round to 1-2 significant figures for clean filenames.
def rnd(x):
    if x >= 0.1:
        return round(x, 2)
    return float(f"{x:.1g}")
thr = [rnd(x) for x in ladder]
print("\nsuggested KL_SWEEP_THRESHOLDS =", thr)
for t in thr:
    frac = (d < t).mean()
    print(f"  t={t:<7} merges pairs below it: {100*frac:5.1f}% of pairs")
