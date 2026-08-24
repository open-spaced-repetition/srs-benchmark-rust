"""Cost/gain frontier. COST = training time relative to the shipped default (9 epochs), i.e.
n_epoch/9 for the chosen candidate (hypergradient adds one 34-element dot product per step, which
is not measurable next to a batch gradient). A policy's cost is the mean cost of what it picks.
No selection overhead is counted here -- these are all zero-search policies."""
import json
import numpy as np

exec(open("_hpprobe/tree.py").read().split("# --- reference points")[0])
EPOCHS = {"default":9,"ep20":20,"ep45":45,"lr_half":9,"lr_double":9,"betas_torch":9,
          "betas_low":9,"lr_double_ep20":20,"hg0.02":9,"hg0.05":9,"hg0.15":9,"hg0.05_ep20":20}
cost = np.array([EPOCHS[n]/9 for n in names])

print(f"{'policy':<26}{'gain':>11}{'cost':>8}")
for c in np.argsort([G[:, c].sum() for c in range(K)]):
    print(f"{'always-'+names[c]:<26}{G[:, c].sum():>+11.6f}{cost[c]:>7.2f}x")

rng = np.random.default_rng(0)
perm = rng.permutation(U); foldof = np.empty(U, int)
for i, u in enumerate(perm): foldof[u] = i % 5

def cv_tree(allowed, depth=3, min_leaf=200):
    """Cross-validated tree restricted to `allowed` candidate indices."""
    global G
    full = G
    G = full[:, allowed]
    try:
        oos = 0.0; costs = []
        for k in range(5):
            tr = np.flatnonzero(foldof[uidx] != k); te = np.flatnonzero(foldof[uidx] == k)
            t = build(tr, depth, min_leaf)
            for i in te:
                j = apply(t, X[i]); oos += G[i, j]; costs.append(cost[allowed[j]])
        return oos, float(np.mean(costs))
    finally:
        G = full

cheap = [i for i in range(K) if cost[i] <= 2.0]
print(f"\ncandidates costing <=2x: {[names[i] for i in cheap]}")
for label, allowed in (("all 12 candidates", list(range(K))), ("<=2x candidates only", cheap)):
    g, c = cv_tree(allowed)
    print(f"  tree, {label:<22} cross-validated gain {g:+.6f}   mean cost {c:.2f}x")
