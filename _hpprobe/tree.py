"""Can an interpretable rule on user statistics pick the hyperparameters?

Joins the `--hp_probe` loss table to the `--hp_features` statistics on (user, fold index) and
fits a COST-SENSITIVE decision tree: each fold is one sample carrying a 12-vector of costs (the
LogLoss each candidate would contribute), and a leaf emits the candidate with the lowest summed
cost. That objective IS the benchmark metric, so no surrogate loss is involved.

Cost bookkeeping. A user's LogLoss is sum_f bce_f / N_u, and the headline is an unweighted mean
over users, so fold f of user u contributes
    g[f, c] = (bce_f[c] - bce_f[default]) / (N_u * n_users)
and the total gain of a policy is just the sum of g[f, policy(f)] over every fold. Additive, so a
tree can optimise it directly.

Everything is cross-validated over USERS (all of a user's folds move together), because a rule fit
and scored on the same users re-derives the optimism that the `size >= 8000` scan had.
"""
import json, sys
import numpy as np

PROBE = "result/FSRS-7-short-secs-recency-hpprobe.jsonl"
FEAT = "result/FSRS-7-short-secs-recency-hpfeat.jsonl"
FEATNAMES = ["n_train", "n_cards", "reviews_per_card", "batches", "p_again", "p_hard",
             "p_good", "p_easy", "median_log1p_dt", "mean_log1p_dt", "sd_log1p_dt",
             "same_day_share", "mean_pos"]

def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            o = json.loads(line)
            d[o["user"]] = o["folds"]
    return d

probe, feat = load(PROBE), load(FEAT)
users = sorted(set(probe) & set(feat))
names = [c["name"] for c in probe[users[0]][0]["cand"]]
K = len(names)
U = len(users)

X, G, uidx = [], [], []
for ui, u in enumerate(users):
    pf, ff = probe[u], feat[u]
    assert len(pf) == len(ff)
    N_u = sum(f["n_test"] for f in pf)
    for f, fe in zip(pf, ff):
        base = f["cand"][0]["test_full"]
        G.append([(f["cand"][c]["test_full"] - base) / (N_u * U) for c in range(K)])
        X.append([fe[k] for k in FEATNAMES])
        uidx.append(ui)
X = np.asarray(X, float); G = np.asarray(G, float); uidx = np.asarray(uidx)
print(f"users={U}  folds={len(X)}  candidates={K}\n")

def best_const(g):
    return int(np.argmin(g.sum(0))) if len(g) else 0

def build(idx, depth, min_leaf):
    """Greedy cost-sensitive tree. Returns a nested (feature, threshold, left, right) or a leaf."""
    g = G[idx]
    if depth == 0 or len(idx) < 2 * min_leaf:
        return ("leaf", best_const(g))
    cur = g.sum(0).min()
    best = None
    for fi in range(X.shape[1]):
        col = X[idx, fi]
        order = np.argsort(col, kind="stable")
        gs = g[order]
        cum = np.cumsum(gs, 0)
        tot = cum[-1]
        cs = col[order]
        # split after position i (left = 0..i); only where the value actually changes
        for i in range(min_leaf - 1, len(idx) - min_leaf):
            if cs[i] == cs[i + 1]:
                continue
            val = cum[i].min() + (tot - cum[i]).min()
            if best is None or val < best[0]:
                best = (val, fi, 0.5 * (cs[i] + cs[i + 1]), order[: i + 1], order[i + 1 :])
    if best is None or best[0] >= cur - 1e-15:
        return ("leaf", best_const(g))
    _, fi, thr, li, ri = best
    return ("split", fi, thr, build(idx[li], depth - 1, min_leaf), build(idx[ri], depth - 1, min_leaf))

def apply(node, x):
    while node[0] == "split":
        node = node[3] if x[node[1]] <= node[2] else node[4]
    return node[1]

def show(node, ind="  "):
    if node[0] == "leaf":
        return f"{ind}-> {names[node[1]]}\n"
    f, t = FEATNAMES[node[1]], node[2]
    return (f"{ind}if {f} <= {t:.4g}:\n" + show(node[3], ind + "  ")
            + f"{ind}else:\n" + show(node[4], ind + "  "))

# --- reference points -------------------------------------------------------
print(f"{'policy':<34}{'gain':>12}")
for c in range(K):
    print(f"{'always-' + names[c]:<34}{G[:, c].sum():>+12.6f}")
print(f"{'ORACLE per fold (biased)':<34}{G.min(1).sum():>+12.6f}")

# --- cross-validated trees --------------------------------------------------
rng = np.random.default_rng(0)
perm = rng.permutation(U)
foldof = np.empty(U, int)
for i, u in enumerate(perm):
    foldof[u] = i % 5

print()
for depth in (1, 2, 3, 4, 6):
    for min_leaf in (50, 200):
        oos = 0.0
        for k in range(5):
            tr = np.flatnonzero(foldof[uidx] != k)
            te = np.flatnonzero(foldof[uidx] == k)
            t = build(tr, depth, min_leaf)
            for i in te:
                oos += G[i, apply(t, X[i])]
        print(f"tree depth={depth} min_leaf={min_leaf:<4} cross-validated gain = {oos:+.6f}")

t = build(np.arange(len(X)), 3, 200)
print(f"\ndepth-3 tree fit on ALL users (in-sample gain {sum(G[i, apply(t, X[i])] for i in range(len(X))):+.6f}):")
print(show(t), end="")
