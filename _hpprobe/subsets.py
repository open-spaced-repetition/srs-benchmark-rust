"""If the <2x budget is relaxed, what does honest per-user selection buy, and at what price?

Selector = train each candidate on the first 80% of the fold's train rows, score on the last 20%,
pick the candidate with the lowest validation loss SUMMED over the user's folds (one config per
user -- pooling folds cuts the selection noise), then refit that one on 100%.
    cost = 0.8 * sum(epochs)/9  (the screening runs)  +  epochs(winner)/9  (the refit)
"""
import json, itertools
import numpy as np

exec(open("_hpprobe/tree.py").read().split("# --- reference points")[0])
EPOCHS = np.array([{"default":9,"ep20":20,"ep45":45,"lr_half":9,"lr_double":9,"betas_torch":9,
                    "betas_low":9,"lr_double_ep20":20,"hg0.02":9,"hg0.05":9,"hg0.15":9,
                    "hg0.05_ep20":20}[n] for n in names]) / 9.0

probe = load(PROBE)
VAL = {}   # user -> (K,) summed validation loss
for u in users:
    v = np.zeros(K)
    ok = True
    for f in probe[u]:
        for c in range(K):
            x = f["cand"][c]["val"]
            if x is None: ok = False
            else: v[c] += x
    VAL[u] = v if ok else None

Gu = {}    # user -> (K,) gain contribution per candidate
for i, u in enumerate(users):
    Gu[u] = G[uidx == i].sum(0)

def score(sub):
    sub = list(sub); tot = 0.0; cst = 0.0
    for u in users:
        v = VAL[u]
        j = sub[int(np.argmin(v[sub]))] if v is not None else 0
        tot += Gu[u][j]
        cst += 0.8 * EPOCHS[sub].sum() + EPOCHS[j]
    return tot, cst / len(users)

print(f"{'candidate subset':<44}{'gain':>11}{'cost':>8}")
subs = [("default","hg0.05"), ("default","lr_double"), ("default","ep20"), ("default","ep45"),
        ("default","ep20","ep45"), ("default","lr_double","ep20"),
        ("default","lr_half","lr_double"), ("default","ep20","ep45","lr_double_ep20"),
        tuple(names)]
for sub in subs:
    idx = [names.index(x) for x in sub]
    g, c = score(idx)
    label = "+".join(sub) if len(sub) < 5 else f"all {len(sub)}"
    print(f"{label:<44}{g:>+11.6f}{c:>7.2f}x")
