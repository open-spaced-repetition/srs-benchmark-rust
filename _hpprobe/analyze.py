"""Analyse a `--hp_probe` dump: what could per-user hyperparameter selection actually buy?

Every number is a mean-over-users LogLoss, pooled per user exactly like the benchmark
(sum of eps-clipped BCE over all folds / sum of test rows), so it is directly comparable
to result/FSRS-7-*.jsonl and to the +-0.0005 gate.

Selectors priced here (candidate 0 = the shipped default = the baseline):
  always-<c>      : every user gets candidate c            (a better global default)
  ORACLE          : per user, the candidate with the lowest TEST loss.  UPPER BOUND, and
                    optimistically biased -- it fits the test noise. Not achievable.
  val->refit      : per fold, pick argmin inner-validation loss, then report that candidate's
                    model trained on 100% of the fold's train rows.  Honest, Anki-shaped.
                    Cost = 0.8*sum(epochs)/9_ep + 1.0.
  val->no refit   : same pick, but report the 80%-trained model itself (no refit).
                    Cost = 0.8*sum(epochs)/9_ep.  This is the only variant that can fit
                    under 2x with more than one candidate.
  user-level val  : pick ONE candidate per user (argmin of val summed over folds) instead of
                    one per fold. Less selection noise, still honest.
"""
import json, sys, itertools
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "result/FSRS-7-short-secs-recency-hpprobe.jsonl"
users = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if line:
        users.append(json.loads(line))
names = [c["name"] for c in users[0]["folds"][0]["cand"]]
K = len(names)
print(f"users={len(users)}  candidates={K}: {names}\n")

def user_ll(u, pick, key="test_full"):
    """pick(fold_index, fold) -> candidate index."""
    s = n = 0.0
    for i, f in enumerate(u["folds"]):
        s += f["cand"][pick(i, f)][key]
        n += f["n_test"]
    return s / n

base = [user_ll(u, lambda i, f: 0) for u in users]
mb = sum(base) / len(base)
print(f"baseline (always '{names[0]}') mean LogLoss = {mb:.6f}\n")

rows = []
for c in range(K):
    m = sum(user_ll(u, lambda i, f, c=c: c) for u in users) / len(users)
    rows.append((f"always-{names[c]}", m - mb, None))

orac = sum(
    sum(min(cd["test_full"] for cd in f["cand"]) for f in u["folds"])
    / sum(f["n_test"] for f in u["folds"])
    for u in users
) / len(users)
rows.append(("ORACLE per fold (biased)", orac - mb, None))

def argmin_val(f):
    vals = [cd["val"] for cd in f["cand"]]
    return 0 if vals[0] is None else min(range(len(vals)), key=lambda j: vals[j])

has_val = all(f["cand"][0]["val"] is not None for u in users for f in u["folds"])
m = sum(user_ll(u, lambda i, f: argmin_val(f)) for u in users) / len(users)
rows.append(("val->refit (per fold)", m - mb, "0.8*E + 1.0"))
m = sum(user_ll(u, lambda i, f: argmin_val(f), "test_val") for u in users) / len(users)
rows.append(("val->no refit (per fold)", m - mb, "0.8*E"))

def user_pick(u):
    tot = [0.0] * K
    for f in u["folds"]:
        for j, cd in enumerate(f["cand"]):
            if cd["val"] is None:
                return 0
            tot[j] += cd["val"]
    return min(range(K), key=lambda j: tot[j])

m = sum(user_ll(u, lambda i, f, p=user_pick(u): p) for u in users) / len(users)
rows.append(("user-level val->refit", m - mb, "0.8*E + 1.0"))

print(f"{'selector':<28} {'d LogLoss vs baseline':>22}   cost")
for nm, d, cost in rows:
    print(f"{nm:<28} {d:>+22.6f}   {cost or ''}")

# how often does the validation pick agree with the test pick?
agree = tot = 0
for u in users:
    for f in u["folds"]:
        if f["cand"][0]["val"] is None:
            continue
        tot += 1
        best_t = min(range(K), key=lambda j: f["cand"][j]["test_full"])
        agree += argmin_val(f) == best_t
print(f"\nvalidation pick == test-optimal pick: {agree}/{tot} = {agree/max(tot,1):.1%}  (chance = {1/K:.1%})")

# per-candidate win share on test
wins = defaultdict(int)
for u in users:
    for f in u["folds"]:
        wins[min(range(K), key=lambda j: f["cand"][j]["test_full"])] += 1
print("\ntest-optimal candidate share (per fold):")
for c in range(K):
    print(f"   {names[c]:<16} {wins[c]/max(sum(wins.values()),1):6.1%}")
