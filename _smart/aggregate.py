"""Aggregate a smart-preset experiment vs the baseline into the xlsx metrics:
LogLoss + RMSE(bins), each as weight=users (unweighted mean) and weight=reviews
(size-weighted mean), with Δ (relative, %) vs the baseline.

Usage: python _smart/aggregate.py <experiment.jsonl> <baseline.jsonl> [maxuser]
"""
import json, sys

def load(p, maxu):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "metrics" not in o:
            continue
        if maxu and o["user"] > maxu:
            continue
        d[o["user"]] = o
    return d

def agg(rows, key):
    vals = [r["metrics"][key] for r in rows]
    sizes = [r["size"] for r in rows]
    mean_u = sum(vals) / len(vals)
    mean_r = sum(v * s for v, s in zip(vals, sizes)) / sum(sizes)
    return mean_u, mean_r

exp_path, base_path = sys.argv[1], sys.argv[2]
maxu = int(sys.argv[3]) if len(sys.argv) > 3 else None
exp = load(exp_path, maxu)
base = load(base_path, maxu)
common = sorted(set(exp) & set(base))
er = [exp[u] for u in common]
br = [base[u] for u in common]
print(f"experiment: {exp_path}")
print(f"baseline:   {base_path}")
print(f"users compared: {len(common)}   total reviews: {sum(r['size'] for r in er):,}")
# size sanity
mism = sum(1 for u in common if exp[u]["size"] != base[u]["size"])
print(f"size mismatches vs baseline: {mism}")
print(f"{'metric':>12} {'weight':>8} {'experiment':>12} {'baseline':>12} {'d_abs':>11} {'d_rel%':>8}")
for key in ["LogLoss", "RMSE(bins)"]:
    eu, er_ = agg(er, key)
    bu, br_ = agg(br, key)
    for name, e, b in [("users", eu, bu), ("reviews", er_, br_)]:
        drel = (e - b) / b * 100
        print(f"{key:>12} {name:>8} {e:>12.6f} {b:>12.6f} {e-b:>+11.6f} {drel:>+8.3f}")
