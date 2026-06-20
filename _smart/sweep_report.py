"""Aggregate the smart-preset sweep(s) into the xlsx metrics table.

Auto-discovers every result/<base>-smart-*.jsonl experiment, and reports LogLoss + RMSE(bins)
under weight=users (unweighted mean) and weight=reviews (size-weighted), each with Δ relative
(%) vs exp 0 (the <base> baseline). Compares on users common to the baseline + each experiment.

Usage: python _smart/sweep_report.py [base=FSRS-7-short-secs] [result_dir=result] [maxuser]
"""
import json, os, sys, glob, re

base = sys.argv[1] if len(sys.argv) > 1 else "FSRS-7-short-secs"
rdir = sys.argv[2] if len(sys.argv) > 2 else "result"
maxu = int(sys.argv[3]) if len(sys.argv) > 3 else None

def load(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
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

exp0 = load(f"{rdir}/{base}.jsonl")
if not exp0:
    print(f"baseline {rdir}/{base}.jsonl not found / empty")
    sys.exit(1)

# discover experiments
paths = sorted(glob.glob(f"{rdir}/{base}-smart-*.jsonl"))

def sort_key(p):
    """Order hierarchical (method, threshold) then hdbscan (mcs, ms, eom/leaf)."""
    s = os.path.basename(p)[len(base) + len("-smart-"):-len(".jsonl")]
    methods = ["single", "complete", "average", "centroid", "ward"]
    if s.startswith("hdbscan"):
        m = re.match(r"hdbscan-mcs(\d+)-ms(\d+)-(\w+)", s)
        return (1, int(m.group(1)), int(m.group(2)), m.group(3) != "eom", s)
    m = re.match(r"([a-z]+)-([\d.]+)", s)
    return (0, methods.index(m.group(1)) if m.group(1) in methods else 9, float(m.group(2)), 0, s)

paths.sort(key=sort_key)

def agg(d, common, key, weighted):
    vals = [d[u]["metrics"][key] for u in common]
    if not weighted:
        return sum(vals) / len(vals)
    sizes = [d[u]["size"] for u in common]
    return sum(v * s for v, s in zip(vals, sizes)) / sum(sizes)

# baseline numbers on its own user set (per-experiment common set recomputed below)
hdr = (f"{'experiment':>34} {'users':>5} | {'LL_u':>9} {'dLL_u%':>7} {'RMSEb_u':>8} {'dR_u%':>7}"
       f" | {'LL_r':>9} {'dLL_r%':>7} {'RMSEb_r':>8} {'dR_r%':>7}")
print(hdr)
print("-" * len(hdr))

for path in [f"{rdir}/{base}.jsonl"] + paths:
    exp = load(path)
    common = sorted(set(exp0) & set(exp))
    if not common:
        continue
    name = os.path.basename(path)[:-len(".jsonl")]
    cells = []
    for w in (False, True):
        ll = agg(exp, common, "LogLoss", w)
        rb = agg(exp, common, "RMSE(bins)", w)
        bll = agg(exp0, common, "LogLoss", w)
        brb = agg(exp0, common, "RMSE(bins)", w)
        dll = (ll - bll) / bll * 100
        drb = (rb - brb) / brb * 100
        cells += [f"{ll:>9.6f}", f"{dll:>+7.2f}", f"{rb:>8.5f}", f"{drb:>+7.2f}"]
    label = name.replace(f"{base}-smart-", "").replace(base, "(baseline)")
    print(f"{label:>34} {len(common):>5} | {cells[0]} {cells[1]} {cells[2]} {cells[3]}"
          f" | {cells[4]} {cells[5]} {cells[6]} {cells[7]}")
