"""Compare result/<name>.jsonl against an explicit reference jsonl (e.g. the Python
srs-benchmark result). Reports size-exactness and mean dLogLoss over common users (<= max id).

Usage: python _phase2/compare_ref.py <name> <ref_path> [max_user_id]
"""
import json, sys

def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            o = json.loads(line); d[o["user"]] = o
    return d

name = sys.argv[1]
ref_path = sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

new = load(f"result/{name}.jsonl")
ref = load(ref_path)
common = sorted(u for u in (set(new) & set(ref)) if u <= nmax)
size_ok = True
sum_new = sum_ref = 0
diffs = []
for u in common:
    n, r = new[u], ref[u]
    sum_new += n["size"]; sum_ref += r["size"]
    if n["size"] != r["size"]:
        size_ok = False
        print(f"  SIZE MISMATCH user {u}: new {n['size']} ref {r['size']}")
    diffs.append(n["metrics"]["LogLoss"] - r["metrics"]["LogLoss"])
mean_new = sum(new[u]["metrics"]["LogLoss"] for u in common) / len(common)
mean_ref = sum(ref[u]["metrics"]["LogLoss"] for u in common) / len(common)
print(f"== {name}  vs  {ref_path} ==")
print(f"  users compared : {len(common)} (new {len(new)}, ref {len(ref)})")
print(f"  size exact     : {size_ok}  (sum new {sum_new}, sum ref {sum_ref})")
print(f"  mean LogLoss   : new {mean_new:.6f}  ref {mean_ref:.6f}")
print(f"  mean dLogLoss  : {mean_new-mean_ref:+.6f}   (gate: |d| <= 0.0005)")
print(f"  max |dLogLoss| : {max(abs(x) for x in diffs):.6f}")
print(f"  RESULT         : {'PASS' if size_ok and abs(mean_new-mean_ref) <= 0.0005 else 'FAIL'}")
