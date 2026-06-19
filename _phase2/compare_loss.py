"""Phase-2 correctness check: compare a candidate FSRS-7 result jsonl against the FROZEN
baseline (_fsrs7_baseline/). Verifies size is exact (per-user + sum) and reports the mean
ΔLogLoss. Gate: |mean_new - mean_base| <= 0.0005 (within ±0.0005 of the frozen original).

Usage: python _phase2/compare_loss.py <name>
  where <name> is e.g. FSRS-7-short-secs-recency (compares result/<name>.jsonl vs
  _fsrs7_baseline/<name>.jsonl).
"""
import json
import sys


def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        d[o["user"]] = o
    return d


def main():
    name = sys.argv[1]
    base = load(f"_fsrs7_baseline/{name}.jsonl")
    new = load(f"result/{name}.jsonl")
    common = sorted(set(base) & set(new))
    size_ok = True
    sum_base = sum_new = 0
    diffs = []
    for u in common:
        b, n = base[u], new[u]
        sum_base += b["size"]
        sum_new += n["size"]
        if b["size"] != n["size"]:
            size_ok = False
            print(f"  SIZE MISMATCH user {u}: base {b['size']} new {n['size']}")
        diffs.append(n["metrics"]["LogLoss"] - b["metrics"]["LogLoss"])
    mean_base = sum(base[u]["metrics"]["LogLoss"] for u in common) / len(common)
    mean_new = sum(new[u]["metrics"]["LogLoss"] for u in common) / len(common)
    mean_d = mean_new - mean_base
    print(f"== {name} ==")
    print(f"  users compared : {len(common)} (base {len(base)}, new {len(new)})")
    print(f"  size exact     : {size_ok}  (sum base {sum_base}, sum new {sum_new})")
    print(f"  mean LogLoss   : base {mean_base:.6f}  new {mean_new:.6f}")
    print(f"  mean dLogLoss  : {mean_d:+.6f}   (gate: |d| <= 0.0005)")
    print(f"  max |dLogLoss| : {max(abs(x) for x in diffs):.6f}")
    gate = size_ok and abs(mean_d) <= 0.0005
    print(f"  RESULT         : {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
