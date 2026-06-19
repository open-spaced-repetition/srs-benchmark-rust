"""Phase-2 speed test: Wilcoxon signed-rank on paired per-user time_ms (before vs after),
measured SIMULTANEOUSLY (1 thread each). Accept a speedup iff p < 0.01 AND the after is
faster (median ratio < 1).

Usage: python _phase2/wilcoxon_time.py <before.jsonl> <after.jsonl>
"""
import json
import sys

from scipy.stats import wilcoxon


def load_time(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        d[o["user"]] = o["time_ms"]
    return d


def main():
    before = load_time(sys.argv[1])
    after = load_time(sys.argv[2])
    users = sorted(set(before) & set(after))
    b = [before[u] for u in users]
    a = [after[u] for u in users]
    tot_b, tot_a = sum(b), sum(a)
    ratios = sorted(ai / bi for ai, bi in zip(a, b) if bi > 0)
    median_ratio = ratios[len(ratios) // 2]
    # two-sided Wilcoxon; report one-sided "after < before" too.
    stat, p_two = wilcoxon(b, a)  # H0: before==after
    stat_l, p_less = wilcoxon(a, b, alternative="less")  # H1: after < before (faster)
    print(f"  users paired      : {len(users)}")
    print(f"  total time_ms     : before {tot_b:.0f}  after {tot_a:.0f}  (x{tot_b/tot_a:.2f} faster)")
    print(f"  median per-user ratio (after/before): {median_ratio:.4f}  (x{1/median_ratio:.2f} faster)")
    print(f"  Wilcoxon two-sided p = {p_two:.3e}")
    print(f"  Wilcoxon one-sided (after<before) p = {p_less:.3e}")
    accept = p_less < 0.01 and median_ratio < 1.0
    print(f"  RESULT            : {'ACCEPT (p<0.01, faster)' if accept else 'REJECT'}")
    return 0 if accept else 1


if __name__ == "__main__":
    sys.exit(main())
