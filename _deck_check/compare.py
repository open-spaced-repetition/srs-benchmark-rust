"""Compare a Rust result jsonl against a reference jsonl on the common users.
Reports per-user size match (EXACT) and LogLoss diff (rust - ref), plus partition-key match.
Usage: python _deck_check/compare.py <rust.jsonl> <ref.jsonl>
"""
import json, sys

def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        v = json.loads(line)
        if "metrics" not in v:   # skip stubs
            continue
        out[v["user"]] = v
    return out

rust = load(sys.argv[1])
ref = load(sys.argv[2])
common = sorted(set(rust) & set(ref))
print(f"rust users: {len(rust)}  ref users: {len(ref)}  common: {len(common)}")
print(f"{'user':>6} {'size_r':>9} {'size_ref':>9} {'sz_ok':>5} {'LL_rust':>9} {'LL_ref':>9} {'dLL':>10} {'nP_r':>5} {'nP_ref':>6} {'keys_ok':>7}")
diffs = []
sz_fail = 0
for u in common:
    r, f = rust[u], ref[u]
    sok = r["size"] == f["size"]
    if not sok:
        sz_fail += 1
    llr = r["metrics"]["LogLoss"]
    llf = f["metrics"]["LogLoss"]
    d = llr - llf
    diffs.append(d)
    kr = set(r["parameters"].keys())
    kf = set(f["parameters"].keys())
    kok = kr == kf
    print(f"{u:>6} {r['size']:>9} {f['size']:>9} {str(sok):>5} {llr:>9.6f} {llf:>9.6f} {d:>+10.6f} {len(kr):>5} {len(kf):>6} {str(kok):>7}")
if diffs:
    mean = sum(diffs) / len(diffs)
    print(f"\nmean dLL (rust-ref): {mean:+.6f}   max|dLL|: {max(abs(x) for x in diffs):.6f}   size mismatches: {sz_fail}")
    print("PASS" if abs(mean) <= 0.0005 and sz_fail == 0 else "FAIL")
