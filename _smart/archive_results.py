"""Archive the smart-preset clustering results (metrics only) for committing.

The live result/*.jsonl carry per-cluster `parameters` (weights) and total ~249 MB. For the repo
we keep just the results: metrics + user + size + time_ms (drop `parameters`). Writes the 46
clustering experiments + the baseline into _smart/results/ (~10 MB).
"""
import json, os, glob

SRC = "result"
DST = "_smart/results"
KEEP = ("metrics", "user", "size", "time_ms")

os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(f"{SRC}/FSRS-7-short-secs-smart-*.jsonl")) + [f"{SRC}/FSRS-7-short-secs.jsonl"]
total = 0
for src in files:
    if not os.path.exists(src):
        continue
    dst = f"{DST}/{os.path.basename(src)}"
    with open(dst, "w") as out:
        for line in open(src):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.write(json.dumps({k: o[k] for k in KEEP if k in o}) + "\n")
    total += os.path.getsize(dst)
print(f"wrote {len(files)} files to {DST}/, total {total/1e6:.1f} MB")
