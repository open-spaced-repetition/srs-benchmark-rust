"""Copy the research result files into the tracked archive (`result/` is gitignored).

Mirrors `_smart/archive_results.py`: the per-user `parameters` array is dropped (it is ~90% of the
bytes and nothing downstream reads it), keeping metrics/user/size/time_ms. The `--hp_probe` and
`--hp_features` dumps have no `parameters` field, so they are gzipped instead of stripped.
"""
import gzip, json, os, shutil

SRC, DST = "result", "_hpprobe/results"
STRIP = ["FSRS-7-short-secs-recency-regrow0.003", "FSRS-7-short-secs-recency-regrow0.05"]
GZIP = ["FSRS-7-short-secs-recency-hpprobe", "FSRS-7-short-secs-recency-hpfeat"]

os.makedirs(DST, exist_ok=True)
for name in STRIP:
    src, dst = f"{SRC}/{name}.jsonl", f"{DST}/{name}.jsonl"
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.pop("parameters", None)
            o.write(json.dumps(d, separators=(", ", ": ")) + "\n")
            n += 1
    print(f"stripped {n:>6} users  {src} -> {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")

for name in GZIP:
    src, dst = f"{SRC}/{name}.jsonl", f"{DST}/{name}.jsonl.gz"
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=9) as o:
        shutil.copyfileobj(f, o)
    print(f"gzipped               {src} -> {dst}  "
          f"({os.path.getsize(src)/1e6:.1f} -> {os.path.getsize(dst)/1e6:.1f} MB)")
