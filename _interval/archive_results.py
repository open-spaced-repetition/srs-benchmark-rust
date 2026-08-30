"""Copy the interval-definition result files into the tracked archive (`result/` is gitignored).

Same convention as `_smart/` and `_hpprobe/`: the per-user `parameters` array is dropped (~90% of
the bytes, nothing downstream reads it), keeping metrics/user/size/time_ms.

`-id` runs live under `_idrun/` because the output filename carries no dataset name, so a base run
and an `-id` run of the same config would otherwise collide.
"""
import json, os

DST = "_interval/results"
SRC = {
    "base-stored":  "result/FSRS-7-short-secs-recency.jsonl",
    "id-stored":    "_idrun/result/FSRS-7-short-secs-recency.jsonl",
    "id-e2e":       "_idrun/result/FSRS-7-short-secs-recency-e2e.jsonl",
    "id-e2s":       "_idrun/result/FSRS-7-short-secs-recency-e2s.jsonl",
    "id-e2e-min1s": "_idrun/result/FSRS-7-short-secs-recency-e2e-min1s.jsonl",
    "id-e2s-min1s": "_idrun/result/FSRS-7-short-secs-recency-e2s-min1s.jsonl",
}

os.makedirs(DST, exist_ok=True)
for name, src in SRC.items():
    dst = f"{DST}/FSRS-7-short-secs-recency--{name}.jsonl"
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
    print(f"{n:>6} users  {src} -> {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")
