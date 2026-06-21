"""Archive the smart-preset clustering results (metrics only) for committing.

The live result/*.jsonl carry per-cluster `parameters` (weights) and are large (~hundreds of MB). For
the repo we keep just the results: metrics + user + size + time_ms (drop `parameters`). Writes the
clustering experiments + the baseline into _smart/results/.

By default this MOVES the experiment files: after each `-smart-*` file is archived it is deleted from
result/ (the stripped archive is the kept copy; the full result/ file is regenerable). The baseline
FSRS-7-short-secs.jsonl is archived but NEVER deleted (it's the live reference, also extended by the
10k run). Pass --no-delete to copy without deleting. Only run after the sweeps writing these files
have finished.
"""
import argparse
import glob
import json
import os

SRC = "result"
DST = "_smart/results"
KEEP = ("metrics", "user", "size", "time_ms")
BASELINE = "FSRS-7-short-secs.jsonl"


def strip_archive(src, dst):
    """Write src -> dst keeping only the KEEP keys per line; return dst byte size."""
    with open(dst, "w") as out:
        for line in open(src):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.write(json.dumps({k: o[k] for k in KEEP if k in o}) + "\n")
    return os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser(description="Archive smart-preset results (metrics only) for committing.")
    ap.add_argument(
        "--no-delete",
        action="store_true",
        help="copy only; keep the full originals in result/ (default: move = delete after archiving)",
    )
    args = ap.parse_args()

    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(f"{SRC}/FSRS-7-short-secs-smart-*.jsonl")) + [f"{SRC}/{BASELINE}"]
    archived, total, deleted = 0, 0, 0
    for src in files:
        if not os.path.exists(src):
            continue
        dst = f"{DST}/{os.path.basename(src)}"
        total += strip_archive(src, dst)  # raises before any delete if the archive write fails
        archived += 1
        # Move semantics: delete the full original after a successful archive — but NEVER the
        # baseline (it's the live reference).
        if not args.no_delete and os.path.basename(src) != BASELINE:
            os.remove(src)
            deleted += 1
    mode = "kept originals (--no-delete)" if args.no_delete else f"deleted {deleted} originals from {SRC}/"
    print(f"archived {archived} files to {DST}/ ({total / 1e6:.1f} MB); {mode}")


if __name__ == "__main__":
    main()
