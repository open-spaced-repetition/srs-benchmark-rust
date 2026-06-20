#!/usr/bin/env bash
# Phase-2 SIMULTANEOUS timing: baseline (forward-mode) vs candidate (analytic gradient),
# 1 thread each (= 2 threads total), same 200 users, same config, started together so
# thermal/scheduling noise hits both equally. Each writes to its own result/ dir.
set -e
REPO="C:/Users/Andrew/srs-benchmark-rust"
DATA="C:/Users/Andrew/anki-revlogs-10k"
BASE_EXE="$REPO/target/release/script_baseline.exe"
CAND_EXE="$REPO/target/release/script_analytic.exe"
CFG="$1"            # e.g. "--algo FSRS-7 --short --secs"
NAME="$2"           # e.g. "FSRS-7-short-secs"
NUSERS="${3:-200}"

mkdir -p "$REPO/_phase2/time_base/result" "$REPO/_phase2/time_cand/result"
rm -f "$REPO/_phase2/time_base/result/$NAME.jsonl" "$REPO/_phase2/time_cand/result/$NAME.jsonl"

( cd "$REPO/_phase2/time_base" && "$BASE_EXE" $CFG --data "$DATA" --max-user-id "$NUSERS" --processes 1 ) &
PID_BASE=$!
( cd "$REPO/_phase2/time_cand" && "$CAND_EXE" $CFG --data "$DATA" --max-user-id "$NUSERS" --processes 1 ) &
PID_CAND=$!
wait $PID_BASE
wait $PID_CAND
echo "=== TIMING RUN DONE: $NAME ==="
