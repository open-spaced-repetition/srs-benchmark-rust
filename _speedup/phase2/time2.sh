#!/usr/bin/env bash
# Parametrized SIMULTANEOUS timing: two arbitrary binaries, 1 thread each (2 total), same
# users/config, started together so thermal/scheduling noise hits both equally. Each writes
# to its own result dir. Usage: time2.sh <before_exe> <after_exe> "<cfg>" <name> [nusers]
set -e
REPO="C:/Users/Andrew/srs-benchmark-rust"
DATA="C:/Users/Andrew/anki-revlogs-10k"
BEFORE_EXE="$1"; AFTER_EXE="$2"; CFG="$3"; NAME="$4"; NUSERS="${5:-200}"
mkdir -p "$REPO/_phase2/t_before/result" "$REPO/_phase2/t_after/result"
rm -f "$REPO/_phase2/t_before/result/$NAME.jsonl" "$REPO/_phase2/t_after/result/$NAME.jsonl"
( cd "$REPO/_phase2/t_before" && "$BEFORE_EXE" $CFG --data "$DATA" --max-user-id "$NUSERS" --processes 1 ) &
P1=$!
( cd "$REPO/_phase2/t_after" && "$AFTER_EXE" $CFG --data "$DATA" --max-user-id "$NUSERS" --processes 1 ) &
P2=$!
wait $P1; wait $P2
echo "=== TIMING DONE: $NAME ==="
