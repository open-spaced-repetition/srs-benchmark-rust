#!/bin/bash
# 30-experiment hierarchical smart-preset sweep on 1000 users, 100-user chunks (checkpoints),
# 2 threads. Uses the snapshot binary so HDBSCAN rebuilds of script.exe don't disturb it.
cd /c/Users/Andrew/srs-benchmark-rust || exit 1
for M in 100 200 300 400 500 600 700 800 900 1000; do
  echo "=== chunk max-user-id=$M @ $(date +%H:%M:%S) ==="
  ./target/release/script_sweep.exe --algo FSRS-7 --short --secs --partitions smart --cluster_sweep \
    --data "C:\\Users\\Andrew\\anki-revlogs-10k" --processes 2 --max-user-id "$M"
done
echo "=== ALL DONE @ $(date +%H:%M:%S) ==="
