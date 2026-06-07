# srs-benchmark-rust

A Rust port of [open-spaced-repetition/srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark),
built to run the same benchmark **much faster** while reproducing the same results.

The command-line interface mirrors the Python `script.py` (same flags). Model *definitions*
remain authored in Python upstream as the canonical spec; the math for the ported
algorithms is reimplemented natively in Rust for speed. Algorithms that rely on the Reptile
optimizer (GRU, LSTM) and other neural models keep the Python runtime path for now.

## Build

```bash
cargo build --release
```

## Run

```bash
# binary is named `script` to match the Python entry point
target/release/script --algo FSRS-7 --short --secs --data ../anki-revlogs-10k --processes 16
```

See `--help` for the full flag list. Output is written to `result/<name>.jsonl`, one JSON
object per user, including a per-user processing time.

## Correctness target

A ported model is accepted when its unweighted mean LogLoss across 10k users is within
±0.0005 of the original Python result.

## Status

Work in progress — see `CLAUDE.md` for the architecture, phase plan, and current status.
