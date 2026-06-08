# srs-benchmark-rust

A Rust port of [open-spaced-repetition/srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark),
built to run the same benchmark **much faster** while ensuring that results don't become significantly worse.

The command-line interface mirrors the Python `script.py` (same flags, same output
filenames). Model *definitions* remain authored in Python upstream as the canonical spec;
the math for the ported algorithms is reimplemented natively in Rust for speed. Algorithms
that rely on the Reptile optimizer (GRU, LSTM) and other neural models keep the Python
runtime path for now.

## Build

```bash
cargo build --release          # binary: target/release/script
```

## Run

```bash
# binary is named `script` to match the Python entry point
target/release/script --algo DASH --short --secs --data ../anki-revlogs-10k --processes 16
```

Output is written to `result/<name>.jsonl`, one JSON object per user (sorted by user id),
matching the Python `evaluate()` schema plus a per-user `time_ms` field (milliseconds) so
slow users can be found:

```
{"metrics": {"RMSE": .., "LogLoss": .., "RMSE(bins)": .., "AUC": .., "precision@90": ..,
 "recall@90": .., "MBE": ..}, "user": N, "size": M, "parameters": ..., "time_ms": 123.4}
```

Runs resume: users already present in the result file are skipped (delete it for a fresh
run).

## Reproduction status

Every **(algorithm + flags)** configuration that upstream publishes a reference for is listed
below — one row per config — measured on the first **1000 users** of `anki-revlogs-10k`. Two
criteria:

- **`size` exact** — the per-user review count *and* its total across users must match the
  Python output **exactly** (validates the feature pipeline / row filtering).
- **mean LogLoss — one-sided tolerance** — it must not be **worse** (higher) than upstream by
  more than **0.0005**, but may be **better** (lower) by any amount. `(better)` marks configs
  where the Rust port scores a lower loss than upstream.

### Verified — 58 configurations

| Configuration | `size` | mean LogLoss vs upstream | Status |
| --- | :---: | --- | --- |
| `AVG` | ✅ | +0.000000 | ✅ verified |
| `AVG --secs` | ✅ | +0.000000 ⁴ | ✅ verified |
| `AVG --short --secs` | ✅ | +0.000000 | ✅ verified |
| `SM2` | ✅ | +0.000000 | ✅ verified |
| `SM2 --short` | ✅ | +0.000000 | ✅ verified |
| `SM2 --short --secs` | ✅ | +0.000000 | ✅ verified |
| `SM2-trainable` | ✅ | +0.000205 | ✅ verified |
| `SM2-trainable --short --secs` | ✅ | -0.000466 | ✅ verified |
| `MOVING-AVG` | ✅ | +0.000000 | ✅ verified |
| `MOVING-AVG --short --secs` | ✅ | +0.000000 | ✅ verified |
| `RMSE-BINS-EXPLOIT` | ✅ | +0.000000 | ✅ verified |
| `RMSE-BINS-EXPLOIT --short --secs` | ✅ | -0.019035 (better) ⁴ | ✅ verified |
| `Ebisu-v2` | ✅ | +0.000000 | ✅ verified |
| `Ebisu-v2 --short --secs` | ✅ | +0.000000 | ✅ verified |
| `Anki` | ✅ | +0.000027 | ✅ verified |
| `Anki --default` | ✅ | +0.000000 | ✅ verified |
| `Anki --short --secs` | ✅ | -0.000142 | ✅ verified |
| `DASH` | ✅ | +0.000000 | ✅ verified |
| `DASH --secs` | ✅ | +0.000000 ⁴ | ✅ verified |
| `DASH --short` | ✅ | +0.000155 | ✅ verified |
| `DASH --short --secs` | ✅ | -0.000006 | ✅ verified |
| `DASH --recency` | ✅ | -0.001471 (better) | ✅ verified |
| `DASH[MCM]` | ✅ | -0.000114 | ✅ verified |
| `DASH[MCM] --secs` | ✅ | +0.000000 ⁴ | ✅ verified |
| `DASH[MCM] --short --secs` | ✅ | -0.000001 | ✅ verified |
| `DASH[ACT-R]` | ✅ | +0.000001 | ✅ verified |
| `DASH[ACT-R] --secs` | ✅ | -0.000000 ⁴ | ✅ verified |
| `DASH[ACT-R] --short --secs` | ✅ | -0.000051 | ✅ verified |
| `HLR` | ✅ | -0.000556 (better) | ✅ verified |
| `HLR --short` | ✅ | -0.001039 (better) | ✅ verified |
| `HLR --short --secs` | ✅ | -0.005829 (better) | ✅ verified |
| `ACT-R` | ✅ | -0.008047 (better) | ✅ verified ⁵ |
| `ACT-R --secs` | ✅ | -0.011462 (better) ⁴ | ✅ verified ⁵ |
| `ACT-R --short --secs` | ✅ | -0.001420 (better) | ✅ verified ⁵ |
| `FSRSv1` | ✅ | +0.000445 | ✅ verified |
| `FSRSv1 --short --secs` | ✅ | -0.000238 | ✅ verified |
| `FSRSv2` | ✅ | -0.000368 | ✅ verified |
| `FSRSv2 --short --secs` | ✅ | -0.000303 | ✅ verified |
| `FSRSv3` | ✅ | -0.000186 | ✅ verified |
| `FSRSv3 --short --secs` | ✅ | -0.000119 | ✅ verified |
| `FSRSv4` | ✅ | -0.000523 (better) | ✅ verified |
| `FSRSv4 --short --secs` | ✅ | -0.000353 | ✅ verified |
| `FSRS-4.5` | ✅ | -0.000312 | ✅ verified |
| `FSRS-4.5 --short --secs` | ✅ | +0.000250 | ✅ verified |
| `FSRS-5 --short` | ✅ | +0.000001 | ✅ verified |
| `FSRS-5 --short --secs` | ✅ | +0.000046 | ✅ verified |
| `FSRS-6 --short` | ✅ | -0.000008 | ✅ verified |
| `FSRS-6 --short --secs` | ✅ | -0.000142 | ✅ verified |
| `FSRS-6 --default --short` | ✅ | -0.000000 | ✅ verified |
| `FSRS-6 --default --short --secs` | ✅ | -0.000001 | ✅ verified |
| `FSRS-6 --S0 --short` | ✅ | -0.000007 | ✅ verified |
| `FSRS-6 --S0 --short --secs` | ✅ | +0.000069 | ✅ verified |
| `FSRS-6 --two_buttons --short` | ✅ | +0.000003 | ✅ verified |
| `FSRS-6 --two_buttons --short --secs` | ✅ | +0.000168 | ✅ verified |
| `FSRS-6 --recency` | ✅ | -0.000004 | ✅ verified |
| `FSRS-6 --short --recency` | ✅ | -0.000006 | ✅ verified |
| `FSRS-6 --short --secs --recency` | ✅ | +0.000127 | ✅ verified |
| `FSRS-6 --short --recency --train_equals_test` | ✅ | +0.000430 | ✅ verified |

### Not yet reproduced — 31 configurations

| Configuration(s) | Status |
| --- | --- |
| `FSRS-6 --partitions deck` / `--partitions preset` (3) | 📋 planned — needs deck/preset partitioning |
| LogisticRegression (2), FSRS-6-one-step (1), FSRS-rs (1) | 📋 planned — model ports (FSRS-rs imports the `fsrs` crate) |
| FSRS-7 (10 flag variants) | ⏸ deferred — upstream model still WIP |
| GRU, LSTM, RWKV, RWKV-P, NN-17, Transformer (14) | 🐍 Python path — Reptile/neural, kept in Python |

⁴ The committed upstream file for this config is **stale** (predates a pipeline change), so it
is not a valid reference — the binding target (rule #5) is the *current* Python source, which
the Rust output matches. `-secs` configs are verified against a freshly-generated current-
Python golden (spot-checked on 15 users); everything else is on 1000 users.

⁵ ACT-R is correct but slow — its activation is an O(reviews²) all-pairs sum over prior
reviews, a target for the planned performance pass.

*Both the `--secs` and non-`--secs` feature paths are implemented; the non-`--secs` path
reproduces the upstream outlier / non-continuous-row removal exactly, so `size` matches
bit-for-bit.*

## Options

All flags match the Python `script.py`
([upstream docs](https://github.com/open-spaced-repetition/srs-benchmark#scriptpy-options)).

| Flag | Description | Default |
| --- | --- | --- |
| `--algo` | Algorithm name (e.g. `FSRS-6`, `DASH`, `HLR`, `SM2`, `AVG`). | `FSRSv3` |
| `--data` | Path to the dataset root (containing `revlogs/`, `cards/`, `decks/`). | `../anki-revlogs-10k` |
| `--processes` | Number of parallel worker threads (Python: processes). | `8` |
| `--max-user-id` | Only process users with id ≤ this (inclusive). | no limit |
| `--short` | Include short-term (same-day) reviews. | off |
| `--secs` | Use `elapsed_seconds` (fractional-day intervals) instead of `elapsed_days`. | off |
| `--default` | Evaluate default parameters (no training). | off |
| `--recency` | Weight training reviews by recency (`0.25 + 0.75·x³`). | off |
| `--S0` | FSRS-5/6: optimize only the initial-stability parameters. | off |
| `--sched_penalties` | FSRS-7 scheduling penalties (penalty 1 & 2). | off |
| `--two_buttons` | Treat Hard and Easy as Good (rating remap). | off |
| `--partitions` | Train per partition: `none`, `deck`, or `preset`. | `none` |
| `--n_splits` | Number of `TimeSeriesSplit` folds. | `5` |
| `--batch_size` | Training batch size. | `512` |
| `--max_seq_len` | Max sequence length for batching (also caps reviews/card at `2×`). | `64` |
| `--train_equals_test` | Train and test on the same data (overfit probe). | off |
| `--no_test_same_day` | Exclude `elapsed_days=0` reviews from the test set. | off |
| `--no_train_same_day` | Exclude `elapsed_days=0` reviews from the train set. | off |
| `--equalize_test_with_non_secs` | Test only on reviews that the non-`--secs` run would test. | off |
| `--duration` | Add the review-duration feature (LSTM only). | off |
| `--raw` | Save raw predictions to `raw/<name>.jsonl`. | off |
| `--file` | Save per-user evaluation TSVs to `evaluation/<name>/`. | off |
| `--plot` | Save evaluation plots. | off |
| `--weights` | Save trained model weights. | off |
| `--gpus` | CUDA device ids (e.g. `0,1` or `all`); unused by the CPU models. | unset |
| `--torch_num_threads` | PyTorch intra-op threads (parity flag). | `1` |
| `--dev` | Local-development import mode. | off |

The output filename is derived from the flags exactly as in Python — e.g.
`--algo FSRS-6 --short --secs` → `result/FSRS-6-short-secs.jsonl`.

## Status

Work in progress — see `CLAUDE.md` for the architecture, phase plan, and current status.
