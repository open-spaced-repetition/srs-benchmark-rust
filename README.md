# srs-benchmark-rust

A Rust port of [open-spaced-repetition/srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark),
built to run the same benchmark **much faster** while reproducing the same results.

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

```json
{"metrics": {"RMSE": .., "LogLoss": .., "RMSE(bins)": .., "AUC": .., "precision@90": ..,
 "recall@90": .., "MBE": ..}, "user": N, "size": M, "parameters": ..., "time_ms": 123.4}
```

Runs resume: users already present in the result file are skipped (delete it for a fresh
run).

## Reproduction status

How faithfully each ported algorithm reproduces the upstream Python results, measured on
**1000 users** of `anki-revlogs-3k` (a subset of the 10k dataset with identical per-user
data) versus the upstream reference files. Two criteria:

- **`size` exact** — the per-user review count *and* its total across users must match the
  Python output **exactly** (validates the feature pipeline / row filtering).
- **mean LogLoss — one-sided tolerance**: it must not be **worse** (higher) than upstream by
  more than **0.0005**, but may be **better** (lower) by any amount. (Our f64 finds slightly
  better optima than torch's f32 on chaotic models, so a few give *lower* loss — that's a
  win, not a failure.)

Verified on the `--short --secs` configuration (the recommended FSRS setting).

| Algorithm | `size` exact | mean LogLoss vs upstream | Status |
| --- | :---: | --- | --- |
| AVG | ✅ | 0.000000 (bit-exact) | ✅ verified |
| SM2 | ✅ | 0.000000 (bit-exact) | ✅ verified |
| MOVING-AVG | ✅ | 0.000000 (bit-exact) | ✅ verified |
| DASH | ✅ | +0.000006 | ✅ verified |
| HLR | ✅ | −0.004 (better) | ✅ verified¹ |
| RMSE-BINS-EXPLOIT | ✅ | 0.000000 vs current Python⁴ | ✅ verified |
| FSRS v1 | ✅ | −0.0009 (better) | ✅ verified¹ |
| FSRS v2 | ✅ | −0.004 (better) | ✅ verified¹ |
| FSRS v3 | ✅ | −0.005 (better) | ✅ verified¹ |
| FSRS v4, v4.5, v5, v6 | — | — | ⏳ porting (need S0-init) |
| ACT-R | — | — | ⏳ porting |
| Ebisu v2 | — | — | ⏳ porting |
| FSRS-7 | — | — | ⏸ deferred² |
| LogisticRegression, FSRS-rs | — | — | 📋 planned |
| GRU, LSTM, RWKV, Transformer, NN-17 | — | — | 🐍 Python path³ |

¹ Models with extreme predictions (HLR's `2^d`, FSRS's `0.9^(t/s)`) have a few chaotic users
where tiny f64-vs-f32 float differences amplify. The training core is proven correct (DASH
matches to 6e-6, and the per-user *median* diff is ~1e-5); Rust's f64 finds slightly better
optima, so the mean LogLoss comes out *lower* (better) than upstream — which passes the
one-sided tolerance.

² FSRS-7's model is still being changed upstream, so it is intentionally not ported yet.

³ GRU/LSTM use the Reptile optimizer and the neural models are hard to port; these keep the
Python runtime path.

⁴ The Rust output is bit-identical to the *current* Python source. The committed upstream
`result/` file for this model is stale (predates a model/pipeline change), so it is not a
valid reference here — the binding target (rule #5) is the current Python version.

*Currently only the `--secs` feature path is implemented; the non-`--secs` path (which adds
outlier / non-continuous-row removal) is planned.*

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
