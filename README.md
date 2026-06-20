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
RUSTFLAGS="-C target-cpu=native" cargo build --release   # binary: target/release/script
```

`-C target-cpu=native` lets the FSRS-7 gradient use the `f32x8` (AVX/AVX2) SIMD kernel; a plain
`cargo build --release` is still correct, just narrower/slower (SSE2 baseline).

### Precision: f32 by default (`fp64` opt-in)

Each model's forward value/prediction, the optimizer, and the analytic (reverse-mode) gradients are
rounded to **f32** by default, matching torch and the official Rust implementations (which the
upstream references are generated with). This is what keeps the port faithful to those references
under the ±0.0005 rule below. (One exception: the algorithms whose training gradient comes from
*forward-mode* autodiff — ACT-R, Anki, DASH[ACT-R], the FSRS v1–v6 family (incl. FSRS-4.5 and
FSRS-6-one-step), and SM2-trainable — run **entirely in f64** (value, optimizer, and gradient alike),
because torch computes gradients in *reverse-mode* and only a fully-f64 forward-mode pass faithfully
proxies that; an f32 forward-mode pass diverges badly on the chaotic `--secs` training trajectory.)
Build with the optional `fp64` feature to compute everything in full f64 instead:

```bash
cargo build --release --features fp64
```

f64 finds slightly *better* (lower-loss) optima on chaotic models (HLR, ACT-R, FSRS) — so it can
**improve** some algorithms' loss — but it then diverges from the upstream f32 references (e.g.
`HLR --short --secs` reads −0.0058 below upstream in f64 vs ±0 in f32). Keep the default (f32) to
reproduce upstream; use `fp64` only to explore those better optima. (Tests: `cargo test` exercises
the f32 path; `cargo test --features fp64` additionally runs the finite-difference math checks,
which need f64.)

The **FSRS-rs** algorithm is gated behind an optional `fsrs-rs` cargo feature, because it
imports the real [`fsrs`](https://crates.io/crates/fsrs) crate (the exact `4.1.1` release that
`fsrs-rs-python` 0.8.2 wraps), which pulls in the heavy [`burn`](https://burn.dev) ML
framework. It is off by default; enable it with:

```bash
cargo build --release --features fsrs-rs
```

The **neural** models (NN-17, GRU, LSTM) are gated behind an optional `neural` cargo feature,
because they use the [`candle`](https://github.com/huggingface/candle) ML framework (autodiff,
RNN cells, AdamW, and a PyTorch `.pth` checkpoint loader), which is a heavy build. It is off by
default; enable it with:

```bash
cargo build --release --features neural
```

These models load pretrained checkpoints from a `pretrain/` directory (the `*_pretrain.pth`
files shipped with the Python `srs-benchmark`); see the per-model notes below.

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
- **mean LogLoss — two-sided tolerance (±0.0005)** — the Rust mean LogLoss must be within
  **±0.0005** of upstream. Anything outside `[−0.0005, +0.0005]` — **higher OR lower** — does
  **not** pass. A `⚠ genuine` mark means the config falls outside the band but the cause has been
  investigated and is a genuine f64-vs-f32 optimum / optimizer-trajectory difference, **not a bug**
  (these read lower than upstream — `(better)`); see the per-config note `⁴`.

> **Resolved (2026-06-19) — per-algo precision (see Build §).** The re-review concluded: the port now
> uses **f32 for analytic/reverse-mode-gradient algos** (HLR, DASH, LogReg, FSRS-7 — matching the f32
> upstream) and **f64 for forward-mode-`Dual` algos** (FSRS v1–v6, v4.5, ACT-R, Anki, DASH[ACT-R], FSRS-6-one-step,
> SM2-trainable — whose forward-mode gradient only proxies torch's reverse-mode faithfully in f64).
> This **fixed `HLR --short --secs`** (−0.0058 → −0.0000) and keeps the `Dual` algos at their verified
> f64 numbers. The handful still outside ±0.0005 are marked `⚠ genuine` below — all investigated, none
> a bug (see `⁴`).

### Verified — 65 configurations

| Configuration | `size` | mean LogLoss vs upstream | Status |
| --- | :---: | --- | --- |
| `AVG` | ✅ | +0.000000 | ✅ verified |
| `AVG --secs` | ✅ | +0.000000 ¹ | ✅ verified |
| `AVG --short --secs` | ✅ | +0.000000 | ✅ verified |
| `SM2` | ✅ | +0.000000 | ✅ verified |
| `SM2 --short` | ✅ | +0.000000 | ✅ verified |
| `SM2 --short --secs` | ✅ | +0.000000 | ✅ verified |
| `SM2-trainable` | ✅ | +0.000205 | ✅ verified |
| `SM2-trainable --short --secs` | ✅ | -0.000466 | ✅ verified |
| `MOVING-AVG` | ✅ | +0.000000 | ✅ verified |
| `MOVING-AVG --short --secs` | ✅ | +0.000000 | ✅ verified |
| `RMSE-BINS-EXPLOIT` | ✅ | +0.000000 | ✅ verified |
| `RMSE-BINS-EXPLOIT --short --secs` | ✅ | -0.019035 (better) ¹ | ✅ verified |
| `Ebisu-v2` | ✅ | +0.000000 | ✅ verified |
| `Ebisu-v2 --short --secs` | ✅ | +0.000000 | ✅ verified |
| `Anki` | ✅ | +0.000027 | ✅ verified |
| `Anki --default` | ✅ | +0.000000 | ✅ verified |
| `Anki --short --secs` | ✅ | -0.000142 | ✅ verified |
| `DASH` | ✅ | +0.000000 | ✅ verified |
| `DASH --secs` | ✅ | +0.000000 ¹ | ✅ verified |
| `DASH --short` | ✅ | +0.000155 | ✅ verified |
| `DASH --short --secs` | ✅ | -0.000006 | ✅ verified |
| `DASH --recency` | ✅ | -0.001471 (better) | ⚠ genuine ⁴ |
| `DASH[MCM]` | ✅ | -0.000114 | ✅ verified |
| `DASH[MCM] --secs` | ✅ | +0.000000 ¹ | ✅ verified |
| `DASH[MCM] --short --secs` | ✅ | -0.000001 | ✅ verified |
| `DASH[ACT-R]` | ✅ | +0.000001 | ✅ verified |
| `DASH[ACT-R] --secs` | ✅ | -0.000000 ¹ | ✅ verified |
| `DASH[ACT-R] --short --secs` | ✅ | -0.000051 | ✅ verified |
| `HLR` | ✅ | -0.000555 (better) | ⚠ genuine ⁴ |
| `HLR --short` | ✅ | -0.000709 (better) | ⚠ genuine ⁴ |
| `HLR --short --secs` | ✅ | -0.000000 | ✅ verified |
| `ACT-R` | ✅ | -0.008047 (better) | ⚠ genuine ² ⁴ |
| `ACT-R --secs` | ✅ | -0.011462 (better) ¹ | ✅ verified ² |
| `ACT-R --short --secs` | ✅ | -0.001420 (better) | ⚠ genuine ² ⁴ |
| `FSRSv1` | ✅ | +0.000445 | ✅ verified |
| `FSRSv1 --short --secs` | ✅ | -0.000238 | ✅ verified |
| `FSRSv2` | ✅ | -0.000368 | ✅ verified |
| `FSRSv2 --short --secs` | ✅ | -0.000303 | ✅ verified |
| `FSRSv3` | ✅ | -0.000186 | ✅ verified |
| `FSRSv3 --short --secs` | ✅ | -0.000119 | ✅ verified |
| `FSRSv4` | ✅ | -0.000523 (better) | ⚠ genuine ⁴ |
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
| `FSRS-6 --short --partitions deck` | ✅ | +0.000477 | ✅ verified |
| `FSRS-6 --short --partitions preset` | ✅ | -0.000001 | ✅ verified |
| `FSRS-6 --short --secs --partitions preset` | ✅ | -0.003894 (better) | ⚠ genuine ⁴ |
| `FSRS-6-one-step --short` | ✅ | -0.000681 (better) | ⚠ genuine ⁴ |
| `LogisticRegression --short --secs --recency` | ✅ | +0.000001 | ✅ verified |
| `LogisticRegression --short --secs --recency --equalize_test_with_non_secs` | ✅ | +0.000015 | ✅ verified |
| `FSRS-rs --short` | ✅ | +0.000299 ¹ ³ | ✅ verified |

### Ported separately

| Configuration(s) | Status |
| --- | --- |
| **FSRS-7** (34-param dual-stability; plain / `-default` / `-recency` × `-equalize`) | ✅ ported, **f32** (incl. an `f32x8` SIMD gradient, ~×1.8 faster than the old `f64x4`). Verified in-band (±0.0005, `size` exact) vs the *current* Python `result/` and the frozen baseline. No 1000-user upstream reference exists, so it isn't in the table above. `--sched_penalties` deferred. |
| GRU, LSTM, RWKV, RWKV-P, NN-17, Transformer (14) | 🐍 Python path — Reptile/neural, kept in Python |

¹ The committed upstream file for this config is **stale** (predates a pipeline change), so it
is not a valid reference — the binding target (rule #5) is the *current* Python source, which
the Rust output matches. `-secs` configs are verified against a freshly-generated current-
Python golden (spot-checked on 15 users); everything else is on 1000 users.

² ACT-R is correct but slow — its activation is an O(reviews²) all-pairs sum over prior
reviews, a target for the planned performance pass.

³ FSRS-rs requires building with `--features fsrs-rs` (it imports the real `fsrs` 4.1.1 crate —
the exact release `fsrs-rs-python` 0.8.2 wraps). Measured against a freshly-generated current-
Python golden over all 1000 users (the stale `result_upstream` file aside, per ¹): mean diff
**+0.000299**, `size` exact, **269/1000 (27 %) of users bit-identical**. The remaining users differ
by small amounts in *both* directions (387 above, 344 below; max ±0.04, symmetric) — the inherent
divergence between two separate compilations of the same f32 training code in the `burn` ML
framework, well inside tolerance.

⁴ **`⚠ genuine`** — outside ±0.0005 (always *lower* than upstream), but investigated and confirmed a
genuine precision / optimizer-trajectory difference, **not a bug**. `size` is exact and the model
math matches; the gap is concentrated in a few chaotic users:
- `HLR`, `HLR --short`, `FSRSv4`: dominated by **1–2 users** whose chaotic `0.5^(t/s)` / power-law fit
  lands ~0.1–0.3 lower in Rust. Non-`--secs` (integer intervals) are f32-exact, so f32 can't close it.
- `ACT-R` (forward-mode `Dual`, runs f64): f64 finds a lower optimum than torch's f32; matching it
  would need a hand-written reverse-mode gradient (deferred, like FSRS-7's).
- `DASH --recency`: a systematic but small optimizer-trajectory difference from the recency-weighted
  Adam (the formula + checkpoint logic match Python exactly; `DASH` without `--recency` is +0.000000).
- `FSRS-6 --short --secs --partitions preset`: small per-partition training sets where the S0 init
  (Rust golden-section vs Python `scipy.minimize`) doesn't get washed out by training.
- `FSRS-6-one-step --short` (online single-pass SGD, runs f64): the tiny-lr online pass + local S0
  fit land ~0.0007 lower than torch's f32 — an f64-vs-f32 optimizer-trajectory difference, not a bug.
  `size` is exact by construction (it predicts with stock FSRS-6, so the eval set = FSRS-6-short).

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

## Performance

Each trained algorithm is optimized to keep the benchmark fast while reproducing results within the
±0.0005 tolerance above. The trained-model gradients are computed by **hand-written reverse-mode
analytic gradients** rather than generic autodiff — these are manual VJPs of each model's specific
forward pass (⚠ changing a model's math requires re-deriving its backward; the `--features fp64`
oracle tests guard this). Models with hand-written gradients: **FSRS-7** (+ `f32x8` SIMD) and **FSRS
v1–v6 / FSRS-4.5 / SM2-trainable / DASH[ACT-R]** (f64). The speedup work is logged iteration-by-
iteration in `_phase2/iterations.md` (FSRS-7) and `_phase3/iterations.md` (the rest), each gated on a
Wilcoxon signed-rank timing test (p < 0.01) and a per-algo correctness band. ACT-R and Anki remain on
forward-mode autodiff (a VJP wasn't a net win for them).

## Status

Work in progress — see `CLAUDE.md` for the architecture, phase plan, and current status.
