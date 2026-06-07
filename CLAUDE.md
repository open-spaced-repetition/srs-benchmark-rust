# srs-benchmark-rust — Claude handover

> **GitHub rule (always):** every GitHub comment posted on Andrew's behalf — PR
> descriptions, review replies, issue comments — **must start with the line
> "Written by Claude".** No exceptions.

## 0. What this repo is

A **Rust port of `open-spaced-repetition/srs-benchmark`**, whose sole purpose is to make
the benchmark **faster** while reproducing the same results. It commits to
`https://github.com/open-spaced-repetition/srs-benchmark-rust` (Andrew = `Expertium`, has
ADMIN; `gh` at `C:\Program Files\GitHub CLI\gh.exe`, logged in; git credential helper set
via `gh auth setup-git`).

- **Python source of truth (read-only reference):** `C:\Users\Andrew\srs-benchmark`
  (Expertium's fork). Its own `CLAUDE.md` describes a *different* sub-project (bit-exact
  Python speedup) owned by a *different* Claude — **not us, don't work there.** We only
  read it as the spec.
- Upstream repo: https://github.com/open-spaced-repetition/srs-benchmark

## 1. The 5 rules (from Andrew)

1. **Model definition files in `srs-benchmark/models/` stay in Python.** Interpreted (Andrew
   confirmed 2026-06-07) as: keep those Python files as the canonical spec; **reimplement
   the model math natively in Rust** for the ported algorithms. Keep the Python runtime
   path for models we do *not* port (GRU/LSTM = Reptile, RWKV, etc.). Everything else
   (data pipeline, harness, metrics, IO) → Rust for speed.
2. **GRU & LSTM use the Reptile optimizer** (hard to port). Do the **Adam**-based
   algorithms first; defer Reptile/neural nets to the Python path.
3. **Performance matters. Time each individual user** and record that time in the `.jsonl`
   output (a per-user field). We use it to find slow users / guide optimization.
4. **CLI stays identical** — same flags as
   https://github.com/open-spaced-repetition/srs-benchmark#scriptpy-options. See `config.rs`.
5. **Verify every ported model:** its **unweighted (simple arithmetic) mean LogLoss across
   1k users** (Andrew, 2026-06-07: 1k, not 10k, to save time) must stay within **±0.0005**
   of the original Python result. Reference result files:
   `C:\Users\Andrew\srs-benchmark\result_upstream\*.jsonl` (89 files, 10000 users each;
   compare the matching first-1000-user subset). LogLoss is the binding metric; other
   metrics are best-effort parity. (anki-revlogs-3k is a subset of -10k with identical
   per-user data, so 1k verification can run on the 3k dataset with `--max-user-id 1000`.)
6. **`size` (review count) must be EXACTLY identical** — both the per-user `size` value AND
   the sum of `size` across all users — versus the original Python, for every config. `size`
   = `len(y)` = number of evaluation rows for that user. This is NOT a tolerance: it is
   exact. It means the **feature pipeline's row filtering** (rating filter, `i>128` drop,
   short-term handling, `delta_t>0` final filter, and — for non-`--secs` — the outlier &
   non-continuous-row removal) must be reproduced bit-for-bit so the surviving row set, the
   TimeSeriesSplit, and thus the eval set match exactly. Per-user `size` mismatch ⇒ the port
   is wrong even if mean LogLoss happens to land within tolerance. **Verify `size` first**
   (cheap, exact) before trusting LogLoss.

## 2. Datasets (siblings, read-only — never write there)

Hive-partitioned parquet, `revlogs/cards/decks` each split by `user_id=N`:
- `C:\Users\Andrew\anki-revlogs-3k` — 3000 users; the default for quick iteration.
- `C:\Users\Andrew\anki-revlogs-10k` — 10000 users (matches upstream); for rule-#5 checks.

Parquet schemas:
- `revlogs/user_id=N/data.parquet`: `card_id, day_offset, rating, state, duration,
  elapsed_days, elapsed_seconds, __index_level_0__` (the last is the original row index =
  `review_th` ordering source).
- `cards`: `card_id, note_id, deck_id`. `decks`: `deck_id, parent_id, preset_id`.

## 3. The per-user pipeline (what we reproduce)

For each user (independent → parallelize with rayon):
1. **Load** `revlogs` parquet for the user.
2. **`create_features`** (`features/base.py` + per-model engineer): review_th, nth_today,
   `i` (review count per card), `delta_t`/`delta_t_secs`, `r_history`/`t_history` strings,
   `y` (rating→{1:0,2:1,3:1,4:1}), `rmse_bins_lapse`, `last_rating`, `first_rating`, and
   model-specific tensors. **Outlier/continuity filtering (`remove_outliers`,
   `remove_non_continuous_rows`) runs ONLY for non-`--secs` configs** → target `--short
   --secs` first to defer it.
3. **Split:** sklearn `TimeSeriesSplit(n_splits=5)`; first split is train-only (dropped
   from eval). (Untrainable models still split to define the test set.)
4. **Train** (trainable only): per split, per partition. Adam + CosineAnnealingLR, BCE
   loss (`reduction="none"` × weights, summed). See `Trainer` in `script.py`.
5. **Predict** on each split's test set; collect `(p, y)`.
6. **Evaluate** (`utils.evaluate`) → stats dict → one JSON line.

## 4. Output format

`result/<name>.jsonl`, one JSON object per user, sorted by `user` at the end
(`sort_jsonl`). Current Python `evaluate()` emits:
```json
{"metrics": {"RMSE":..,"LogLoss":..,"RMSE(bins)":..,"smECE":..,"AUC":..,
 "precision@90":..,"recall@90":..,"ICI":..,"MBE":..}, "user": N, "size": M,
 "parameters": [...] or {"<partition>": [...]}}
```
(Older reference files have a subset of metrics — fine, we compare `LogLoss`.) All metric
values are `round(x, 6)`; `AUC` is `null` for single-class users. **We add a per-user
timing field (rule #3).**

Resume behaviour: `script.py` skips users already present in the result file (so delete it
for a fresh run). `--raw` → `raw/<name>.jsonl` (`{user, p[round4], y}`).

## 5. Build & run

```
cargo build --release          # binary: target/release/script(.exe)
target\release\script.exe --algo AVG --short --secs --data C:\Users\Andrew\anki-revlogs-3k --processes 16
```
Rust toolchain 1.95 present. Verify a model (in order):
1. **`size` exact** (rule #6): per-user `size` and the total `sum(size)` must match
   `srs-benchmark\result_upstream\<name>.jsonl` exactly. Do this first — it validates the
   feature pipeline / row filtering independently of any model math.
2. **mean LogLoss within ±0.0005** (rule #5): mean of `metrics.LogLoss` over **1k users**.

Run with `--data C:\Users\Andrew\anki-revlogs-3k --max-user-id 1000`, then compare to the
first-1000-user subset of the matching `result_upstream\<name>.jsonl`. (3k ⊂ 10k verified:
AVG/SM2 match upstream bit-for-bit, so the subset comparison is valid.)

## 6. Status / phase plan

Tracked in the task list. Order:
- **P0** repo + scaffold + verify push.
- **P1** foundation: CLI (`config.rs` ✓ drafted), parquet read, rayon, jsonl out + resume +
  sort + per-user timer.
- **P2** feature engineering (base pipeline; `--short --secs` path first).
- **P3** TimeSeriesSplit + metrics (LogLoss/RMSE/RMSE(bins)/AUC/MBE/precision@90/recall@90;
  then ICI via lowess, smECE via relplot).
- **P4** non-trainable: AVG, SM2, MOVING-AVG, Ebisu, RMSE-BINS-EXPLOIT (verify ±0.0005).
- **P5** Adam-trained: HLR, DASH, ACT-R, FSRS v1–v6 + Rust Adam/autodiff.
  - **FSRS-7 is DEFERRED** (Andrew 2026-06-07: the upstream FSRS-7 model is still WIP /
    being changed — don't port it yet).
- **P6** remaining: LogisticRegression, FSRS-rs, one-step, partitions, equalize, recency,
  non-secs outlier path; Python path for GRU/LSTM/RWKV/Transformer/NN-17.

### Trained-model matching (key finding, 2026-06-07)

The upstream trained references are **exactly reproducible** by the source Python on this
machine (HLR sourcePy == upstream to 6 dp). So a ported trained model only has to match the
Python training algorithm; the one uncontrolled variable is the **batch-visitation order**
(`BatchLoader` uses `torch.randperm(batch_nums, generator=Generator().manual_seed(2023))`,
advanced once per epoch). `train.rs` reproduces ATen's **MT19937 + 32-bit Fisher–Yates
`randperm` exactly** (unit-tested vs torch 2.10). Adam (no weight decay), CosineAnnealingLR
(`T_max = batch_nums*n_epoch`), summed BCE×weights, and best-weights-by-eval-loss all match
`script.py::Trainer`. Note Rust uses f64 vs torch f32 — fine within the ±0.0005 tolerance.

## 7. Conventions

- Match Python numerics closely but **bit-exactness is NOT required** (rule #5 is a ±0.0005
  tolerance), which buys freedom on reduction order, batch-shuffle RNG, fp32-vs-fp64, etc.
  Still, prefer the same math/order where cheap, to stay well inside tolerance.
- Keep flags/filenames identical to Python (`config.rs`). When unsure of a feature detail,
  read the Python in `C:\Users\Andrew\srs-benchmark` — it is the spec.
- Andrew is Python/PyTorch-first; keep Rust readable and explain non-Python tooling.
