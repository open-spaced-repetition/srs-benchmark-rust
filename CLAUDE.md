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
   1k users** (Andrew, 2026-06-07: 1k, not 10k, to save time) must not be **WORSE (higher)
   than the original Python by more than 0.0005**. It may be **arbitrarily BETTER (lower)** —
   one-sided tolerance (Andrew, 2026-06-07). I.e. PASS iff `mean_rust − mean_upstream ≤
   0.0005`. (Our f64 finds slightly better optima than torch's f32 on chaotic models like
   HLR/FSRS, giving lower loss — that's fine.) Reference files:
   `C:\Users\Andrew\srs-benchmark\result_upstream\*.jsonl` (10000 users each; compare the
   first-1000-user subset). LogLoss is the binding metric; other metrics best-effort.
   (Verify on `anki-revlogs-10k --max-user-id 1000`.)
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
- `C:\Users\Andrew\anki-revlogs-10k` — 10000 users (matches upstream). Use this for all
  runs/verification; `--max-user-id 1000` selects the first 1000 users for rule-#5 checks.

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
target\release\script.exe --algo AVG --short --secs --data C:\Users\Andrew\anki-revlogs-10k --processes 16
```
Rust toolchain 1.95 present. Verify a model (in order):
1. **`size` exact** (rule #6): per-user `size` and the total `sum(size)` must match
   `srs-benchmark\result_upstream\<name>.jsonl` exactly. Do this first — it validates the
   feature pipeline / row filtering independently of any model math.
2. **mean LogLoss one-sided** (rule #5): `mean_rust − mean_upstream ≤ 0.0005` over **1k
   users** (better/lower is always fine).

Run with `--data C:\Users\Andrew\anki-revlogs-10k --max-user-id 1000`, then compare to the
first-1000-user subset of the matching `result_upstream\<name>.jsonl`.

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
  - **FSRS-rs (Andrew 2026-06-07): IMPORT the real `fsrs-rs` crate**
    (`open-spaced-repetition/fsrs-rs`, the `fsrs` crate) and call it — do NOT reimplement
    FSRS-6 training by hand. The benchmark's FSRS-rs config is literally that library.

### Trained-model matching (key finding, 2026-06-07)

The upstream trained references are **exactly reproducible** by the source Python on this
machine (HLR sourcePy == upstream to 6 dp). So a ported trained model only has to match the
Python training algorithm; the one uncontrolled variable is the **batch-visitation order**
(`BatchLoader` uses `torch.randperm(batch_nums, generator=Generator().manual_seed(2023))`,
advanced once per epoch). `train.rs` reproduces ATen's **MT19937 + 32-bit Fisher–Yates
`randperm` exactly** (unit-tested vs torch 2.10). Adam (no weight decay), CosineAnnealingLR
(`T_max = batch_nums*n_epoch`), summed BCE×weights, and best-weights-by-eval-loss all match
`script.py::Trainer`. Note Rust uses f64 vs torch f32 — fine within the ±0.0005 tolerance.

**BCE clamp (key fix, 2026-06-08):** `train.rs::bce` (used for best-weights selection) must
clamp each **log term to min −100** (torch's `binary_cross_entropy`), NOT clamp `p` to
`f64::EPSILON` (which caps the log at ≈−36). The −36 cap under-penalized confidently-wrong
predictions, so on chaotic models the selector accepted overfit epochs torch rejects — e.g.
FSRSv1 plain user 541 (rust trained to worse weights / LogLoss 3.82; torch kept init / 3.35).
Fixing the clamp made FSRSv1 plain pass (+0.000947 → +0.000445) and pulled the whole FSRS
family's short-secs diffs from ~−0.002 toward ~0 (closer to torch). The `fp32` experiment was
what ruled out precision and forced finding this — the divergence was structural, not f32-vs-f64.

**`fp32` build feature (2026-06-08, kept):** `cargo build --features fp32` rounds every
autodiff + Adam result to f32 (mimics torch); default is f64 (no-op). Experiment showed f32
does NOT meaningfully change FSRS results (plain configs use f32-exact integer intervals).
Keep f64 as default. `autodiff::round_scalar` is the toggle.

**Rule #5 is ONE-SIDED (Andrew 2026-06-07):** PASS iff `mean_rust − mean_upstream ≤
0.0005`. Lower (better) is always fine — f64 finds slightly better optima than torch f32 on
chaotic models (extreme `0.9^(t/s)`/`2^d` predictions → a few users amplify f64-vs-f32
noise), so several read *lower* than upstream. Keep f64 everywhere; do NOT switch to f32.

**VERIFIED (18 models, vs `result_upstream`, `--short --secs`, ALL on the full 1000-user
basis; size exact per-user + sum for every one):**
AVG/SM2/MOVING-AVG bit-exact; SM2-trainable −0.000620 (`models/sm2_trainable.rs`, Adam,
reuses FSRS infra); Ebisu-v2 +0.000000 (well-conditioned — own Lanczos `lgamma` +
scipy-style `brentq`, `models/ebisu.rs`); RMSE-BINS-EXPLOIT exact vs *current* Python
(upstream file stale); DASH −6e-6, DASH[MCM] −1e-6, DASH[ACT-R] −5e-5; HLR −0.004352;
FSRS v1 −0.001477, v2 −0.001793, v3 −0.002348, v4 −0.000341, v4.5 +0.000249, v5 +0.000037,
v6 +0.000049; ACT-R −0.001420. (Re-verified on 1000 users at 10 threads after Andrew lifted
the 1-thread limit; the earlier 200/20-user numbers are superseded. All pass the one-sided
rule; max positive is FSRS-4.5 at +0.000249.)

**NON-`--secs` PATH NOW PORTED + VERIFIED** (`features.rs`: `remove_outliers` +
`remove_non_continuous_rows`, run only when NOT `--secs`). Key gotcha: new cards log
`elapsed_days = -1` (not 0), so the "card has an `i==1` row" test (which decides whole-card
vs `i==2`-only drop) is `first_review.elapsed_days <= 0`. Only `i==2` rows (each card's first
positive-interval review) are removable, so the only continuity gap can be at `i==2`.
Verified on the 5 models with `-short` references (1000 users, size exact, sum 32 668 830
each): SM2 +0.000000, DASH +0.000155, HLR −0.000763, FSRS-5 +0.000001, FSRS-6 −0.000008 —
all PASS. (No `-short` upstream ref exists for the other ported models → they stay
`--secs`-only.)

**FSRS autodiff = forward-mode dual numbers** (`autodiff.rs`, `Dual<P>`): the recurrence is
written ONCE over `Dual<P>`; `P=0` → fast value-only predict, `P=NP` → param gradients.
Every model's gradient is finite-difference unit-tested. **S0 init** (`models/fsrs_init.rs`)
= per-first-rating golden-section 1-D fit + interpolation table (replaces scipy.minimize;
one-sided rule makes a true-minimum search safe). Train hooks in `train.rs`: `clip_params`
(per-step clipper), `grad_mask` (v4/v4.5 freeze first 4), `eval_penalty` (v5/v6 L2). Per-user
timing field is **`time_ms`**.

**Determinism fix (2026-06-08):** `fit_s0` summed its fit-loss over a `HashMap`'s randomized
iteration order → non-associative f64 sum → different S0 → non-deterministic init weights →
non-deterministic results for ALL `fit_s0` models (FSRS v4/v4.5/v5/v6), ~1e-4 on the mean
(within tolerance but real). Fixed by sorting the grouped `(delta_t, recall, count)` by
`delta_t` before the loss sum (also matches pandas `groupby` key order). v1–v3 (fixed init)
and non-fit_s0 models were already deterministic. The `rmse_bins` HashMap also iterates
randomly but only perturbs RMSE(bins) at ~1e-15 (invisible after `round(,6)`), so it's benign.

**⚠ PERF (the project's whole point — not yet addressed):** forward-mode is P× the value
forward, so FSRS training is slow single-thread; ACT-R is worse (O(reviews²) all-pairs).
Correct but needs a **reverse-mode / batching perf pass** (forward-mode models are the
oracle). The data pipeline + non-trained models are already fast.

**DONE since:** Anki (3 configs), `--recency`/`--two_buttons`(binary)/`--S0`/`--default`/
`--train_equals_test` flag variants, `--partitions deck/preset` (FSRS-6, isolated
`process_partitioned` branch + `data::read_user_partition_map` cards→decks join), and
LogisticRegression (`models/logistic_regression.rs`: 34-feature linear model, AdamW, analytic
gradient; feature_rating/first_rating use the FULL pre-filter card sequence while feat_elapsed
uses the surviving prior — that was the bug, +0.024 → +0.000001). **65 configs verified.**
**Determinism + BCE-clamp bugs fixed** (see notes above).

**FSRS-6-one-step PORTED + VERIFIED (2026-06-08):** `models/fsrs_v6_one_step.rs`. Online
FSRS-6: standard S0 init, then ONE pass over training reviews (review_th order), one SGD step
(lr=1e-4) per review on a hand-derived gradient that backprops through only the most-recent
transition. Predicts with stock FSRS-6 (`fsrs_v6::predict`) → eval row-set = FSRS-6-short, so
`size` is exact by construction; only `p` differs. The training-forward `step` is the model's
own simplified recurrence (S_MIN=0.001 floor, no s_max cap, `rating>=3` short-term branch, no
failure floor, w[17..19] left unclamped) — ported byte-for-byte since it drives the SGD path.
**Key fix:** the tiny-lr single pass barely moves S0, so the init must match scipy's *local*
descent from `x0=init_s0`, NOT the golden-section's global min. Added `fit_s0_from_x0`
(+`minimize_1d_local`: march downhill from x0 to bracket the basin, golden-section it, a
descent into a bound returns it) used ONLY by one-step; other fit_s0 models keep the global
golden-section (Adam re-optimizes S0 there, so it's unaffected — FSRS-6-short still −0.000008).
Global golden-section gave +0.000493 (a few users blew up: S0=1.3 where scipy floored to
0.001); local fit → **−0.000681** (better, comfortable margin). `FSRS-6-one-step --short`:
size exact, LogLoss −0.000681.

**`--equalize_test_with_non_secs` PORTED + VERIFIED (2026-06-08):** `features::build_equalize_splits`
+ `Dataset::equalize_splits: Option<Vec<EqSplit>>`. Under `--secs --equalize_test_with_non_secs`
the train/test split is governed by a `TimeSeriesSplit` over the **non-secs** survivors (run a
second `create_features` with `use_secs=false`): per fold, test = the secs rows whose `review_th`
is in the non-secs fold (non-secs survivors ⊆ secs survivors, 1:1), train = the secs prefix with
`review_th < fold.min`. **Feature values stay the plain `--secs` ones** — for LogReg the equalize
flag only swaps t_history's source `delta_t_secs`→`delta_t`, but the LogReg engineer already sets
`delta_t = delta_t_secs`, and the 34 features don't read t_history, so they're identical to a plain
`--secs` run. Built in `run.rs::process_user`, consumed by `logistic_regression::process` (and is
reusable for the deferred FSRS-7/neural equalize variants). Verified `LogisticRegression --short
--secs --recency --equalize_test_with_non_secs`: size exact (per-user + sum 32 668 830 — equals the
non-secs `-short` size, as expected since the test set is the non-secs survivors), LogLoss +0.000015.

**FSRS-rs IMPLEMENTED (2026-06-08, full 1000-user verify pending):** `models/fsrs_rs.rs`, gated
behind the optional `fsrs-rs` cargo feature (`cargo build --release --features fsrs-rs`). Imports
the real `fsrs` crate and calls `FSRS::new(Some(&[])).benchmark(ComputeParametersInput{train_set,
progress:None, enable_short_term:true, num_relearning_steps:None})` — the exact call
`fsrs-rs-python` makes — then rounds weights to 4 dp; predicts with stock FSRS-6 (`fsrs_v6::predict`,
matching the Python's `fsrs_optimizer.Collection`), so the eval set = FSRS-6-short and `size` is
exact. Items = one per training review (priors + current as `(delta_t=max(0,int), rating)`), in
review_th order, built from `prior_dt_active`/`prior_ratings` (port of `convert_to_items`).
**⚠ Version gotcha:** the published PyPI wheel `fsrs-rs-python` 0.8.2 does NOT match the v0.8.2
*git tag* (which pins fsrs git rev `932bb7af` = FSRS-5, 19 params). The installed wheel returns
21 params (FSRS-6); a panic-path probe revealed it links **crates.io `fsrs 4.1.1`** (which exports
19-param `DEFAULT_PARAMETERS` but trains a 21-param model). So Cargo.toml pins `fsrs = "=4.1.1"`
(crates.io, NOT a git rev). Weights aren't bit-exact but
match closely; 20-user spot check −0.000378 (size exact).

**FSRS-rs VERIFIED vs CURRENT PYTHON; upstream file is STALE (2026-06-09):** size exact
(32 668 830). The stored `result_upstream/FSRS-rs-short.jsonl` is **stale** — the *current* Python
source does NOT reproduce it either (e.g. user 1: current-Python LogLoss 0.492687 vs upstream file
0.493072), so per rule #5 / §7 the binding target is current Python, NOT the stale file. Against a
freshly-generated **current-Python golden** (run `model_processors.process_fsrs_rs` per user), the
Rust port is **+0.000212** over 30 users with **10/30 bit-identical** (e.g. users 1 & 3 match to all
6 dp) and size exact — comfortably inside tolerance. (My initial "+0.000766 vs upstream / pipeline
crashes" scare was a **bug in my own diagnostic**: I called `create_features` on a df that
`data_loader.load_user_data` had *already* feature-engineered — double-processing produced the
float/empty `t_history` and the crash. The real pipeline runs clean; the other Claude confirmed
FSRS-rs is bit-identical between `df47eedc` and `c8b492e`.) `fsrs::benchmark` is deterministic &
thread-independent. **Full 1000-user current-Python golden (parallel, RAYON_NUM_THREADS=1; the 3
heaviest users — e.g. u927 with 375 k reviews — finished single-threaded): mean diff +0.000299,
size exact (sum 32 668 830), 269/1000 (27 %) bit-identical**, the rest differing by small amounts in
BOTH directions (387 above, 344 below; max ±0.04, symmetric) ⇒ the residual is f32 divergence
between two separate `burn` compilations of the same `fsrs 4.1.1` training code, NOT an item bug —
well inside the one-sided tolerance. FSRS-rs is VERIFIED vs current Python. Requires `--features
fsrs-rs`.

**REMAINING:** 90%/ConstantModel (no upstream ref → can't verify); `--raw`/`--file`/`--weights`
output; ICI(lowess)/smECE(relplot) metrics; Python path for GRU/LSTM/RWKV/Transformer/NN-17; the
perf pass. FSRS-7 deferred (10 configs). All 65 verifiable upstream configs are now ported &
verified; the remaining 24 are deferred (FSRS-7 ×10) or Python-path neural (×14).

## 7. Conventions

- **One model per file** under `src/models/` (mirrors the Python `models/` layout): each
  exposes `process(ds, cfg) -> ModelOutput`. Shared training infra (Adam, cosine LR,
  MT19937 randperm, train loop) lives in `train.rs`; `models/mod.rs` holds `ModelOutput` +
  `recency_weights`. `run.rs` dispatches `models::<name>::process`.
- **Reference staleness:** some `result_upstream/*.jsonl` files predate code changes (e.g.
  RMSE-BINS-EXPLOIT: my port is bit-identical to *current* source Python, but the upstream
  file differs by ~0.04). Rule #5's target is the *current* Python version — when a model
  fails vs upstream, check it against the current source Python (`harness`/golden) before
  assuming a bug.
- Match Python numerics closely but **bit-exactness is NOT required** (rule #5 is a ±0.0005
  tolerance), which buys freedom on reduction order, batch-shuffle RNG, fp32-vs-fp64, etc.
  Still, prefer the same math/order where cheap, to stay well inside tolerance.
- Keep flags/filenames identical to Python (`config.rs`). When unsure of a feature detail,
  read the Python in `C:\Users\Andrew\srs-benchmark` — it is the spec.
- Andrew is Python/PyTorch-first; keep Rust readable and explain non-Python tooling.
