# srs-benchmark-rust — Claude handover

> **GitHub rule (always):** every GitHub comment posted on Andrew's behalf — PR descriptions,
> review replies, issue comments — **must start with the line "Written by Claude".** No exceptions.

## 0. What this repo is

A **Rust port of `open-spaced-repetition/srs-benchmark`** whose sole purpose is to run the same
benchmark **faster** while reproducing its results. Commits go directly to `main` on
`https://github.com/open-spaced-repetition/srs-benchmark-rust` (Andrew = `Expertium`, ADMIN; `gh` at
`C:\Program Files\GitHub CLI\gh.exe`, logged in).

- **Python source of truth (read-only spec):** `C:\Users\Andrew\srs-benchmark` (Expertium's fork).
  Read it for any feature detail; **never write there.** Its own `CLAUDE.md` is a *different*
  sub-project — not us.
- Upstream: https://github.com/open-spaced-repetition/srs-benchmark

## 1. The rules (from Andrew)

1. **Model math is reimplemented natively in Rust** for the ported algorithms; the Python
   `srs-benchmark/models/*.py` stay as the canonical spec. Data pipeline, harness, metrics, IO → Rust.
2. **GRU/LSTM/RWKV/Transformer/NN-17 keep the Python path** (Reptile optimizer / neural). The
   Adam-based algorithms are the ones ported.
3. **Time each user** and record it in the jsonl (`time_ms`) to find slow users.
4. **CLI stays identical** to Python `script.py` (same flags/filenames; see `config.rs`). The
   smart-preset flags are the one Rust-only extension.
5. **mean LogLoss within ±0.0005, TWO-SIDED**, over **1000 users** (`--max-user-id 1000`). Outside
   [−0.0005, +0.0005] (higher OR lower) fails and is investigated — a lower loss can be a genuine
   f64-vs-f32 optimum but can also hide a bug. Reference = the **current Python**
   (`srs-benchmark/result*/`); some committed `result_upstream/*.jsonl` are stale. LogLoss binds;
   other metrics best-effort.
6. **`size` (review count) EXACTLY identical** — per-user AND the sum — vs Python, every config. It
   validates the feature pipeline's row filtering. **Check `size` first** (cheap, exact) before LogLoss.

## 2. Datasets (read-only siblings — never write there)

`C:\Users\Andrew\anki-revlogs-10k` — 10000 users, hive-partitioned parquet
(`revlogs`/`cards`/`decks`, each split `user_id=N`). `--max-user-id 1000` = the rule-#5/#6 subset.
- `revlogs/user_id=N/*.parquet`: `card_id, day_offset, rating, state, duration, elapsed_days,
  elapsed_seconds, __index_level_0__` (last = review_th order).
- `cards`: `card_id, note_id, deck_id`. `decks`: `deck_id, parent_id, preset_id`.

## 3. Per-user pipeline (parallelized with rayon)

Load revlogs → `create_features` (review_th, `i`, delta_t/delta_t_secs, r/t_history, y, rmse bins,
first/last rating, model tensors; non-`--secs` also runs `remove_outliers`/`remove_non_continuous_rows`)
→ `TimeSeriesSplit(5)` (first split train-only) → train (Adam + CosineAnnealingLR, summed BCE×weights)
→ predict → evaluate → one jsonl line. `--short --secs` is the simplest path (no non-secs outlier filter).

## 4. Output

`result/<name>.jsonl`, one object/user sorted by user: `{"metrics": {...}, "user": N, "size": M,
"parameters": [...] or {"<partition>": [...]}, "time_ms": ..}`. Metrics `round(,6)`; AUC null for
single-class. Resume skips users already present (delete the file for a fresh run).

## 5. Build & run

```
RUSTFLAGS="-C target-cpu=native" cargo build --release   # binary: target/release/script(.exe)
target\release\script.exe --algo FSRS-6 --short --secs --data C:\Users\Andrew\anki-revlogs-10k --processes 16
```
`-C target-cpu=native` enables FSRS-7's `f32x8` AVX2 gradient (plain build is correct, just SSE2).
Cargo features: `fp64` (all-f64), `fsrs-rs` (real `fsrs` 4.1.1 crate), `neural`/`neural-cuda` (candle
GRU/LSTM). Verify: **`size` exact first**, then **|mean LogLoss| ≤ 0.0005**. Tests: `cargo test` (f32);
`cargo test --features fp64` adds finite-diff/oracle math checks.

## 6. Precision: per-algo f32/f64 (critical)

`autodiff::round_scalar` rounds to f32 iff the static `autodiff::ROUND_F32` flag is set; `run.rs::run`
sets it per-algo via `algo_uses_f64()`. **"Use the precision that reproduces upstream":**
- **f32 algos** (analytic/reverse-mode gradient + non-trained): HLR, DASH, DASH[MCM], LogReg, FSRS-7,
  AVG/SM2/MOVING-AVG/Ebisu/RMSE-BINS/FSRS-rs. f32 matches torch (e.g. `HLR --short --secs` ±0 in f32
  vs −0.0058 in f64).
- **f64 algos** (`algo_uses_f64`; forward-mode `Dual` gradient): ACT-R, Anki, DASH[ACT-R], FSRS
  v1–v6/v4.5, FSRS-6-one-step, SM2-trainable. torch's f32 *reverse*-mode is only faithfully proxied by
  a *f64 forward*-mode gradient; an f32 forward-mode broke chaotic `--secs` training badly.

`--features fp64` forces all-f64 (finds slightly better optima but diverges from the f32 upstream).

## 7. Status

**All 65 upstream-referenced configs ported + verified** (size exact; mean LogLoss in-band, a few
`⚠ genuine` lower-than-upstream f64/optimizer-trajectory diffs — see the README table). Plus: FSRS-7
(f32, own SIMD gradient; verified vs current Python, `--sched_penalties` deferred), FSRS-rs
(`--features fsrs-rs`), GRU/LSTM (`--features neural`), `--partitions deck|preset`, equalize, recency,
two_buttons, S0, default, train_equals_test, the non-`--secs` path. **Smart presets** — §9.

**Perf (the project's point) — Phases 2 & 3 done** (per-iteration logs + frozen baselines in
`_speedup/phase2/`, `_speedup/phase3/`):
- **Autodiff** = forward-mode `Dual<P>` (`autodiff.rs`): the recurrence is written once; `P=0`
  predict, `P=NP` gradient; every gradient finite-diff tested.
- **Most trained algos now use hand-written reverse-mode analytic gradients** (one
  `src/models/<m>_grad.rs` each) instead of forward-mode in `Model::grad`: FSRS v1–v6/v4.5,
  SM2-trainable, DASH[ACT-R] (closed-form), and FSRS-7 (+`f32x8` SIMD). **⚠ These are MANUAL VJPs of
  each model's specific forward — changing a model's math requires re-deriving its backward; the
  `--features fp64` `*_grad_matches_*` oracle tests guard against drift.** Still forward-mode: Anki
  (VJP wasn't a net win at NP=7), ACT-R, FSRS-6-one-step (hand-derived single-transition grad already).
  **ACT-R got two speedups** (2026-06-21, `_speedup/phase3/iterations.md`): (1) algorithmic — its
  activation recurrence `m[i]` is a prefix shared by all of a card's rows, so `Model::retentions`
  computes it ONCE per card instead of per row (O(N³)→O(N²)/card), ×1.65, bit-identical; (2) a leaner
  `Dual::powd` (`autodiff.rs`) reuses the value (`da = e·aᵉ/a`) instead of a 2nd `powf` — ×1.35 more on
  ACT-R, predictions bit-identical (P=0 elides the dead grad terms), grad perturbed ~1e-16. (2) helps
  every forward-mode-`Dual` grad user (ACT-R, Anki, FSRS-6-one-step); the VJP'd algos hand-write grads.
  (3) ACT-R inner power `a.ln().mul(e).exp()` instead of `a.powd(e)` — `ln a` computed once & reused for
  value+grad (vs powf's internal ln + a separate one), ×1.55 more; value `exp(e·ln a)` ~1 ULP off powf,
  output bit-identical at 6-dp. ACT-R cumulative ≈×3.5; it's transcendental-bound (a VJP would NOT help).
- **DASH precompute log1p(features)** (2026-06-21): `Dash::z` recomputed `ln(feat+1)` for all 8 features
  every predict/grad call; features are constant so store `logfeat` once in `from_rows` → ×1.5 on all 8
  DASH configs, bit-identical. (A per-card O(N log N) feature-*build* rewrite was REJECTED first — ×0.99,
  the build wasn't the bottleneck. Lesson: measure, don't assume O(N²) build dominates.) Anki/SM2-trainable
  per-card predict also REJECTED (cheap recurrence → grouping overhead made it slower).
  **FSRS v1–v6 per-card predict** (2026-06-21): `predict`/`eval_loss` replayed the recurrence per row
  (O(N²)/card); extracted `forward_states` (records stability after each review) so `predict` runs it
  ONCE per card, each row reads `states[pos-1]` → O(N)/card, **bit-identical**. ×1.36 FSRS-6 / ×1.27
  FSRS-5; all FSRS configs. NOT grad: grad is per seq-len-sorted batch (a card's rows split across
  batches), so per-card sharing doesn't apply there (and that matches Python's batched O(N²)).
- **Speedup protocol** (if resumed): 200 users, before/after run SIMULTANEOUSLY 1-thread-each, accept
  iff Wilcoxon p<0.01 AND faster AND within ±0.0005 of a FROZEN baseline AND size exact; log EVERY
  iteration with the exact p-value in `_speedup/phase{2,3}/iterations.md` (harness there too).

## 8. Key gotchas (still live)

- **BCE clamp:** `train.rs::bce` (best-weights selection) clamps each log term to min −100 (torch's
  `binary_cross_entropy`), NOT `p` to EPSILON (which caps log at −36 and accepts overfit epochs).
- **Determinism:** `fit_s0` sorts its grouped `(delta_t, recall, count)` before the loss sum (a
  HashMap's random order gave non-deterministic S0 → non-deterministic results).
- **S0 init** (`models/fsrs_init.rs`): per-first-rating golden-section 1-D fit + interpolation table.
  FSRS-6-one-step uses a *local* descent variant (`fit_s0_from_x0`).
- **Non-`--secs` row filtering:** new cards log `elapsed_days = -1` (not 0); only `i==2` rows are
  removable; the whole-card-vs-`i==2` test is `first_review.elapsed_days <= 0`.
- **Reference staleness:** some `result_upstream/*.jsonl` predate code changes — bind to *current*
  Python when a config "fails" only vs the stale file.
- **`--secs`-only configs are BLOCKED (upstream bug, 2026-06-22):** `ACT-R-secs`, `AVG-secs`,
  `DASH-secs`, `DASH[ACT-R]-secs`, `DASH[MCM]-secs`, `NN-17-secs` (`--algo X --secs`, no `--short`).
  In srs-benchmark Python, `--secs`-only `size` matches **neither** plain (no flags) **nor**
  `--short --secs`, but it *should* == plain no-flags (both drop same-day; `--secs` only changes
  day→sec intervals). So the Python reference is itself buggy → no valid binding target. GitHub issue
  filed. **Do NOT benchmark these in Rust or touch their result files until upstream is fixed.** The
  README Log Loss tables already exclude `--secs`-only configs. (Verified live, not just stale files.)
- **Batch RNG:** `train.rs` reproduces ATen MT19937 + 32-bit Fisher–Yates `randperm` (unit-tested).

## 9. Smart presets (`--partitions smart`, FSRS-7) — Rust-only extension

Cluster a user's decks by FSRS-7 param similarity (Mahalanobis) into data-driven presets, train per
cluster. Per `TimeSeriesSplit` fold: per-deck train (the deck path) → log-transform params 0–3 + whiten
(`src/smart.rs`, MinCovDet covariance `_smart/smart_preset_cov.json`) → cluster → per-cluster train →
predict (test-only deck → nearest cluster to the user's global params). Eval row-set == non-partitioned
⇒ `size` == `FSRS-7-short-secs`.
- Clustering: `src/cluster.rs` hierarchical linkage (kodama) + SciPy-exact `fcluster(distance)`;
  `src/hdbscan.rs` sklearn-matching HDBSCAN (eps inert; sweep varies mcs/ms/eom-leaf). Both unit-tested.
- `--cluster_sweep` runs the whole matrix sharing per-deck training (hierarchical 30; HDBSCAN 16 via
  `--cluster_method hdbscan`). Tooling + stripped result archive (`_smart/results/`) + xlsx appender
  (`_smart/append_smart_xlsx.py`) in `_smart/`.
- **`--cluster_distance kl`**: cluster by similarity of deck *predictions* (mean symmetric Bernoulli
  KL) instead of params. Closed form `½·(pₐ−p_b)·(logit pₐ−logit_b)` + a 256-row strided subsample
  (`KL_DIST_CAP`) — the raw O(decks²·rows) matrix is intractable (users have up to ~4947 decks; see
  [[anki-revlogs-deck-counts-kl-cost]]). Thresholds calibrated via `_smart/kl_calibrate.py`.
- **`--cluster_method optimal --cluster_sweep`**: skip distance clustering; search partitions for the
  min AIC/BIC on the *training* fold (no test peek). Tiers: exhaustive (≤6 decks, memoized 2^N−1
  subsets), greedy agglomerative (7–12), Maha/KL pre-merge→12 then greedy (>12). 4 files
  (`-smart-opt-{bic,aic}-{maha,kl}`); fallback clusters don't count toward the param penalty `k`.
- **Finding (1000 users): NOTHING beats the per-user global model** — across Mahalanobis + KL
  distances, hierarchical + HDBSCAN, and the honest AIC/BIC optimal partition, every config's mean
  LogLoss is ≥ baseline (best −0.00003 = f32 noise). The 10-user optimal signal did not hold. Pooling
  all of a user's decks into one model wins. See `Smart Preset Assignment.xlsx`.
- **Shared partition fixes** (deck/preset too): missing-`cards` card → partition −1 (Python
  `fillna(-1)`); card-less user → all −1; inadequate-partition double-fallback (→ user-level → INIT_W).

## 10. Research modes (Rust-only) — `_hpprobe/FINDINGS.md`

Three flags that ask "how much better could FSRS-7 be" instead of reproducing Python. All keep the
evaluated row set identical to a normal run, so **`size` stays exact** and the LogLoss is directly
comparable. Full write-up + numbers: **`_hpprobe/FINDINGS.md`**; analysis scripts in `_hpprobe/`.

- **`--reopt_growth <eps>`** (`fsrs_v7::process_geometric`): refit from `INIT_W` whenever the
  training set has grown by `(1+eps)`, instead of only at the 5 `TimeSeriesSplit` boundaries — an
  upper bound on re-optimizing often. `TimeSeriesSplit` pools test folds covering exactly
  `rows[eval_start..]` and the geometric schedule partitions that same range, so `size` is identical
  by construction. Cost `~n(1+eps)/eps` vs the 5-fold `2.5n` ⇒ eps=1.0 (doubling) is *cheaper* than
  the status quo; eps=0.003 is 97x. **Result (10k users, eps=0.003, 99.7% freshness): FSRS-7 recency
  0.317947 → 0.304891 (−0.013056), 99.8% of users improved.** Beats GRU/LSTM, still **+0.0075 short
  of RWKV** (0.297427) and +0.0389 of RWKV-P. ⚠ Train prefixes are passed as SLICES, never cloned
  (~1500 refits/user). ⚠ **Chunk long runs** (`--max-user-id 1000, 2000, …`): results are written
  only after the whole parallel loop, so an unchunked 18 h run loses everything on interrupt.
- **`--hp_probe` / `--hp_features`**: per-user hyperparameter study (lr/betas/n_epoch). **Conclusion:
  under a <2x budget there is NOTHING to win (~−0.00008).** A decision tree on user statistics gets
  −0.000383 CV vs −0.000356 for just always-45-epochs — i.e. nothing; the per-user optimum is not a
  function of user statistics (a NN would not fix it — the limit is signal, not model class).
  Per-user selection *does* work but needs real trial runs: 3-way lr choice = −0.000491 at 3.40x CPU
  (~1.2x wall on 3 cores, since the runs are independent). **Dead ends: selecting then skipping the
  refit is +0.0058** (losing 20% of training data costs ~10x any tuning gain), and per-fold selection
  is 4x worse than per-user.
- **Hypergradient lr** (`train.rs`, `TrainConfig::hyper_beta`, Baydin et al. 2018): free
  (`dL/da = −∇L(w_t)·u_{t−1}`, one dot product/step), **but worth only −0.000038**. Normalized +
  multiplicative (`base_lr *= exp(β·cos)`) because the raw dot product scales with the summed loss
  and batch size; cosine annealing kept on top. `hyper_beta = 0.0` leaves the loop **bit-identical**
  (verified 30 users). ⚠ `Adam::last_u` is computed as a SEPARATE statement — do NOT factor it out of
  the parameter update, `(lr*mhat)/d ≠ lr*(mhat/d)` in floating point.

## 11. Conventions

- **One model per file** under `src/models/` (mirrors Python `models/`); each exposes
  `process(ds, cfg) -> ModelOutput`. Shared infra (Adam, cosine LR, MT19937 randperm, train loop) in
  `train.rs`; `run.rs` dispatches `models::<name>::process`.
- **Bit-exactness NOT required** (±0.0005 tolerance) — freedom on reduction order, RNG, f32-vs-f64;
  still prefer the same math/order where cheap.
- When unsure of a feature detail, read the Python spec. Andrew is Python/PyTorch-first — keep Rust
  readable and explain non-Python tooling.
- **Long benchmark runs:** launch via detached PowerShell `Start-Process` (chunked `.ps1`), NOT the
  Bash tool's background (which dies on interrupt). 2-thread cap while Andrew benchmarks Python.
