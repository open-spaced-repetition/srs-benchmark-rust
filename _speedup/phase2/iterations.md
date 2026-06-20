# FSRS-7 Phase-2 speedup iteration log

Every accepted **or rejected** speedup is logged here. Required fields (Andrew, 2026-06-19):
**timestamp, iter #, LogLoss before, LogLoss after, avg time/user before, avg time/user after,
Wilcoxon p-value.**

Rules (project CLAUDE.md §6):
- **Correctness gate:** each iteration's 200-user avg LogLoss (per config) must stay within
  **±0.0005 of the FROZEN ORIGINAL** Rust FSRS-7 (`_fsrs7_baseline/`), NOT the rolling champion.
  Frozen LogLoss: plain 0.323582, default 0.346049, default-eq 0.378932, recency 0.320772,
  recency-eq 0.345821.
- **Speed gate:** measure per-user `time_ms` over 200 users, baseline vs candidate run
  **SIMULTANEOUSLY** (1 thread each = 2 total). Accept iff a **Wilcoxon signed-rank** test on the
  200 paired `time_ms` gives **p < 0.01** (faster). "before" = previous champion binary.
- LogLoss before/after below = the **timing config** measured; the correctness gate is checked
  across all 5 configs separately (see each iteration's notes).

| iter | timestamp | change | timing config | LogLoss before | LogLoss after | avg ms/user before | avg ms/user after | speedup | Wilcoxon p | decision |
|------|-----------|--------|---------------|----------------|---------------|--------------------|-------------------|---------|------------|----------|
| 1 | 2026-06-19 | analytic reverse-mode gradient (replace forward-mode `Dual<34>` autodiff with hand-written scalar VJP, f64; per-prefix-item batching unchanged) | FSRS-7-short-secs (plain) | 0.323582 | 0.323700 | 14841 | 3743 | 3.65× median / 3.96× total | 7.181e-35 | **ACCEPT** |
| 2 | 2026-06-19 | windowed O(C) predict (replay each card's sequence ONCE, emit a prediction at each needed position; bit-identical; grad path unchanged) | FSRS-7-default-short-secs | 0.346049 | 0.346049 | 180 | 90 | 1.34× median / 2.00× total | 5.356e-22 | **ACCEPT** |
| 3 | 2026-06-19 | f64×4 SIMD gradient (vectorize per-prefix recurrence fwd+bwd across 4 rows/lane, AVX2; `wide` Cephes exp/ln; EXACT batching preserved) | FSRS-7-short-secs (plain) | 0.323582 | 0.323699 | 4182 | 1366 | 2.84× median / 3.06× total | 7.181e-35 | **ACCEPT** |
| 4 | 2026-06-19 | remove SIMD-driver allocations + dynamic-dispatch closures (reuse `caches`/`lanes` scratch buffers, build lane arrays inline) — **byte-identical** | FSRS-7-short-secs (plain) | 0.323699 | 0.323699 | 1327 | 1176 | 1.16× median / 1.13× total | 8.495e-31 | **ACCEPT** |
| 5 | 2026-06-19 | hoist loop-invariant long-term-decay block (`decay2`/`inv2`/`p28`/`factor2`, dep. only on w24/w26) into `WLanes` — 1 fewer `exp`/step — **byte-identical** | FSRS-7-short-secs (plain) | 0.323699 | 0.323699 | 1256 | 1211 | 1.04× median / 1.04× total | 7.243e-14 | **ACCEPT** |
| 6 | 2026-06-19 | exp-fusion: `s32·ex33 → se = exp(w30·ln_s + (w31−0.5)·(d−5))` in SIMD curve (1 fewer `exp`/step; ~1-ulp) | FSRS-7-short-secs (plain) | 0.323699 | 0.323699 | 1199 | 1126 | 1.07× median / 1.07× total | 4.566e-20 | **ACCEPT** |
| 7 | 2026-06-19 | reciprocal-multiply for reused denominators (`s`/`sf`/`wsum`) in `curve4_bwd` (1/x·y vs x/y) | FSRS-7-short-secs (plain) | 0.323699 | 0.323699 | 1129 | 1138 | 0.99× median / 0.99× total | 9.965e-01 | **REJECT** (slower) |

## Iteration details

### iter 1 — analytic reverse-mode gradient (autodiff removal)

- **Change:** new module `src/models/fsrs_v7_grad.rs` (f64 port of the scalar forward+backward in
  `fsrs-rs-speed-autoresearch/fsrs-rs/src/analytic.rs`: `curve_fwd/bwd`, `stab_fwd/bwd`,
  `next_d_fwd/bwd`, `step_fwd/bwd`, `loss_and_grad_range`). `Model::grad`/`predict` in `fsrs_v7.rs`
  now call it instead of building `Dual<34>`. `train.rs` and the batching are unchanged, so the
  optimization trajectory is identical up to f64 reduction order.
- **Correctness (200 users, vs frozen baseline) — ALL 5 PASS, size exact:**
  plain +0.000118, default 0.000000 (bit-identical), default-eq 0.000000, recency +0.000124,
  recency-eq +0.000178. (max |Δ| 0.000178 ≤ 0.0005.)
- **Validation:** 3 unit tests pass — analytic predict vs forward-mode <1e-9, analytic grad vs
  forward-mode <1e-6, finite-difference grad check.
- **Speed (simultaneous, plain config, 200 users, 1 thread each):** total time_ms 2 968 129 →
  748 648 (**×3.96**); median per-user ratio 0.2740 (**×3.65**); avg ms/user 14 841 → 3 743.
  Wilcoxon two-sided p = 1.436e-34, one-sided (after<before) p = **7.181e-35** ≪ 0.01. Size exact
  between base & cand (sum 9 847 205, 0 per-user mismatches).
- **Decision: ACCEPT.** New champion binary `target/release/script_1.exe` (= `script_analytic.exe`).
  Next iteration's "before" = this champion.

### iter 2 — windowed O(C) predict (the "review preprocessing" lever)

- **Profiling (default config, 100 users):** the recurrence **predict** is 80% of a `--default`
  run (`read` 7%, `feat` 13%) — i.e. the O(C²) per-card prefix *replay*, not `create_features`,
  is the cost. (For training configs predict is only ~2.5%; the grad dominates.)
- **Change:** `predict_card` in `fsrs_v7_grad.rs` + rewired `Model::predict` (`fsrs_v7.rs`): group
  the rows being predicted by card, replay each card's sequence ONCE (emitting a prediction at each
  requested position) instead of replaying every prefix independently. O(C) vs O(C²) per card.
  Grad path untouched.
- **Correctness:** **bit-identical** — default config 0/200 users with any LogLoss change (max |Δ|
  0.00000000), size exact. (Predict order doesn't affect results, so no batching/trajectory risk.)
  All-5-config check vs frozen baseline: _in progress_ (expected identical to iter 1).
- **Speed (simultaneous, FSRS-7-default-short-secs, 200 users, 1 thread each):** total time_ms
  35 963 → 18 000 (**×2.00**); median per-user ratio 0.7464 (**×1.34**) — heavy users (long card
  histories) gain most, as expected. Wilcoxon one-sided p = **5.356e-22** ≪ 0.01.
- **Decision: ACCEPT** (bit-identical + strictly faster-or-equal; big win on `--default` configs,
  neutral on training configs since predict is a small slice there). Champion → `script_2.exe`.

---

### iter 3 — f64×4 SIMD gradient (AVX2)

- **Change:** new `src/models/fsrs_v7_simd.rs` — f64×4 (`wide`) port of the scalar analytic
  forward+backward, vectorizing the per-prefix recurrence across **4 rows/lane**. `Model::grad`
  builds `WLanes` (splatted params), sorts the batch by prefix length, runs groups of 4 (padding
  shorter lanes with `rating==0` = frozen state), reduces lane gradients into the scalar `gw`.
  Rows with `pos==0` (empty prefix; rare) fall back to scalar `grad_one`. **Predict untouched**
  (still scalar, bit-identical from iter 2). Transcendentals = `wide`'s Cephes f64×4 exp/ln (~1 ulp).
  Built with `RUSTFLAGS="-C target-cpu=native"` (Zen 3 AVX2+FMA).
- **Validation:** unit test `simd_grad_matches_scalar_grad` — SIMD group-of-4 grad == Σ scalar
  `grad_one`, rel < 1e-7. The batching is **unchanged** (same per-prefix items, same order) ⇒ no
  trajectory shift; only the ~1e-14 f64×4-transcendental FP difference.
- **Correctness (200 users, vs frozen) — ALL 5 PASS, size exact, *identical to the scalar
  champion at 6 dp*:** plain +0.000117, default 0.000000 (no training ⇒ bit-identical),
  default-eq 0.000000, recency +0.000124, recency-eq +0.000178. The f64×4-transcendental effect on
  every config's mean is invisible after `round(,6)`.
- **Speed (simultaneous, plain, 200 users, 1 thread each):** total time_ms 836 468 → 273 296
  (**×3.06**); median per-user ratio 0.3523 (**×2.84**); avg ms/user 4 182 → 1 366. Wilcoxon
  one-sided p = **7.181e-35** ≪ 0.01.
- **Decision: ACCEPT.** Champion → `script_3.exe`. **Cumulative training-path speedup ≈ ×10.9** vs
  the original forward-mode baseline (14 841 → 1 366 ms/user on the plain config). 4 unit tests pass
  (`simd_grad_matches_scalar_grad` + the 3 prior).

### iter 4 — drop SIMD-driver allocations + closures (byte-identical)

- **Change:** `grad_group`/`grad_simd` in `fsrs_v7_simd.rs` no longer allocate `dts`/`rts`/`caches`
  Vecs or use `&dyn Fn` closures *per group* (~128×/batch). Lane arrays built inline into `[f64;4]`;
  `caches` + `lanes` are scratch buffers reused across all groups (one alloc per `grad_simd` call).
- **Correctness:** **byte-identical** to iter 3 (20-user plain spot check max|Δ| = 0.0; same f64×4
  values fed to the kernels, just fewer allocations). Unit test still passes.
- **Speed (plain, 200 users, 1 thread each):** total 265 337 → 235 226 ms (**×1.13**); median ratio
  0.8653 (**×1.16**); avg ms/user 1 327 → 1 176. Wilcoxon one-sided p = **8.495e-31**.
- **Decision: ACCEPT.** Champion → `script_4.exe`. Cumulative training ≈ **×12** (14 841 → 1 176).

## Rejected by analysis (not measured)

### windowed O(N) training gradient + card batching — REJECTED (2026-06-19)

The training **grad** is ~94% of a trained config, and the expanding-window identity
(`window_grad == Σ per-prefix grads`, proven by unit test) makes the per-card gradient O(C). **But**
it requires batching by *card* instead of per-prefix-row, which changes the SGD trajectory.

**Killed by the reference repo's own data** (`fsrs-rs-speed-autoresearch` iter 18 — "THE WINDOW
BET"): their Phase-1 probe (card-grouped batching, slow per-prefix forward = the exact windowed
trajectory) cost **+0.000912 log loss** vs the per-prefix baseline; the worked example shows
**+0.00084 at 8 epochs**. This forced Andrew to *expand* their band from ±0.0010 to ±0.0015 there.

For us that is **~70% over the ±0.0005 gate**, and — critically — our frozen baseline *matches
Python to ~1e-6*, so a +0.0008 shift means the Rust benchmark stops reproducing Python's numbers
(defeats the project's purpose). **The windowed gradient is math-identical; the cost is purely the
batch-composition change.** Windowing is therefore safe for *predict* (iter 2, no batching,
bit-identical) but unsafe for *training*. Not implemented.

## Cumulative result (Phase 2)

Training path (plain config, 1 thread): **14 841 → ~1 126 ms/user ≈ ×13** (iter 1 ×3.96 analytic ×
iter 3 ×2.84 SIMD × iter 4 ×1.16 alloc × iter 5 ×1.04 hoist × iter 6 ×1.07 exp-fusion).
Default/predict-heavy configs: iter 2 ×2.0. All within ±0.0005 of frozen (every config +0.000117 …
+0.000178, identical to the scalar champion at 6 dp), all size-exact, all still reproducing Python.

**Floor reached (2026-06-19).** iter 7 (reciprocal-multiply) was REJECTED — *slower* on Zen 3 (the
reciprocal is itself a division and serializes the dependency chain that the independent divisions
pipelined). Combined with: (a) `wide`'s exp/ln are already Estrin-optimized (a custom lower-degree
Horner is slower, not faster), (b) the big levers (autodiff removal, SIMD) are spent, (c) the
windowed-training lever is out (breaks ±0.0005 / Python-repro) — the remaining within-gate headroom
is exhausted. Champion = `script_6.exe`.

**Build for the SIMD speed:** `RUSTFLAGS="-C target-cpu=native" cargo build --release` (Zen 3
AVX2+FMA). Plain `cargo build --release` is still correct — `wide` f64×4 runs on SSE2 baseline, just
narrower/slower.

## Planned next (within the ±0.0005 gate)

The two big levers (autodiff removal, SIMD) are spent. Remaining are smaller, all bit-/near-identical:
- **bit-identical micro-opts** of the kernels (hoist/dedup/skip redundant work) — ~1.1–1.3×;
- **SIMD the predict path** (4-wide across cards) — helps `--default` configs further (predict is
  ~90 ms/user there post-iter-2), negligible for training;
- (f32 SIMD / minimax / windowed-training — the reference repo's other wins — trade accuracy beyond
  our ±0.0005 gate, so they stay OUT.)

## iter 8 — PRECISION SWITCH to f32 (2026-06-19, Andrew's decision)

**Context change:** Andrew moved the benchmark to **per-algo precision** (matching torch / the
official fsrs-rs, which are f32): analytic/reverse-mode algos run f32, forward-mode-`Dual` algos run
f64 (see CLAUDE.md §6 — a global f32 broke the Dual algos' `--secs` training; `--features fp64`
forces all-f64). **FSRS-7 is an f32 algo** (its gradient is the hand-written reverse-mode
analytic/SIMD path), so this iter makes it genuinely f32. This *supersedes* the earlier note above
that "f32 SIMD trades accuracy beyond our gate" — that was an unverified presumption. FSRS-7's
forgetting curve is well-behaved (unlike HLR's chaotic `0.5^(t/s)`), so f32 moves it only ~2e-4,
comfortably inside ±0.0005.

**Change:** FSRS-7 scalar analytic path (`fsrs_v7_grad.rs`) now rounds every op to f32; the SIMD
gradient (`fsrs_v7_simd.rs`) ported **f64×4 → f32×8** (genuine f32, 8 rows/lane). Adam already
rounded via the global toggle. `keep_final` training unchanged.

**Correctness (200 users, vs FROZEN f64 baseline AND current Python; size exact everywhere):**
| config | vs frozen f64 | vs Python |
|---|---|---|
| default          | +0.000000 | +0.000000 |
| default-equalize | +0.000000 | +0.000000 |
| plain            | +0.000184 | +0.000288 (57 common, Python mid-regen) |
| recency          | +0.000227 | +0.000225 |
| recency-equalize | +0.000220 | +0.000220 |
All within ±0.0005 of BOTH references ⇒ the speedups still produce in-band LogLoss, AND the f32
build still reproduces Python. (The ~2e-4 is the genuine f32-vs-f64 gap, not a speedup artifact.)

**Speed (plain config, 200 users, 1 thread each, SIMULTANEOUS):** f32×8 (after) vs f64×4 champion
`script_6` (before). total time_ms **218847 → 119921 = ×1.82 faster**; median per-user ratio 0.5585
(×1.79); Wilcoxon one-sided (after<before) **p = 7.181e-35**. **ACCEPT.** The 8-wide f32 SIMD beats
the 4-wide f64 SIMD outright (more lanes + cheaper f32 ops). Cumulative training speedup vs the
original forward-mode baseline is now ≈ ×10.9 × 1.82 ≈ **×20**. New champion = `script_7.exe`.

**Build:** `RUSTFLAGS="-C target-cpu=native" cargo build --release` (f32×8 wants AVX/AVX2). Tests:
`cargo test` runs the f32 path (simd-vs-scalar, analytic-vs-Dual at f32 tol); `cargo test --features
fp64` runs the full math suite (finite-diff checks need f64).
