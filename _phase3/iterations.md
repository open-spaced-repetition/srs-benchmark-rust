# Phase-3 speedup iteration log (forward-mode-`Dual` f64 algos)

Phase-3 generalizes the FSRS-7 Phase-2 speedup work (see `_phase2/iterations.md`) to the **other
slow algorithms** — the ones whose training gradient comes from forward-mode `Dual<P>` autodiff
(`src/autodiff.rs`), which costs ~P× the value pass per op: **FSRS v1–v6, FSRS-4.5, FSRS-6-one-step,
ACT-R, Anki, DASH[ACT-R], SM2-trainable**. All run in f64 (per-algo precision; see CLAUDE.md).

Every accepted **or rejected** speedup is logged here. Required fields (Andrew): **timestamp,
iter #, LogLoss before, LogLoss after, avg time/user before, avg time/user after, the EXACT
Wilcoxon p-value** (never `<0.01`).

## Protocol (per Andrew, this phase)

1. **200 users.** "before" = current champion binary, "after" = candidate. They run
   **SIMULTANEOUSLY**, 1 thread each (2 total), started together so thermal/scheduling noise hits
   both equally (`_phase2/time2.sh`). Record **average AND median** speedup, LogLoss before+after,
   and the Wilcoxon signed-rank p-value on the 200 paired per-user `time_ms`. **ACCEPT iff p < 0.01**
   (and faster).
2. **Hand-written gradients are allowed**, but the algos that have them must be documented, and it
   must be clear that **changing the model formulas will break the hand-written gradient** (it is a
   manual VJP of a specific forward, not autodiff). See the "Hand-written gradients" table below.
3. **±0.0005 is PER-ALGO.** Each algo's candidate must stay within **±0.0005 of that algo's FROZEN
   baseline LogLoss** (captured once from the champion, stored in `_phase3/baseline/`), AND keep
   `size` exact, AND still match upstream. The frozen baseline (not the rolling champion) is the
   anchor, to prevent slow drift across many edits.

Frozen baselines: `_phase3/baseline/<config>.jsonl` (champion `script_p3base.exe`, 200 users).
Harness (shared with Phase-2): `_phase2/time2.sh` (simultaneous timing), `_phase2/wilcoxon_time.py`
(speed gate), `_phase2/compare_ref.py` (correctness vs an arbitrary ref jsonl).

## Hand-written (non-autodiff) gradients

Algos whose `grad` is a hand-derived reverse-mode VJP rather than forward-mode `Dual` autodiff.
**⚠ Editing the model's forward formulas requires re-deriving the matching backward by hand** — the
finite-difference / forward-mode-oracle unit tests (gated to `--features fp64`) guard this.

| Algo | gradient | file |
|------|----------|------|
| FSRS-7 | hand-written reverse-mode VJP (+ `f32x8` SIMD) | `src/models/fsrs_v7_grad.rs`, `fsrs_v7_simd.rs` |
| FSRS-6 | hand-written reverse-mode VJP (f64) | `src/models/fsrs_v6_grad.rs` |
| _(FSRS v1–v5, v4.5, ACT-R, Anki, DASH[ACT-R], SM2-trainable: still forward-mode `Dual` — candidates)_ | | |

## Iterations

| iter | timestamp | change | timing config | LogLoss before | LogLoss after | avg ms/user before | avg ms/user after | speedup (median / total) | Wilcoxon p | decision |
|------|-----------|--------|---------------|----------------|---------------|--------------------|-------------------|--------------------------|------------|----------|
| 1 | 2026-06-20 | strip `round_scalar` from `Dual<P>` ops (f64-only in production ⇒ no-op removed; un-blocks auto-vectorization of the const-`P` gradient loops) — **bit-identical** | FSRS-6 --short --secs (200u) | 0.388226 | 0.388226 | 4740.9 | 2186.8 | ×2.05 median / ×2.17 total | 7.181e-35 | **ACCEPT** |
| 2 | 2026-06-20 | **FSRS-6** hand-written reverse-mode VJP (`fsrs_v6_grad.rs`) replaces forward-mode `Dual<21>` for the training grad; predict keeps `Dual<0>`; f64 | FSRS-6 --short --secs (200u), before=iter1 | 0.388226 | 0.388225 | 2126.1 | 1326.7 | ×1.59 median / ×1.60 total | 7.181e-35 | **ACCEPT** |

## Iteration details

### iter 1 — strip `round_scalar` from `Dual` (bit-identical vectorization win)

- **Change:** `src/autodiff.rs` — the forward-mode `Dual<P>` ops called `round_scalar` (`r()`) on the
  value **and every element of the `[f64; P]` gradient array**, every op. `round_scalar` does a
  runtime atomic-load + branch on the per-algo `ROUND_F32` flag. But `Dual<P>` is used **in
  production only by the f64 algos** (FSRS v1–v6, v4.5, ACT-R, Anki, DASH[ACT-R], SM2-trainable),
  where `ROUND_F32` is always `false` ⇒ `r(x) == x`. (The only f32 algo that references `Dual`,
  FSRS-7, does so only in `#[cfg(feature="fp64")]` test oracles.) So the rounding was always a no-op
  here; removing it is **bit-identical** AND lets the const-`P` gradient loops auto-vectorize (the
  per-element atomic+branch had blocked SIMD). `round_scalar` itself stays (HLR/DASH/LogReg/FSRS-7
  analytic paths + Adam still use it).
- **Correctness:** **bit-identical** — verified `size` exact and max |ΔLogLoss| = 0.00 on every
  measured config; full unit suite green in both `cargo test` (f32) and `cargo test --features fp64`
  (the per-model finite-difference oracle checks). Trivially within ±0.0005 of every frozen baseline.
- **Speed (200 users, 1 thread each, SIMULTANEOUS):** FSRS-6 --short --secs total time_ms
  **948189 → 437358 = ×2.17**; median per-user ratio 0.4876 (×2.05); Wilcoxon one-sided
  (after<before) **p = 7.181e-35**. **ACCEPT.**
- **Breadth (bit-identical; 30-user smokes, same simultaneous method):** the change helps every
  forward-mode-`Dual` algo, scaling with the parameter count P (the vectorized loop length):
  - SM2-trainable --short --secs: ×1.44 (p = 9.313e-10).
  - ACT-R --short --secs: ×1.21 (p = 9.313e-10) — smaller because ACT-R's cost is dominated by its
    O(reviews²) all-pairs activation **value** sum, not the gradient P-loop (that's iter ≥? / task
    #11's separate algorithmic win).
- **Champion:** `target/release/script_p3_iter1.exe` (frozen-baseline binary = `script_p3base.exe`,
  bit-identical output, pre-vectorization speed).

### iter 2 — FSRS-6 hand-written reverse-mode VJP

- **Change:** new `src/models/fsrs_v6_grad.rs` — a manual forward (`step_fwd`/`curve_fwd`, stashing
  intermediates) + reverse-mode backward (`step_bwd`/`curve_bwd`/`init_bwd`, the adjoint of that
  exact forward) for the FSRS-6 BCE-loss gradient. `fsrs_v6::Model::grad` now calls `grad_one` per
  row instead of building a `Dual<21>`; `fsrs_v6::retention` is renamed `retention_dual` (kept as the
  prediction path via `Dual<0>` AND the gradient oracle). f64 throughout (FSRS-6 is an f64 algo).
  ⚠ It is a manual VJP of a specific forward — changing the FSRS-6 formulas requires re-deriving it.
- **Correctness:** the new `fsrs6_analytic_grad_matches_forward_mode` test (gated `--features fp64`)
  checks `grad_one` vs the `Dual<21>` oracle on 4 sequences (init + short/success/fail branches +
  clamps) to <1e-6; full suite green in both modes. **200-user vs the FROZEN baseline:** `size`
  exact, mean LogLoss 0.388226 → 0.388225 (**Δ = −0.000001**, max per-user 0.00027) — the
  reverse-mode and forward-mode compute the same f64 derivative (differing only ~1e-15 in summation
  order), so the trajectory barely moves. Still matches upstream (FSRS-6 --short --secs ≈ −0.00014).
- **Speed (200 users, 1 thread each, SIMULTANEOUS; before = iter-1 champion):** total time_ms
  **425216 → 265335 = ×1.60**; median per-user ratio 0.6297 (×1.59); Wilcoxon one-sided
  **p = 7.181e-35**. **ACCEPT.** Cumulative vs the original forward-mode baseline: ×2.17 (iter 1) ×
  ×1.60 ≈ **×3.5** on FSRS-6 training.
- **Champion:** `target/release/script_p3_iter2.exe`.
