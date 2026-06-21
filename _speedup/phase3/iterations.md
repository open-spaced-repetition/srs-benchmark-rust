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
| FSRS-5 | hand-written reverse-mode VJP (f64) | `src/models/fsrs_v5_grad.rs` |
| FSRS-4.5 | hand-written reverse-mode VJP (f64) | `src/models/fsrs_v4dot5_grad.rs` |
| FSRS-4 | hand-written reverse-mode VJP (f64) | `src/models/fsrs_v4_grad.rs` |
| FSRS-3 | hand-written reverse-mode VJP (f64) | `src/models/fsrs_v3_grad.rs` |
| FSRS-2 | hand-written reverse-mode VJP (f64) | `src/models/fsrs_v2_grad.rs` |
| FSRS-1 | hand-written reverse-mode VJP (f64) | `src/models/fsrs_v1_grad.rs` |
| SM2-trainable | hand-written reverse-mode VJP (f64) | `src/models/sm2_trainable_grad.rs` |
| DASH[ACT-R] | closed-form analytic gradient (f64, static sum) | `src/models/dash_act_r_grad.rs` |
| _(Anki: reverse-mode VJP REJECTED — slower than vectorized `Dual<7>`; stays forward-mode. ACT-R: still forward-mode `Dual`.)_ | | |

## Iterations

| iter | timestamp | change | timing config | LogLoss before | LogLoss after | avg ms/user before | avg ms/user after | speedup (median / total) | Wilcoxon p | decision |
|------|-----------|--------|---------------|----------------|---------------|--------------------|-------------------|--------------------------|------------|----------|
| 1 | 2026-06-20 | strip `round_scalar` from `Dual<P>` ops (f64-only in production ⇒ no-op removed; un-blocks auto-vectorization of the const-`P` gradient loops) — **bit-identical** | FSRS-6 --short --secs (200u) | 0.388226 | 0.388226 | 4740.9 | 2186.8 | ×2.05 median / ×2.17 total | 7.181e-35 | **ACCEPT** |
| 2 | 2026-06-20 | **FSRS-6** hand-written reverse-mode VJP (`fsrs_v6_grad.rs`) replaces forward-mode `Dual<21>` for the training grad; predict keeps `Dual<0>`; f64 | FSRS-6 --short --secs (200u), before=iter1 | 0.388226 | 0.388225 | 2126.1 | 1326.7 | ×1.59 median / ×1.60 total | 7.181e-35 | **ACCEPT** |
| 3 | 2026-06-20 | **FSRS-5** hand-written reverse-mode VJP (`fsrs_v5_grad.rs`) replaces forward-mode `Dual<19>`; predict keeps `Dual<0>`; f64 | FSRS-5 --short --secs (200u), before=iter2 | 0.458876 | 0.458928 | 1416.1 | 1087.2 | ×1.27 median / ×1.30 total | 1.297e-33 | **ACCEPT** |
| 4 | 2026-06-20 | **FSRS-4.5** reverse-mode VJP (`fsrs_v4dot5_grad.rs`) replaces forward-mode `Dual<17>`; f64 | FSRS-4.5 --short --secs (200u), before=iter3 | 0.428544 | 0.428537 | 1709.3 | 1238.5 | ×1.32 median / ×1.38 total | 7.860e-35 | **ACCEPT** |
| 5 | 2026-06-20 | **FSRS-4** reverse-mode VJP (`fsrs_v4_grad.rs`) replaces forward-mode `Dual<17>`; f64 | FSRSv4 --short --secs (200u), before=iter3 | 0.482724 | 0.482708 | 1152.4 | 863.9 | ×1.28 median / ×1.33 total | 1.821e-33 | **ACCEPT** |
| 6 | 2026-06-20 | **FSRS-3** reverse-mode VJP (`fsrs_v3_grad.rs`, `Dual<13>`); nd-before-ns, `0.9^(t/s)` curve | FSRSv3 --short --secs (200u) | 0.639411 | 0.639411 | 1232.1 | 970.9 | ×1.25 median / ×1.27 total | 2.142e-33 | **ACCEPT** |
| 7 | 2026-06-20 | **FSRS-2** reverse-mode VJP (`fsrs_v2_grad.rs`, `Dual<14>`) | FSRSv2 --short --secs (200u) | 0.657160 | 0.657145 | 1545.2 | 1159.9 | ×1.32 median / ×1.33 total | 9.415e-35 | **ACCEPT** |
| 8 | 2026-06-20 | **FSRS-1** reverse-mode VJP (`fsrs_v1_grad.rs`, `Dual<7>`); 3-state (lapse count) | FSRSv1 --short --secs (200u) | 0.725029 | 0.725029 | 1475.6 | 1322.1 | ×1.09 median / ×1.12 total | 5.234e-24 | **ACCEPT** |
| 9 | 2026-06-20 | **SM2-trainable** reverse-mode VJP (`sm2_trainable_grad.rs`, `Dual<6>`) | SM2-trainable --short --secs (200u) | 0.811821 | 0.811821 | 293.6 | 273.7 | ×1.10 median / ×1.07 total | 5.002e-11 | **ACCEPT** |
| 10 | 2026-06-20 | **DASH[ACT-R]** closed-form analytic gradient (`dash_act_r_grad.rs`, `Dual<5>`); static sum, not a recurrence | DASH[ACT-R] --short --secs (200u) | 0.382135 | 0.382135 | 907.4 | 760.6 | ×1.15 median / ×1.19 total | 2.551e-26 | **ACCEPT** |
| 11 | 2026-06-20 | **Anki** reverse-mode VJP (`Dual<7>`; max/leaky_relu/branch routing) | Anki --short --secs (30u smoke) | (bit-identical) | (bit-identical) | — | — | ×0.97 (slower) | **REJECT** (reverted) |

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

### iter 3 — FSRS-5 hand-written reverse-mode VJP

- **Change:** new `src/models/fsrs_v5_grad.rs` (FSRS-5 is FSRS-6 with a FIXED decay −0.5, no `w20`,
  and a simpler short-term branch `s·exp(w17·(w18+rating-3))`; success/fail/difficulty/init match).
  `fsrs_v5::Model::grad` calls it; `retention` → `retention_dual` (prediction + oracle). f64.
- **Correctness:** `fsrs5_analytic_grad_matches_forward_mode` (fp64) checks vs the `Dual<19>` oracle
  to <1e-6; suite green. 200-user vs FROZEN baseline: `size` exact, mean LogLoss 0.458876 → 0.458928
  (**Δ = +0.000053**, within ±0.0005; max per-user 0.018 — one chaotic user's training trajectory
  shifts on the ~1e-15 summation-order difference, mean unaffected). Still matches upstream
  (FSRS-5 --short --secs ≈ +0.0001).
- **Speed (200 users, simultaneous; before = iter-2 champion):** total time_ms **283212 → 217432 =
  ×1.30**; median per-user 0.7901 (×1.27); Wilcoxon **p = 1.297e-33**. **ACCEPT.**
- **Champion:** `target/release/script_p3_iter3.exe`.

### iters 4 & 5 — FSRS-4.5 and FSRS-4 reverse-mode VJPs

- **Change:** new `src/models/fsrs_v4dot5_grad.rs` and `fsrs_v4_grad.rs` (both NP=17). FSRS-4.5 = power
  curve `(1+factor·t/s)^-0.5` + `min(nf, old_s)` after-failure cap; FSRS-4 = `(1+t/9s)^-1` curve +
  uncapped `nf`; both: NO short-term branch, LINEAR difficulty reverting to `w4`, linear init, and
  `w0..3` frozen by the model `grad_mask` (the VJP computes the full gradient; masking is downstream
  and identical for both paths). `retention` → `retention_dual` in each; built into one candidate.
- **Correctness:** `fsrs45_/fsrs4_analytic_grad_matches_forward_mode` (fp64) vs the `Dual<17>` oracle
  to <1e-6; suite green. 200-user vs FROZEN baselines, `size` exact, still match upstream:
  - FSRS-4.5 --short --secs: 0.428544 → 0.428537 (**Δ = −0.000007**, max per-user 0.0022).
  - FSRSv4 --short --secs: 0.482724 → 0.482708 (**Δ = −0.000015**, max per-user 0.0017).
- **Speed (200 users, simultaneous; before = iter-3 champion):**
  - FSRS-4.5: 341853 → 247700 ms = **×1.38** (median ×1.32); **p = 7.860e-35**. **ACCEPT.**
  - FSRSv4: 230475 → 172779 ms = **×1.33** (median ×1.28); **p = 1.821e-33**. **ACCEPT.**
- **Champion:** `target/release/script_p3_iter5.exe` (FSRS-4/4.5/5/6 all reverse-mode VJP now).

### iters 6–10 — FSRS-3/2/1, SM2-trainable, DASH[ACT-R] reverse-mode gradients (ACCEPT)

- **Change:** reverse-mode VJP modules for the remaining lower-NP f64 Dual algos — `fsrs_v3_grad.rs`
  (NP=13, nd-before-ns, `0.9^(t/s)`), `fsrs_v2_grad.rs` (NP=14), `fsrs_v1_grad.rs` (NP=7, 3-state
  with a data-driven lapse count), `sm2_trainable_grad.rs` (NP=6, interval/ease machine), and
  `dash_act_r_grad.rs` (NP=5, a **closed-form** analytic gradient — DASH[ACT-R] is a static sum, not
  a recurrence, so the sigmoid+BCE seed simplifies to `weight·(p-y)`). Each model's `retention` →
  `retention_dual` (kept as `Dual<0>` predict + the gradient oracle). f64.
- **Correctness:** each has a `*_analytic_grad_matches_forward_mode` test (fp64) vs the `Dual` oracle
  to <1e-6; full suite green (33 fp64 / 9 f32). 200-user vs FROZEN baselines, `size` exact, all still
  match upstream: FSRS-3 Δ=+0.000000, FSRS-2 Δ=−0.000015, FSRS-1 Δ=+0.000000, SM2 Δ=+0.000000,
  DASH[ACT-R] Δ=+0.000000.
- **Speed (200 users, simultaneous):** FSRS-3 ×1.27 (p=2.142e-33), FSRS-2 ×1.33 (p=9.415e-35),
  FSRS-1 ×1.12 (p=5.234e-24), SM2 ×1.07 (p=5.002e-11), DASH[ACT-R] ×1.19 (p=2.551e-26). All ACCEPT.
  The wins shrink with the parameter count P (the vectorized forward-mode loop is short at low P).

### iter 11 — Anki reverse-mode VJP (REJECT)

- A hand-written reverse-mode VJP for Anki (NP=7; `max`/`leaky_relu`/early-vs-non-early branch
  routing). The VJP was **correct** (oracle-matched <1e-6, output bit-identical), but at NP=7 the
  fat per-step routing cache + branch logic made it **×0.97 (slower)** than the already-vectorized
  forward-mode `Dual<7>` (30-user smoke median ratio 1.0283). Per the speed gate it was **REJECTED**
  and reverted — Anki stays on forward-mode `Dual`. (Lesson: below ~NP≈10 with heavy branch routing,
  reverse-mode's bookkeeping can cost more than the forward-mode P-loop it removes.)
- **Champion (end of batch):** `target/release/script_p3_iter10.exe` (= v1–v6 + SM2 + DASH[ACT-R]
  reverse-mode; Anki/ACT-R forward-mode).

## ACT-R — per-card recurrence reuse (2026-06-21)

**Bottleneck:** ACT-R's `retention` recomputed the full O(pos²) base-level-activation recurrence
`m[i]` from scratch for EVERY row, and a card with N reviews has N rows (pos 0..N-1) → effectively
**O(N³) per card** (N capped at 2·max_seq_len = 128). `m[i]` is a prefix shared by all of a card's rows.

**Change (`src/models/act_r.rs`):** split `retention` into `recurrence` (returns all `m[i]` for the
card prefix) + `ret_from_m` (per-row sigmoid); new `Model::retentions` groups the idx rows by
`card_idx` and runs the recurrence ONCE per card (up to the deepest pos needed), reused by every row
→ **O(N²) per card**. predict/grad use it; `retention` kept as a thin wrapper for the grad unit test.

**Speed (ACT-R --short --secs, 200 users, 1 thread each, SIMULTANEOUS, CPU locked at base):** total
time_ms 1295820 → 787312 = **×1.65 faster**; median per-user ratio 0.650 (×1.54); Wilcoxon one-sided
(after<before) **p = 8.308e-30**. **Correctness:** size exact, mean dLogLoss **+0.00000000**, max|d|
0.0 (bit-identical — same math). **ACCEPT.** Applies to all 3 ACT-R configs (still forward-mode Dual<5>).

## ACT-R — `Dual::powd` value reuse (2026-06-21)

**Bottleneck:** `Dual::powd` (`src/autodiff.rs`) computed `self^e` with TWO `powf` calls — one for
the value `v = self^e` and a SECOND for the self-derivative `da = e·self^(e-1)` — plus a `ln`. That
is ~5 transcendentals per call. ACT-R's base-level-activation recurrence is an O(N²)-per-card sum of
`a.powd(exponent[j])`, so `powd` dominates its gradient.

**Change (`src/autodiff.rs::powd`):** reuse the already-computed value — `da = e·self^(e-1) = e·v/self`
— instead of the second `powf`. The value `v = self.v.powf(e.v)` is kept EXACT (so every prediction
is bit-identical; for P=0 predict, `da`/`de`/`ln` are dead and elide to the single `powf`). Only the
gradient's `da` term changes, by f64 rounding (~1e-16). Helps every forward-mode-`Dual` grad user
(ACT-R, Anki, FSRS-6-one-step + the oracle tests); the VJP'd FSRS algos hand-write their gradients so
are untouched. powd: 2·powf + ln  →  1·powf + ln + a div.

**Speed (ACT-R --short --secs, 200 users, 1 thread each, SIMULTANEOUS, CPU locked at base):** total
time_ms 750170 → 554267 = **×1.35 faster**; median per-user ratio 0.8447 (×1.18); the total beats the
median because the win scales with reviews/card (more O(N²) powd). Wilcoxon one-sided (after<before)
**p = 2.965e-30**. **Correctness:** size exact (sum 9847205), mean dLogLoss **+0.00000000**, max|d|
0.0 (bit-identical at the 6-dp reported precision). Full suite green (f32 19 / fp64 43). **ACCEPT.**
Cumulative ACT-R from the original O(N³): 1295820 → 787312 (per-card) → 554267 (powd) = ×2.34.

## FSRS v1–v6 — per-card predict (shared recurrence reuse) (2026-06-21)

**Bottleneck:** every FSRS version's `predict` (and the per-epoch best-weights `eval_loss`, which calls
`predict` over ALL rows) replayed the full stability recurrence FROM SCRATCH for each row —
`retention_dual(prior_dt[..pos], prior_r[..pos], …)` loops `0..pos`. A card with N reviews has ~N
rows at pos 1..N, so eval/test predict was **O(N²) per card** (the same trap ACT-R had, never fixed
for FSRS). The state after k reviews is a shared prefix; a row at pos p only needs the stability AFTER
p reviews.

**Change (`fsrs_v1..v6` + `fsrs_v4dot5`):** extracted the recurrence loop into `forward_states<P>`
returning the post-clamp stability `s` after EVERY review (difficulty `d`, and v1's lapse `l`, stay
internal); `retention_dual` now = `forward_states` + final `fc` (bit-identical). `Model::predict`
groups the requested rows by card, runs `forward_states` ONCE per card up to the deepest pos, and each
row at pos p reads `states[p-1]` → **O(N) per card**. Bit-identical (same ops, just not recomputed).

**Scope note — predict only, not grad:** `eval_loss` and the final test predict call `predict` over
ALL rows at once, so per-card grouping fully applies. `grad` runs per-BATCH and batches are sorted by
seq_len (= pos), so a card's rows land in different batches — per-card sharing wouldn't help grad (and
grad's O(N²) matches Python's batched structure). So this targets the eval+test predict fraction only.

**Correctness:** full suite green (f32 19 / fp64 43 — the `*_grad_matches_*` oracles call the refactored
`retention_dual`). 150–200-user new-vs-powd-binary check, ALL bit-identical: v1/v2/v3/v4/v4.5/v5/v6 all
size exact, mean dLogLoss +0.00000000, max|d| 0.0.

**Speed (--short --secs, 200 users, 1 thread each, SIMULTANEOUS, CPU locked at base):**
- **FSRS-6**: total time_ms 276209 → 203495 = **×1.36** (median 0.8412); Wilcoxon p=6.458e-24. ACCEPT.
- **FSRS-5**: total time_ms 228276 → 179641 = **×1.27** (median 0.8818); Wilcoxon p=1.867e-14. ACCEPT.
Applies to ALL FSRS configs (v1–v6 + every FSRS-6 variant + FSRS-6-one-step, which predicts via
`fsrs_v6::predict`). The win = the eval+test predict fraction collapsing from O(N²) to O(N).

## Anki + SM2-trainable — per-card predict (REJECT)

Tried the same per-card predict (forward_states + group-by-card) on Anki and SM2-trainable. Output
bit-identical (size exact, dLogLoss 0.0), but **both got SLOWER**: Anki ×0.72 (median ratio 1.576,
one-sided after<before p=1.000), SM2-trainable ×0.75 (median 1.371, p=1.000). **REJECTED & reverted.**
Reason: their per-step recurrence is CHEAP (interval/ease/EF arithmetic, no `powf` in the loop — the
single `powd` is only the final forgetting curve), so the O(N²)→O(N) saving on cheap work is tiny,
while the per-card grouping bookkeeping (a `HashMap<card,Vec>` + a per-card `Vec` allocation in EVERY
`eval_loss` call, ×6/training over all rows) costs more. Lesson: per-card predict only pays off when
each recurrence STEP is expensive (FSRS's `powf`/`powd`); for cheap recurrences the grouping overhead
dominates. (Mirrors the earlier Anki-VJP reject: at low cost-per-step, the bookkeeping isn't worth it.)

## ACT-R — reuse ln(a) in the inner power (a.powd(e) -> a.ln().mul(e).exp()) (2026-06-21)

**Bottleneck:** ACT-R's O(N²)-per-card activation sum computes `a.powd(exponent[j])` per pair. Even
after the powd value-reuse fix, `powd` does a `powf` (itself ln+exp internally) PLUS a separate `ln`
for the exponent-derivative `de = v·ln(a)` — 2 ln + 1 exp = 3 transcendentals per pair. ACT-R's cost
is dominated by these transcendentals (not the gradient arithmetic — so a reverse-mode VJP would NOT
help: it does the same number of powf/ln).

**Change (`src/models/act_r.rs`):** compute `a^exponent[j]` as `a.ln().mul(exponent[j]).exp()` —
`ln(a)` is computed ONCE (as a `Dual`) and reused for both the value and the gradient → 1 ln + 1 exp
= 2 transcendentals per pair. The value becomes `exp(e·ln a)` instead of `powf(a,e)` (same math, ~1
ULP different); `a` clamped to 1 ⇒ ln a = 0 ⇒ term = 1 (zero grad), matching the clamp.

**Speed (ACT-R --short --secs, 200 users, 1 thread each, SIMULTANEOUS, CPU locked):** total time_ms
572927 → 370680 = **×1.55 faster**; median ratio 0.7931 (×1.26); Wilcoxon one-sided p=6.467e-34.
**Correctness:** size exact, mean dLogLoss **+0.00000000**, max|d| 0.0 (the ~1-ULP value shift didn't
move LogLoss at the 6-dp reported precision). Full suite green (f32 19 / fp64 43). **ACCEPT.**
Cumulative ACT-R from the original O(N³): 1295820 → 787312 (per-card) → 554267 (powd) → 370680 = ×3.5.
Lesson: ACT-R is transcendental-bound; cutting a `ln`/`exp` per inner pair beats any gradient-arithmetic
trick (VJP). The remaining cost is 1 ln + 1 exp per pair (irreducible without changing the model math).
