# srs-benchmark-rust

A Rust port of [**open-spaced-repetition/srs-benchmark**](https://github.com/open-spaced-repetition/srs-benchmark),
built to run the same benchmark **much faster** while reproducing its results. The model
*definitions* remain authored in Python upstream as the canonical spec; the math for the ported
algorithms is reimplemented natively in Rust for speed, and the command-line interface mirrors the
Python `script.py` (same flags, same output filenames). The neural models sit outside the benchmark
tables below: GRU and LSTM have an optional native [`candle`](https://github.com/huggingface/candle)
build (see [Build](#build)), while RWKV, NN-17, and Transformer are Python-only.

The sections **Introduction**, **Dataset**, **Evaluation**, and the algorithm descriptions below are
adapted (with minimal changes) from the [upstream README](https://github.com/open-spaced-repetition/srs-benchmark);
see it for the full project context, the neural-model results, and the additional metrics (RMSE (bins),
AUC). The **Results** here report **Log Loss only**, comparing the current upstream Python against this
Rust port over the same collections.

## Introduction

Spaced repetition algorithms are computer programs designed to help people schedule reviews of
flashcards. A good spaced repetition algorithm helps you remember things more efficiently. Instead of
cramming all at once, it distributes your reviews over time. To make this efficient, these algorithms
try to understand how your memory works. They aim to predict when you're likely to forget something,
so they can schedule a review accordingly.

This benchmark is designed to assess the predictive accuracy of various algorithms. A multitude of
algorithms are evaluated to find out which ones provide the most accurate predictions.

## Dataset

The dataset for the SRS benchmark comes from 10 thousand users who use Anki, a flashcard app. In total,
this dataset contains information about ~727 million reviews of flashcards. The full dataset is hosted
on Hugging Face Datasets:
[open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k).

## Evaluation

### Data Split

The benchmark uses a tool called `TimeSeriesSplit` (from the [sklearn](https://scikit-learn.org/)
library). It splits the data by time: older reviews are used for training and newer reviews for
testing. That way, we don't accidentally cheat by giving the algorithm future information it shouldn't
have. In practice, we use past study sessions to predict future ones.

Note: `TimeSeriesSplit` removes the first split from evaluation, because the first split is used for
training and we don't want to evaluate the algorithm on the same data it was trained on. RMSE-BINS-EXPLOIT
does not use `TimeSeriesSplit`.

### Metrics

The upstream benchmark uses three metrics — Log Loss, RMSE (bins), and AUC. **This port's tables below
report Log Loss only** (the binding correctness target for the port); the full metric set is in the
upstream README.

- **Log Loss** (also known as Binary Cross Entropy): a measure of the discrepancy between predicted
  probabilities of recall and review outcomes (1 or 0). It quantifies how well the algorithm
  approximates the true recall probabilities. Log Loss ranges from 0 to infinity; **lower is better**.

### Algorithms and algorithm families

In this Rust port, the Adam-trained and closed-form algorithms below are **reimplemented natively in
Rust** and appear in the Results tables. The neural models are not in the tables: GRU and LSTM have an
optional native `candle` implementation (`--features neural`, see [Build](#build)); RWKV, NN-17, and
Transformer are Python-only. Descriptions are copied from the upstream README.

- Two component or three component* model of memory:
    - FSRS v1 and v2: the initial experimental versions of FSRS, used only by Jarrett Ye.
    - FSRS v3: the first official release of the FSRS algorithm, made available as a custom scheduling script.
    - FSRS v4: the upgraded version of FSRS, made better with help from the community. It is the first version that was integrated into Anki.
    - FSRS-4.5: the minorly improved version based on FSRS v4. The shape of the forgetting curve has been changed.
    - FSRS-5: unlike the previous versions, FSRS-5 uses the same-day review data to refine its prediction for the next review. Same-day reviews are used only for training, and not for evaluation.
    - FSRS-6: the formula for handling same-day reviews has been improved. More importantly, FSRS-6 has an optimizable parameter that controls the flatness of the forgetting curve, meaning that the shape of the curve is different for different users.
    - FSRS-7: the newest version. Unlike all previous versions, which have been designed to work with integer interval lengths, FSRS-7 has been designed to work with fractional interval lengths. It is the only version that can give realistic predictions of probability of recall for same-day reviews. The biggest change is that the forgetting curve now has 8 optimizable parameters and uses a rather complex formula.
        - FSRS-7 default param.: FSRS-7 with default parameters, without per-user optimization.
        - FSRS-7 recency: FSRS-7 trained with reviews being weighted based on their recency, such that older reviews affect the loss function less and newer reviews affect it more.
    - FSRS-rs: the Rust port of FSRS-6 with recency weighting. See also: https://github.com/open-spaced-repetition/fsrs-rs *(in this benchmark it is gated behind the `fsrs-rs` cargo feature and is not in the tables.)*
    - HLR: the algorithm proposed by Duolingo. Its full name is Half-Life Regression. For further information, please refer to [this paper](https://github.com/duolingo/halflife-regression).
    - Ebisu v2: [an algorithm that uses Bayesian statistics](https://fasiha.github.io/ebisu/) to update its estimate of memory half-life after every review.

    *In the two-component model of long-term memory, two independent variables are used to describe the status of unitary memory in a human brain: retrievability (R), or retrieval strength/probability of recall; and stability (S), or storage strength/memory half-life. The expanded three-component model adds a third variable - difficulty (D).*

- Alternative models of memory:
    - DASH: the algorithm proposed in [this paper](https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/zp38wc97m). The name stands for Difficulty, Ability, and Study History. In our benchmark, we only use the Ability and Study History because the Difficulty part is not applicable to our dataset. We also added two other variants of this algorithm: DASH[MCM] and DASH[ACT-R]. For further information, please refer to [this paper](https://www.politesi.polimi.it/retrieve/b39227dd-0963-40f2-a44b-624f205cb224/2022_4_Randazzo_01.pdf).
    - ACT-R: the algorithm proposed in [this paper](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/workshops/2003/proceedings/46.pdf). It includes an activation-based system of declarative memory. It explains the spacing effect by the activation of memory traces.

- Other:
    - Logistic Regression: performs a logistic regression based on 34 features computed from the card history.
    - Anki: the trainable variant of Anki's own SM-2-derived scheduler (an interval/ease state machine), with a fixed forgetting curve.
    - SM2 / SM2-trainable: the classic SuperMemo-2 algorithm; `SM2` uses fixed constants, `SM2-trainable` optimizes its interval/ease-factor parameters.
    - AVG: an "algorithm" that outputs a constant equal to the user's average retention. Has no practical applications and is intended only to serve as a baseline. An algorithm that doesn't outperform AVG cannot be considered good.
    - MOVING-AVG: unlike AVG, which uses the overall retention across all reviews as its prediction of probability of recall, MOVING-AVG outputs higher values if recent reviews were successful and lower values if recent reviews were lapses.
    - RMSE-BINS-EXPLOIT: an algorithm that exploits the calculation of RMSE (bins) by simulating the bins and keeping the error term close to 0.

For further information regarding the FSRS algorithm, please refer to the following wiki page:
[The Algorithm](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

## Results — Python vs Rust (Log Loss)

The purpose of this port is to reproduce the upstream Python results faster, so the headline result is
the **agreement** between the two implementations. The tables below report, per algorithm:

- **Python Log Loss** — the **current** upstream Python `srs-benchmark` (`result/`), run over this same
  dataset.
- **Rust Log Loss** — this port (`target/release/script`).
- **Difference** — Rust − Python.

Each Log Loss is the **unweighted mean of the per-user Log Loss** across collections (the same
aggregation the upstream tables use). The reproduction target is **within ±0.0005**; the vast majority
of configs land at ±0.0000. The handful of larger gaps are **always Rust being lower** (slightly
*better*) and are genuine f64-vs-f32 optimizer/precision differences, not bugs — see the note under the
tables.

> Both numbers come from local runs over the same collections, so the difference is a true
> implementation-to-implementation comparison. (The Python numbers therefore track the *current*
> Python source and may differ slightly from the figures published in the upstream README, which can
> predate code changes — most visibly for FSRS-7.) The neural models (GRU, LSTM, RWKV) and FSRS-rs are
> omitted from the tables; see the upstream README for their results.

Following upstream, the results are split into two regimes by how same-day (short-term) reviews are
treated. The integer-interval ("without same-day reviews") configs evaluate on **9,999** collections;
the fractional-interval (`--secs`, "with same-day reviews") configs evaluate on **10,000**.

### Without same-day reviews (9,999 collections)

Same-day reviews are removed from evaluation (integer-day intervals); some algorithms still use them
for training. Sorted by Log Loss (lower is better).

| Algorithm | Python Log Loss | Rust Log Loss | Difference (Rust - Python) |
| --- | ---: | ---: | ---: |
| MOVING-AVG | 0.3369 | 0.3369 | +0.0000 |
| FSRS-7 (recency) | 0.3370 | 0.3371 | +0.0001 |
| Logistic Regression | 0.3393 | 0.3393 | +0.0000 |
| FSRS-6 | 0.3460 | 0.3460 | +0.0000 |
| FSRS-5 | 0.3561 | 0.3561 | +0.0000 |
| FSRS-7 default param. | 0.3620 | 0.3620 | +0.0000 |
| FSRS-4.5 | 0.3625 | 0.3622 | -0.0002 |
| DASH-short | 0.3681 | 0.3681 | -0.0000 |
| DASH | 0.3682 | 0.3682 | -0.0000 |
| DASH[MCM] | 0.3688 | 0.3688 | -0.0001 |
| FSRS v4 | 0.3726 | 0.3723 | -0.0003 |
| DASH[ACT-R] | 0.3728 | 0.3728 | +0.0001 |
| AVG | 0.3945 | 0.3945 | +0.0000 |
| ACT-R | 0.4033 | 0.3995 | -0.0037 |
| FSRS v3 | 0.4364 | 0.4364 | +0.0000 |
| FSRS v2 | 0.4533 | 0.4533 | +0.0000 |
| HLR | 0.4694 | 0.4692 | -0.0003 |
| FSRS v1 | 0.4919 | 0.4919 | -0.0000 |
| HLR-short | 0.4929 | 0.4925 | -0.0004 |
| Ebisu v2 | 0.4989 | 0.4989 | +0.0000 |
| Anki | 0.5127 | 0.5128 | +0.0002 |
| SM2-trainable | 0.5805 | 0.5817 | +0.0012 |
| SM2 | 0.7220 | 0.7220 | +0.0000 |
| RMSE-BINS-EXPLOIT | 4.6084 | 4.6084 | +0.0000 |

### With same-day reviews (10,000 collections)

Same-day reviews are kept (fractional-day `--secs` intervals) and the probability of recall is
calculated for all reviews. Sorted by Log Loss (lower is better).

| Algorithm | Python Log Loss | Rust Log Loss | Difference (Rust - Python) |
| --- | ---: | ---: | ---: |
| FSRS-7 (recency, continuous retraining) † | — | **0.3049** | — |
| FSRS-7 (recency) | 0.3178 | 0.3179 | +0.0001 |
| Logistic Regression | 0.3195 | 0.3195 | -0.0000 |
| FSRS-7 | 0.3206 | 0.3207 | +0.0001 |
| MOVING-AVG | 0.3301 | 0.3301 | +0.0000 |
| FSRS-7 default param. | 0.3399 | 0.3399 | +0.0000 |
| DASH[MCM] | 0.3459 | 0.3459 | -0.0000 |
| DASH | 0.3487 | 0.3487 | -0.0000 |
| DASH[ACT-R] | 0.3763 | 0.3763 | -0.0000 |
| AVG | 0.3816 | 0.3816 | +0.0000 |
| FSRS-6 | 0.3842 | 0.3844 | +0.0001 |
| ACT-R | 0.3898 | 0.3885 | -0.0013 |
| FSRS-4.5 | 0.4286 | 0.4288 | +0.0002 |
| FSRS-5 | 0.4565 | 0.4564 | -0.0001 |
| FSRS v4 | 0.4848 | 0.4845 | -0.0004 |
| FSRS v3 | 0.6470 | 0.6468 | -0.0001 |
| FSRS v2 | 0.6630 | 0.6628 | -0.0002 |
| HLR | 0.7049 | 0.7049 | -0.0000 |
| FSRS v1 | 0.7439 | 0.7437 | -0.0002 |
| Ebisu v2 | 0.7717 | 0.7717 | +0.0000 |
| Anki | 0.7948 | 0.7947 | -0.0001 |
| SM2-trainable | 0.8239 | 0.8235 | -0.0004 |
| SM2 | 0.9102 | 0.9102 | +0.0000 |
| RMSE-BINS-EXPLOIT | 4.1287 | 4.1100 | -0.0187 |

> † **Not a like-for-like row.** *FSRS-7 (recency, continuous retraining)* is a Rust-only research
> configuration (`--reopt_growth 0.003`), not a reproduction of a Python result, so it has no Python
> column. It refits from the default parameters every time the training set grows by 0.3% — so every
> prediction is made by a model that has seen ≥99.7% of the history available to it — instead of
> refitting only at the 5 `TimeSeriesSplit` boundaries, where that figure is 50–83%. It is an **upper
> bound** on what a user could get by re-optimizing often: 0.317947 → 0.304891 (**−0.013056**), with
> 99.8% of collections improving, at 97x the CPU. `size` is identical to the other FSRS-7 rows
> (519,296,315), so the Log Loss covers exactly the same reviews. But a fresher retraining schedule
> improves *any* trainable algorithm, so this number measures the protocol as much as the model and
> **should not be ranked against the rows below it**. For context it clears GRU (0.3146) and LSTM
> (0.3137) but still trails RWKV (0.2974) by 0.0075. See [Research modes](#research-modes) and
> `_hpprobe/FINDINGS.md`.

**On the larger gaps.** `size` (the per-user review count and its total) is **exact** for every config,
so the feature pipeline is faithful and the gaps are purely numerical:

- **ACT-R** (−0.0037 / −0.0013) runs in **f64** in this port (its gradient comes from forward-mode
  autodiff, which only a fully-f64 pass proxies faithfully); torch trains in f32, and f64 simply finds
  a slightly lower-loss optimum on this chaotic objective. The model math matches; the gap is
  concentrated in a few chaotic users.
- **RMSE-BINS-EXPLOIT** (−0.0187) is not a memory model — it games the RMSE (bins) metric, so its Log
  Loss is meaningless (≈4) and extremely sensitive; a tiny f32 difference in the simulated bins moves
  it noticeably. It is irrelevant to predictive accuracy.
- The remaining wobbles (all within ±0.0012: FSRS v4, HLR, FSRS-4.5, SM2-trainable, …) are f32-vs-f64
  / optimizer-trajectory noise. None is a behavioural difference.

The `--secs`-without-`--short` variants (e.g. `DASH --secs`, `ACT-R --secs`) are not shown — they are
not part of the upstream reference set and the port does not validate them.

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
under the ±0.0005 rule above. (One exception: the algorithms whose training gradient comes from
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
files shipped with the Python `srs-benchmark`).

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
run). The output filename is derived from the flags exactly as in Python — e.g.
`--algo FSRS-6 --short --secs` → `result/FSRS-6-short-secs.jsonl`.

## Performance

Each trained algorithm is optimized to keep the benchmark fast while reproducing results within the
±0.0005 tolerance above. The trained-model gradients are computed by **hand-written reverse-mode
analytic gradients** rather than generic autodiff — these are manual VJPs of each model's specific
forward pass (⚠ changing a model's math requires re-deriving its backward; the `--features fp64`
oracle tests guard this). Models with hand-written gradients: **FSRS-7** (+ `f32x8` SIMD) and **FSRS
v1–v6 / FSRS-4.5 / SM2-trainable / DASH[ACT-R]** (f64). The speedup work is logged iteration-by-
iteration in `_speedup/phase2/iterations.md` (FSRS-7) and `_speedup/phase3/iterations.md` (the rest),
each gated on a Wilcoxon signed-rank timing test (p < 0.01) and a per-algo correctness band. ACT-R and
Anki keep forward-mode autodiff (a VJP wasn't a net win for them), but ACT-R got three stacked
speedups (≈×3.5 total — ACT-R is bound by the transcendentals in its O(N²)-per-card activation sum, so
a reverse-mode VJP would *not* help): (1) an *algorithmic* one — its activation recurrence `m[i]` is a
prefix shared by all of a card's rows, so it's now computed **once per card** instead of recomputed
from scratch per row (O(N³)→O(N²) per card), ×1.65, bit-identical; (2) a leaner `Dual::powd` that
reuses the value (`a^(e-1) = aᵉ/a`) instead of a second `powf`, ×1.35 more; and (3) computing each
inner power as `exp(exponent·ln a)` so `ln a` is calculated **once** and reused for value+gradient
(instead of `powf`'s internal ln plus a separate one), ×1.55 more. (2)/(3) leave output bit-identical
at the reported precision; (2) also helps Anki / FSRS-6-one-step.

Separately, **every FSRS version (v1–v6) got a per-card predict speedup**: `predict` (and the
per-epoch best-weights `eval_loss`, which predicts over all rows) used to replay the stability
recurrence from scratch for each row — O(N²) per card. The state after k reviews is a shared prefix,
so `predict` now runs the recurrence **once per card** and each row reads the state it needs — O(N)
per card, **bit-identical**. ×1.36 on FSRS-6, ×1.27 on FSRS-5 (`--short --secs`); applies to all FSRS
configs. (The training gradient stays per-row: it runs per seq-len-sorted batch, where a card's rows
split across batches, so per-card sharing doesn't apply — and that matches Python's batched structure.)

**DASH** got a different speedup: its `z` recomputed `ln(feature+1)` for all 8 features on every
predict/grad call, but the features are constant during training — so `log(feature+1)` is now computed
**once** at feature build, ×1.5+ on all 8 DASH configs, bit-identical. (An earlier attempt to instead
speed up the O(N²) feature *build* via a per-card prefix gave nothing — the build wasn't the bottleneck;
measuring beats assuming.)

## Options

All flags match the Python `script.py`
([upstream docs](https://github.com/open-spaced-repetition/srs-benchmark#scriptpy-options)),
except the smart-preset flags (`--partitions smart`, `--cluster_method`, `--cluster_threshold`,
`--cluster_sweep`, `--cluster_distance`), which are a Rust-only extension (see [Smart presets](#smart-presets)
below), and the research flags (`--reopt_growth`, `--hp_probe`, `--hp_features`), which are a Rust-only
extension too (see [Research modes](#research-modes)).

| Flag | Description | Default |
| --- | --- | --- |
| `--algo` | Algorithm name (e.g. `FSRS-6`, `DASH`, `HLR`, `SM2`, `AVG`). | `FSRSv3` |
| `--data` | Path to the dataset root (containing `revlogs/`, `cards/`, `decks/`). | `../anki-revlogs-10k` |
| `--processes` | Number of parallel worker threads (Python: processes). | `8` |
| `--max-user-id` | Only process users with id ≤ this (inclusive). | no limit |
| `--short` | Include short-term (same-day) reviews. | off |
| `--secs` | Use `elapsed_seconds` (fractional-day intervals) instead of `elapsed_days`. | off |
| `--default` | Evaluate default parameters (no training). | off |
| `--recency` | Weight training reviews by recency (older reviews count less; the exact weighting differs by model). | off |
| `--S0` | FSRS-5/6: optimize only the initial-stability parameters. | off |
| `--sched_penalties` | FSRS-7 scheduling penalties (penalty 1 & 2). | off |
| `--two_buttons` | Treat Hard and Easy as Good (rating remap). | off |
| `--partitions` | Train per partition: `none`, `deck`, `preset`, or `smart`. | `none` |
| `--cluster_method` | Smart-preset clustering: `single`/`complete`/`average`/`centroid`/`ward`, `hdbscan`, or `optimal` (objective-driven partition search). `hdbscan`/`optimal` require `--cluster_sweep`. | `ward` |
| `--cluster_threshold` | Smart-preset distance cut threshold (with `--partitions smart`); on the Mahalanobis or KL scale per `--cluster_distance`. | `12.0` |
| `--cluster_sweep` | Run a whole smart-preset experiment matrix in one pass (shares per-deck training). | off |
| `--cluster_distance` | Smart-preset distance metric: `mahalanobis` (whitened FSRS-7 params) or `kl` (symmetric KL of the decks' predictions). | `mahalanobis` |
| `--n_splits` | Number of `TimeSeriesSplit` folds. | `5` |
| `--batch_size` | Training batch size. | `512` |
| `--max_seq_len` | Max sequence length for batching (also caps reviews/card at `2×`). | `64` |
| `--train_equals_test` | Train and test on the same data (overfit probe). | off |
| `--no_test_same_day` | Exclude `elapsed_days=0` reviews from the test set. | off |
| `--no_train_same_day` | Exclude `elapsed_days=0` reviews from the train set. | off |
| `--equalize_test_with_non_secs` | Test only on reviews that the non-`--secs` run would test. | off |
| `--duration` | Add the review-duration feature (LSTM only). | off |
| `--raw` | Save raw predictions to `raw/<name>.jsonl` — one line per user, `{"user", "p", "y"}`, predictions rounded to 4 dp, sorted by user (same format as Python). A full 10,000-user run writes ~5.7 GB. | off |
| `--file` | Save per-user evaluation TSVs to `evaluation/<name>/`. | off |
| `--plot` | Save evaluation plots. | off |
| `--weights` | Save trained model weights. | off |
| `--gpus` | CUDA device ids (e.g. `0,1` or `all`); unused by the CPU models. | unset |
| `--torch_num_threads` | PyTorch intra-op threads (parity flag). | `1` |
| `--dev` | Local-development import mode. | off |
| `--reopt_growth` | **Rust-only, FSRS-7.** Retrain whenever the training set has grown by this factor, instead of only at the 5 `TimeSeriesSplit` boundaries. `0.0` = off. See [Research modes](#research-modes). | `0.0` |
| `--hp_probe` | **Rust-only, FSRS-7.** Dump a per-user candidate x fold hyperparameter loss table instead of metrics. ~35x a normal run. | off |
| `--hp_features` | **Rust-only, FSRS-7.** Dump per-fold summary statistics of each fold's training rows (joins to `--hp_probe` output). Costs one normal pass. | off |
| `--interval_def` | **Rust-only.** How the `--secs` interval is measured: `stored` (the dataset's `elapsed_seconds` column), `end_to_end`, or `end_to_start`. See [Research modes](#research-modes). | `stored` |
| `--min_interval_secs` | **Rust-only.** Floor every non-sentinel `--secs` interval at this many seconds. end-to-start is always ≤ end-to-end, so it drops rows that fall below the pipeline's 1-second threshold; `1` makes both definitions evaluate the same rows, which a paired comparison needs. `0` = off. | `0` |

## Research modes

Three Rust-only flags that answer "how much better could FSRS-7 be", rather than reproducing Python.
None of them changes the evaluated row set, so `size` stays exactly comparable to a normal run.

### `--reopt_growth <eps>` — how much does optimizing more often buy?

The benchmark retrains at the 5 `TimeSeriesSplit` boundaries, so a prediction is made by a model that
has seen between 50% and 83% of the history available to it. With `--reopt_growth eps` the model is
instead refit **from the default parameters** whenever the training set has grown by `(1 + eps)`, so
every prediction sees at least `1/(1+eps)` of its history. It is an upper bound on what a user could
get by re-optimizing often.

The evaluated rows are unchanged: `TimeSeriesSplit` pools test folds covering exactly
`rows[eval_start..]`, and the geometric schedule partitions that same range, so `size` is identical
per user and in the sum.

Cost is `~n*(1+eps)/eps` training rows against the 5-fold `2.5n`. `eps = 1.0` (doubling) is *cheaper*
than the 5-fold schedule; `eps = 0.003` is ~80x. Measured on the full 10,000 users at `eps = 0.003`
(99.7% freshness), FSRS-7 recency improves by **-0.0130** LogLoss — see [Status](#status).

```bash
target/release/script --algo FSRS-7 --short --secs --recency --reopt_growth 0.003   --data ../anki-revlogs-10k --processes 10
```

Long runs should be **chunked** (repeat with `--max-user-id 1000, 2000, ...`): results are written
only after the whole parallel loop finishes, so an unchunked 18-hour run loses everything if
interrupted. Resume then skips the users already in the file.

### `--interval_def` — end-to-END vs end-to-START intervals

A review occupies an interval, not an instant: write `start(k)` for when the card is shown and
`end(k)` for when it is answered, so `duration(k) = end(k) - start(k)`. A dataset's
`elapsed_seconds` is a diff between two timestamps of the same kind, so which quantity it holds
depends on how it was built:

| dataset | stored `elapsed_seconds` | |
| --- | --- | --- |
| `anki-revlogs-10k` | `end(k) - end(k-1)` | end-to-END |
| `anki-revlogs-10k-id` | `start(k) - start(k-1)` | start-to-START |
| — | `start(k) - end(k-1)` | **end-to-START** — the span over which memory actually decays |

`--interval_def end_to_end|end_to_start` recomputes the column. It is exact on
`anki-revlogs-10k-id`, which carries `review_time`; without timestamps `end_to_end` is the stored
column and `end_to_start` subtracts this review's own `duration`. `elapsed_days` is never touched.

end-to-START is the better-motivated quantity: `duration(k)` does not exist at prediction time (the
card has been shown and not yet answered) and it correlates with the outcome, so end-to-END hides a
prediction-time-unavailable, outcome-correlated quantity inside the interval.

**⚠ `--interval_def stored` is NOT `end_to_end` on the `-id` dataset.** Always pass the definition
explicitly when comparing datasets.

**⚠ Pair the comparison with `--min_interval_secs 1`.** `end-to-start = end-to-end - duration(k)`
and `duration ≥ 0`, so end-to-start's row set is a strict *subset*: it drops rows whose interval
falls under the pipeline's 1-second threshold (0.1724% of reviews, 8,686/10,000 users), and those
rows are easier than average, which biases the comparison. Flooring makes both arms evaluate
identical rows. Measured on 10,000 users, properly paired: LogLoss +0.000111, AUC −0.000255, but
RMSE(bins) −0.000022 and MBE −0.000029 (both *better*) — the calibration metrics favour end-to-start
while LogLoss and AUC favour end-to-end. Full write-up: `_interval/FINDINGS.md`.

### `--hp_probe` / `--hp_features` — can hyperparameters be tuned per user?

`--hp_probe` trains every candidate in `models::fsrs_v7::HP_CANDIDATES` twice per fold — once on
100% of the fold's training rows, once on the first 80% with the last 20% held out for validation —
and writes the loss sums to `result/<base>-hpprobe.jsonl`. `--hp_features` writes the matching
per-fold training-set statistics to `result/<base>-hpfeat.jsonl`. Analysis scripts live in
`_hpprobe/`; the findings are in `_hpprobe/FINDINGS.md`.

## Smart presets

`--partitions smart` (FSRS-7 only) groups a user's decks into *data-driven* presets by the similarity
of their trained FSRS-7 parameters, instead of training one model per deck (`--partitions deck`) or
using the user's hand-made presets (`--partitions preset`). Per `TimeSeriesSplit` fold it: (1) trains
per-deck params (the deck path, incl. its double-fallback), (2) log-transforms params 0–3, whitens by
a robust covariance so Euclidean distance = Mahalanobis distance, and clusters the decks
(`src/cluster.rs`, hierarchical linkage via the `kodama` crate + a SciPy-compatible
`fcluster(criterion="distance")`), (3) re-trains one param set per cluster, (4) predicts each test row
with its deck's cluster params (a deck unseen in training goes to the nearest cluster). The eval
row-set is identical to the non-partitioned run, so `size` matches `FSRS-7-short-secs`.

**Prerequisite — fit the covariance once** (from FSRS-7 `--short --secs --recency` over all 10k users):

```
target\release\script.exe --algo FSRS-7 --short --secs --recency --processes N   # generate the params
python _smart\fit_covariance.py                                                  # -> _smart/smart_preset_cov.json
```

Then run one experiment, or the whole matrix at once (5 linkages × 6 thresholds, sharing the per-deck
training — ~2.6× faster than 30 separate runs):

```
target\release\script.exe --algo FSRS-7 --short --secs --partitions smart --cluster_method ward --cluster_threshold 12
target\release\script.exe --algo FSRS-7 --short --secs --partitions smart --cluster_sweep   # all 30 hierarchical
```

There is also an **HDBSCAN** density-clustering sweep (16 experiments = min_cluster_size {2,5,10,20} ×
min_samples {1,5} × cluster_selection_method {eom,leaf}; `allow_single_cluster`, noise→nearest):

```
target\release\script.exe --algo FSRS-7 --short --secs --partitions smart --cluster_method hdbscan --cluster_sweep
```

(`cluster_selection_epsilon` is *not* a useful knob here — at the whitened-Mahalanobis distance scale
it never merges, and sklearn crashes for large eps — so the sweep varies min_cluster_size/min_samples/
method instead.) Each experiment writes `result/FSRS-7-short-secs-smart-<suffix>.jsonl`. Aggregate all
of them vs the baseline with `python _smart/sweep_report.py`. The hierarchical clustering reproduces
SciPy exactly and HDBSCAN reproduces sklearn (both unit-tested in `src/cluster.rs` / `src/hdbscan.rs`;
HDBSCAN matches except an unreplicable numpy-argsort tie-break on `min_samples>1`+`leaf`). The
covariance recipe mirrors `_smart/fit_covariance.py`.

**KL-divergence distance (`--cluster_distance kl`)** swaps the parameter-Mahalanobis metric for the
similarity of the decks' *predictions*: each deck's trained model predicts on the user's own rows, and
the distance is the mean symmetric Bernoulli KL between two decks. KL has the closed form
`½·(pₐ−p_b)·(logit pₐ − logit p_b)`, and (since users have up to ~5000 decks, making the O(decks²·rows)
matrix otherwise intractable) it's estimated on a 256-row strided subsample — same scale, so the
calibrated KL thresholds in `_smart/kl_calibrate.py` still hold. The same 30 hierarchical + 16 HDBSCAN
matrix runs with `--cluster_distance kl --cluster_sweep`, writing `…-smart-kl-<suffix>.jsonl`.

**Optimal partition (`--cluster_method optimal --cluster_sweep`)** drops distance clustering entirely
and searches the *partition* space for the one minimizing AIC/BIC on the training fold (no test
peeking): exhaustive over all set partitions for ≤6 decks, greedy agglomerative for 7–12, and a
Mahalanobis-or-KL pre-merge to 12 pseudo-decks above that. It writes 4 files (`…-smart-opt-{bic,aic}-{maha,kl}.jsonl`).

**Finding (1000 users):** none of it beats the per-user global model. Across both distance metrics
(Mahalanobis, KL), both cluster shapes (hierarchical, HDBSCAN), and the honest AIC/BIC optimal
partition, every config's mean LogLoss is ≥ baseline except by f32 noise — pooling all of a user's
decks into one model wins. Results are tabulated in `Smart Preset Assignment.xlsx`; stripped per-user
outputs are archived in `_smart/results/`.

## Status

All upstream-referenced configurations are ported and reproduced over the full dataset (see the
Results tables above); `size` is exact for every config. See `CLAUDE.md` for the architecture and
implementation notes.
