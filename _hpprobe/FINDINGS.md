# Research modes: per-user hyperparameters, and retraining frequency

Two studies on FSRS-7, both asking "how much better could this be" rather than reproducing Python.
Config throughout: `--algo FSRS-7 --short --secs --recency`, dataset `anki-revlogs-10k`.
Both keep the evaluated row set identical to a normal run, so `size` is exact and every LogLoss
below is directly comparable to `result/FSRS-7-short-secs-recency.jsonl` (Rust baseline 0.317947).

Analysis scripts: `analyze.py` (selectors), `tree.py` (cost-sensitive tree), `frontier.py`
(cost/gain), `subsets.py` (candidate subsets). Archived data in `results/`.

---

## Study 1 — can lr / betas / n_epoch be chosen per user? (`--hp_probe`)

**Constraint.** Anki's "Optimize" fits ONE model on ONE training set from the default parameters:
no warm start, no folds. So a per-user rule is only deployable if it can be decided from the
training set alone. Target overhead: <2x.

**Probe.** 1000 users, 5 folds, 12 candidates, each trained twice per fold — once on 100% of the
fold's training rows (`test_full`), once on the first 80% with the last 20% held out (`val`, and
`test_val` for that same 80%-trained model). 35x a normal run; 1911 s wall on 10 threads. The
`default` candidate reproduces the benchmark's per-user LogLoss to 5e-7 (the 6-dp rounding).

**Zero-search policies.** Cost = training time relative to the 9-epoch default.

| policy | gain | cost |
| --- | --- | --- |
| always-ep45 | -0.000356 | 5.00x |
| always-hg0.05_ep20 | -0.000322 | 2.22x |
| always-ep20 | -0.000261 | 2.22x |
| always-lr_double | -0.000067 | 1.00x |
| always-hg0.05 (hypergradient) | -0.000038 | 1.00x |
| always-lr_half | +0.000617 | 1.00x |

**An interpretable decision tree does not work.** Cost-sensitive tree over user statistics
(review count, cards, reviews/card, Adam steps, the 4 button shares, `log1p(delta_t)` mean/median/sd,
same-day share, mean pos), cross-validated over USERS:

| tree | CV gain | mean cost |
| --- | --- | --- |
| all 12 candidates, depth 3 | -0.000383 | 3.24x |
| restricted to <=2x candidates | -0.000085 | 1.00x |

-0.000383 against `always-ep45`'s -0.000356 is a 0.00003 improvement over a single global setting —
noise. Depth 6 is WORSE than depth 3 (-0.00016), and in-sample the depth-3 tree claims -0.000661
against a true -0.000383, so 42% of the apparent gain is overfitting. **The per-user optimum is not
a function of these statistics.** A neural net would not fix this; the limit is signal, not model
class. (Also: retention rate is exactly `1 - p_again` — `features.rs` labels `y = 0` iff
`rating == 1` — so the 4 button shares replace it.)

**Hypergradient descent on the lr is real but negligible.** Baydin et al. 2018, adapted:
`dL/da = -grad(w_t) . u_{t-1}`, one dot product over 34 parameters per step, so it is free. Two
deviations were needed to make one `beta` work across users spanning 4 orders of magnitude of review
count: the update is **normalized and multiplicative** (`base_lr *= exp(beta*cos)`, since the raw dot
product scales with the summed loss and the batch size), and **cosine annealing is kept** on top.
Result: **-0.000038**. On a 20-user smoke test it looked like -0.00024; that did not survive. It also
does not stack — `hg0.05_ep20` (-0.000322) is no better than `ep20` (-0.000261).
`TrainConfig::hyper_beta = 0.0` leaves the loop bit-identical (verified: 30 users, LogLoss and `size`
unchanged to all 6 dp).

**Per-user selection DOES work — but it needs a trial run, not a statistic.** Train candidates on
80%, pick by validation loss summed over the user's folds, refit the winner on 100%:

| candidate subset | gain | cost |
| --- | --- | --- |
| default + lr_half + lr_double | **-0.000491** | 3.40x |
| default + ep20 | -0.000452 | 4.39x |
| default + lr_double | -0.000306 | 2.60x |
| default + ep45 | -0.000697 | 8.42x |
| all 12 | -0.000860 | 18.15x |

`always-lr_half` is terrible globally (+0.000617) and `always-lr_double` is nearly nothing
(-0.000067), yet **choosing among the three per user gives -0.000491** — better than `always-ep45`
and cheaper. The lr optimum genuinely varies per user; it is just not predictable from summary
statistics, which is exactly why the tree fails.

**Dead ends worth not repeating.**

* *Select on validation, skip the refit* (`test_val`): **+0.0058**. Losing 20% of the training data
  costs ~10x more than any tuning gain. This was the only multi-candidate scheme that fit under 2x,
  so its death takes the whole "search per user" family out of budget.
* *Per-FOLD selection* (-0.000210) is 4x worse than per-USER selection (-0.000860) — pooling folds
  cuts the selection noise. Validation picks the test-optimal candidate 19.7% of the time
  (chance 8.3%).

**Verdict.** Under a hard <2x budget there is nothing here: about -0.00008, one sixth of the
+-0.0005 gate. The 3.40x row is the best value on the frontier, and since it is three independent
9-epoch runs from default parameters, it parallelizes to ~1.2x WALL time on three cores even though
CPU is 3.4x. That is the only version worth considering.

---

## Study 2 — how much does optimizing more often buy? (`--retrain_growth`)

**Motivation** (Andrew): compare FSRS-7's *best case* against RWKV's *average case*. RWKV does
delta-rule updates internally, so a frequently-optimized FSRS-7 is the fairer comparison — a
remaining gap is then attributable to representational capacity, not optimization staleness.

**Free upper bound first.** `FSRS-6-short-recency-train_equals_test` trains on the test rows
themselves (including their labels), which no honest schedule can match: **-0.014346**. That already
said the gap to RWKV (0.0204) probably would not close.

**Cost of the schedules** (whole 10k FSRS-7 run = 1.8 h CPU baseline). Retraining after literally
every review is `n^2/2` training rows against the 5-fold `2.5n`, and `n` averages 51,930:

| schedule | freshness | x baseline | 10 threads |
| --- | --- | --- | --- |
| every review | 100% | 41,440x | 10 months |
| every 512 reviews | varies | 81x | 14 h |
| growth 100% (doubling) | >=50% | **0.8x** | 8 min |
| growth 25% | >=80% | 2.0x | 24 min |
| growth 0.3% | >=99.7% | 97x | 17.7 h |

Fixed-K is the wrong shape: 512 reviews is a small user's entire history and a rounding error for a
large one. A **geometric** schedule bounds staleness relatively. Note the doubling row: it is
*cheaper* than the current 5-fold split (2n of training work vs 2.5n) and gives ~20 fitting points
per user instead of 5. (Sanity check: `growth 0.5` scored **+0.000242**, i.e. WORSE — the 5-fold
split steps additively by n/6, so it is fresher than a coarse multiplicative schedule near the end.
Beating it everywhere needs eps <= 0.2.)

**Result — 10,000 users, `--retrain_growth 0.003`, 150.4 h CPU (97x), 17.7 h wall on 10 threads:**

| | LogLoss |
| --- | --- |
| FSRS-7 recency, 5-fold | 0.317947 |
| **FSRS-7 recency, continuous** | **0.304891** |
| delta | **-0.013056** |

`size` exact (519,296,315, 0 mismatches). RMSE(bins) -0.009859, AUC +0.027326.
**99.8% of users improved** (9980/10000). The gain is 91% of the cheating `train_equals_test` bound.

**Against the leaderboard** (`--short --secs`, 10,000 collections):

| model | LogLoss |
| --- | --- |
| FSRS-7 | 0.320564 |
| FSRS-7 recency (5-fold) | 0.317947 |
| GRU | 0.314574 |
| LSTM | 0.313654 |
| **FSRS-7 recency (continuous)** | **0.304891** |
| RWKV | 0.297427 |
| RWKV-P | 0.265997 |

**Verdict.** FSRS-7's best case clears GRU and LSTM comfortably but **does not reach RWKV's average
case — it is 0.00746 short**, and 0.0389 short of RWKV-P.

**Comparability caveat.** These numbers are comparable to the table above only in the mechanical
sense that `size` and the scored rows are identical. A fresher retraining schedule improves ANY
trainable algorithm, so a continuous-retraining number cannot sit in the leaderboard next to 5-fold
entries — it measures the protocol, not the model. Publishing it needs a separate column with every
algorithm re-run under the same schedule, and the untrainable baselines (AVG, SM2, Ebisu) do not
move at all, so their apparent rank would drop for a reason unrelated to them.
