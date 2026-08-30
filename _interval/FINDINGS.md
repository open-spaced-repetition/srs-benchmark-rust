# Interval definition, and what actually differs between the two datasets

A review occupies an interval, not an instant. Write `start(k)` for the moment the card is shown
and `end(k)` for the moment it is answered, so `duration(k) = end(k) - start(k)`, and Anki's
`revlog.id` is `end(k)` (a row is written when the user answers).

| name | formula | what it measures |
| --- | --- | --- |
| end-to-**END** | `end(k) - end(k-1)` | the gap plus this review's own duration |
| start-to-**START** | `start(k) - start(k-1)` | the gap plus `duration(k-1)` |
| end-to-**START** | `start(k) - end(k-1)` | the span over which the memory actually decays |

end-to-START is the better-motivated quantity for two reasons. Decay begins when the user last
finished being shown the answer, and the test happens when the card is next shown; anything after
that is the user thinking. More importantly `duration(k)` **does not exist at prediction time** (the
card has been shown and not yet answered) and it **correlates with the outcome**, because a review
the user struggles with takes longer — so end-to-END hides a prediction-time-unavailable,
outcome-correlated quantity inside the interval that every algorithm consumes.

---

## 1. What each dataset stores (measured, not assumed)

Verified against `-id`'s `review_time`, per card, 100% of rows with a same-card predecessor
(users 1 and 7):

| dataset | stored `elapsed_seconds` | |
| --- | --- | --- |
| `anki-revlogs-10k` | `end(k) - end(k-1)` | end-to-END |
| `anki-revlogs-10k-id` | `start(k) - start(k-1)` | **start-to-START** |

So the `-id` set is **not** "the end-to-start dataset" — it is a third definition. `--interval_def
stored` is therefore NOT `end_to_end` on `-id`; always pass the definition explicitly when comparing
datasets. A PR adding `--elapsed-end-to-start` to the builder is open:
<https://github.com/open-spaced-repetition/anki-revlogs-dataset-builder/pull/2>

## 2. `-id` differs from the base set in THREE ways, not one

Both sets contain **exactly the same reviews** — identical `rating/state/duration` multiset on all
40 sampled users. The builder's `--show-time` changed three things at once:

| difference | extent (40 users, 2,606,463 rows) |
| --- | --- |
| `elapsed_seconds` end-to-END -> start-to-START | all rows |
| **sort order** answer-time -> show-time | 25/40 users, **0.584% of rows** at a different index |
| **`day_offset`** at the rollover (begun before midnight, answered after) | 12/40 users, net 88 review-days |

The sort order is the sneaky one: `review_th` is assigned from file row order, so reordering moves
the `TimeSeriesSplit` boundaries and changes which reviews are train vs test.

**Consequence: a base-vs-`-id` metric difference is not an interval-definition measurement.**
Measured, both on their stored columns: LogLoss 0.317947 -> 0.318137 (+0.000190), AUC -0.000414,
`size` 519,296,315 -> 519,358,720. That number bundles all three changes.

The same effect explains why `--interval_def end_to_end` on `-id` does not *exactly* reproduce the
base run: mean LogLoss differs by only -0.0000034 and `size` by 1,580 rows (0.0003%), which is the
residue of the reordering and the day-rollover shifts, not an error in the recomputation.

## 3. The clean experiment: end-to-END vs end-to-START, same dataset

`FSRS-7 --short --secs --recency` on `anki-revlogs-10k-id`, 10,000 users, `--interval_def` switched.
Same rows, same order, same day assignment; only `elapsed_seconds` moves.

| metric | end-to-END | end-to-START | diff |
| --- | ---: | ---: | ---: |
| **LogLoss** | 0.317944 | 0.318275 | **+0.000331** |
| RMSE | 0.294900 | 0.295040 | +0.000140 |
| RMSE(bins) | 0.063584 | 0.063616 | +0.000031 |
| AUC | 0.752031 | 0.751524 | -0.000506 |
| MBE | 0.001838 | 0.001863 | +0.000026 |

Only 29.6% of users improve. Per-user `dLogLoss`: median +0.000143, p05 -0.001086, p95 +0.002156 —
a small systematic shift, not a few outliers.

**Confound, and its direction.** end-to-START drops **895,435 rows (0.1724%)** across 8,686 users:
removing a whole previous duration pushes sub-second gaps to 0, which then fail the `delta_t > 0`
filter. Those rows are **easier than average** (recall 0.9208 vs 0.8577), so dropping them
mechanically *raises* mean log loss. Estimating their contribution (assuming the model predicts them
at their own base rate) puts it at roughly **+0.00007 of the +0.00033** — about a fifth. The true
penalty is nearer +0.00026, and the confound makes end-to-start look worse than it is.

**Interpretation.** The theoretical case for end-to-start is unchanged and still good. What this
measures is that FSRS-7 was mildly *benefiting* from the leakage: removing a
prediction-time-unavailable, outcome-correlated quantity costs ~0.0003. That is an argument for
end-to-start on correctness grounds, not an argument that it improves accuracy.

The pre-registered prediction (recorded before running: "aggregate LogLoss moves less than the gap
between adjacent rows of the with-same-day table, and no ranking changes") **holds** — the move is
0.00033 against a 0.0016 gap to the next row. A flag and a footnote, not a correction.

## 4. Where the effect lives

It is a **tail, not a median** (40 stride-sampled users of `-id`):

| | same-day (38.1% of rows) | longer interval (61.9%) |
| --- | --- | --- |
| median gap | 473 s | 446,184 s |
| `duration(k)` as % of the gap, median | 1.21% | 0.001% |
| ...p90 | **12.20%** | 0.01% |
| ...p99 | 69.93% | 0.06% |
| rows moving >= 10% | **11.89%** | 0.00% |

On longer intervals it is numerically invisible, so **only the with-same-day tables can move**.

## 5. Rust vs Python predictions (base dataset)

Not an interval question, but recorded here because it constrains what raw predictions can be used
for. 20 users, 655,345 predictions, `FSRS-7 --short --secs --recency`:

| check | result |
| --- | --- |
| label (`y`) mismatches | 0 |
| `size` mismatches | 0 |
| per-user LogLoss diff | mean +0.000180, max 0.002654 |
| \|dp\| > 0.001 | 88,623 (13.52%) |
| \|dp\| > 0.01 | 28,093 (4.29%) |
| max \|dp\| | 0.2375 |

The evaluated rows and labels are identical; the **trained parameters differ** (mean \|dw\| 0.00275,
max 0.151, 3/20 users with some \|dw\| > 0.01). FSRS-7 trains in f32 on a chaotic objective, so Rust's
Adam and torch's Adam reach different, near-equally-good optima. Aggregate LogLoss still agrees to
+0.0001 — but **individual predictions are not interchangeable between the two implementations.**
