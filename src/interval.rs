//! `--interval_def`: how the `--secs` interval between two reviews of a card is measured.
//!
//! A review occupies an interval, not an instant. Write `start(k)` for the moment the card is
//! shown and `end(k)` for the moment it is answered, so `duration(k) = end(k) - start(k)` and the
//! revlog row's timestamp (`duration` = `taken_millis`) is `end(k)`.
//!
//! | name | formula | what it measures |
//! |---|---|---|
//! | end-to-**END** | `end(k) - end(k-1)` | the gap plus the current review's own duration |
//! | start-to-**START** | `start(k) - start(k-1)` | the gap plus `duration(k-1)` |
//! | end-to-**START** | `start(k) - end(k-1)` | the gap over which the memory actually decays |
//!
//! End-to-start is the better-motivated quantity for two reasons. Decay begins when the user last
//! finished being shown the answer and the test happens when the card is next shown; anything after
//! that is the user thinking. More importantly, `duration(k)` **does not exist at prediction time**
//! (the card has been shown and not yet answered) and it **correlates with the outcome**, because a
//! review the user struggles with takes longer — so end-to-end feeds every algorithm an
//! outcome-correlated quantity hidden inside the interval.
//!
//! # The stored column means different things in different datasets
//!
//! Verified empirically against `review_time` (100% of rows with a previous review, users 1 and 7):
//!
//! | dataset | stored `elapsed_seconds` |
//! |---|---|
//! | `anki-revlogs-10k` | `end(k) - end(k-1)` — end-to-END |
//! | `anki-revlogs-10k-id` | `start(k) - start(k-1)` — start-to-START, a THIRD definition |
//!
//! So `--interval_def stored` is NOT `end_to_end` on the `-id` dataset. Always pass the definition
//! explicitly when comparing the two datasets, or the comparison silently mixes in this difference.
//!
//! # Flooring the interval (`--min_interval_secs`)
//!
//! `end-to-start = end-to-end - duration(k)` and `duration >= 0`, so the end-to-start interval is
//! always <= the end-to-end one. A row is evaluated only if its interval is >= 1 s (the pipeline's
//! `delta_t > 0`, and `elapsed_seconds` is whole seconds), so switching to end-to-start silently
//! DROPS rows whose own duration was nearly the whole gap — 0.1724% of reviews, and they are easier
//! than average (recall 0.9208 vs 0.8577), which biases any comparison against end-to-start.
//!
//! `--min_interval_secs 1` floors every non-sentinel interval, so both definitions evaluate exactly
//! the same rows (every row with a predecessor survives) and the comparison is properly paired. It
//! applies to `stored` too, so it is a plain, predictable knob rather than a per-definition special
//! case. `0` (the default) leaves everything unchanged.
//!
//! `elapsed_days` is deliberately untouched: it is a calendar-day index difference matching Anki's
//! scheduling semantics, "subtract a duration" is not well defined on a day index, and the effect at
//! day resolution is ~0.001% anyway.

use std::collections::HashMap;

use crate::data::RawRevlogs;

/// Rewrite `elapsed_seconds` in place to the requested definition. A no-op for `stored`.
///
/// Rows keep the `-1` sentinel when there is **no previous review of that card** in the frame, or
/// when the stored value was already `-1` (`state == 0`). Note those are not the same condition: a
/// card whose first row in the frame is not a state-0 row has no previous review yet is not `-1`.
///
/// The clamp to zero happens BEFORE the conversion to whole seconds. Truncating -0.4 s would give
/// -1, silently minting a fake "first review" — the sentinel value.
pub fn apply_interval_def(raw: &mut RawRevlogs, interval_def: &str, min_interval_secs: i64) {
    let floor_to = min_interval_secs.max(0);
    let end_to_start = match interval_def {
        "stored" => {
            apply_floor(raw, floor_to);
            return;
        }
        "end_to_start" => true,
        "end_to_end" => false,
        other => {
            eprintln!("warning: unknown --interval_def {other}, using the stored column");
            apply_floor(raw, floor_to);
            return;
        }
    };

    if raw.review_time.len() == raw.len() {
        // Exact path (`-id`): recompute from timestamps. `review_time` is the SHOW time.
        let mut prev_answer: HashMap<i64, i64> = HashMap::new();
        for i in 0..raw.len() {
            let card = raw.card_id[i];
            let show = raw.review_time[i];
            let answer = show + raw.duration[i];
            let out = match prev_answer.get(&card) {
                Some(&pa) if raw.elapsed_seconds[i] >= 0 => {
                    let ms = if end_to_start { show - pa } else { answer - pa };
                    ms.max(0) / 1000
                }
                _ => -1,
            };
            raw.elapsed_seconds[i] = out;
            prev_answer.insert(card, answer);
        }
    } else if end_to_start {
        // No timestamps (`anki-revlogs-10k`): the stored column is already end-to-end, so
        // end-to-start is one subtraction of THIS review's own duration. `end_to_end` is a no-op.
        for i in 0..raw.len() {
            if raw.elapsed_seconds[i] >= 0 {
                let ms = raw.elapsed_seconds[i] * 1000 - raw.duration[i];
                raw.elapsed_seconds[i] = ms.max(0) / 1000;
            }
        }
    }
    apply_floor(raw, floor_to);
}

/// Raise every non-sentinel interval to at least `floor_to` seconds. The `-1` sentinel ("no known
/// previous review") is left alone — raising it would mint a fake interval for a card's first row.
fn apply_floor(raw: &mut RawRevlogs, floor_to: i64) {
    if floor_to <= 0 {
        return;
    }
    for v in &mut raw.elapsed_seconds {
        if *v >= 0 && *v < floor_to {
            *v = floor_to;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two reviews of one card: show at t=1000 ms taking 500 ms (answer 1500), then show at
    /// t=101_500 taking 300. end-to-end = 101_800-1500 = 100_300 ms = 100 s;
    /// end-to-start = 101_500-1500 = 100_000 ms = 100 s. Second card row has no predecessor.
    fn raw2() -> RawRevlogs {
        RawRevlogs {
            card_id: vec![7, 7],
            day_offset: vec![0, 1],
            rating: vec![3, 3],
            state: vec![0, 1],
            duration: vec![500, 300],
            elapsed_days: vec![-1, 1],
            elapsed_seconds: vec![-1, 100],
            review_time: vec![1000, 101_500],
        }
    }

    #[test]
    fn end_to_end_and_start_from_timestamps() {
        let mut r = raw2();
        apply_interval_def(&mut r, "end_to_end", 0);
        assert_eq!(r.elapsed_seconds, vec![-1, 100]); // 100_300 ms -> 100 s

        let mut r = raw2();
        apply_interval_def(&mut r, "end_to_start", 0);
        assert_eq!(r.elapsed_seconds, vec![-1, 100]); // 100_000 ms -> 100 s
    }

    #[test]
    fn end_to_start_clamps_before_truncating() {
        // A same-day review whose duration exceeds the whole recorded gap must land on 0, not -1:
        // -1 is the "no previous review" sentinel and would mint a fake first review.
        let mut r = raw2();
        r.review_time = vec![1000, 1600]; // gap to prev answer = 100 ms, duration 300 ms
        r.elapsed_seconds = vec![-1, 0];
        apply_interval_def(&mut r, "end_to_start", 0);
        assert_eq!(r.elapsed_seconds[1], 0);
    }

    #[test]
    fn without_timestamps_end_to_start_subtracts_this_duration() {
        let mut r = raw2();
        r.review_time = Vec::new(); // the published dataset has no timestamps
        r.elapsed_seconds = vec![-1, 100]; // stored = end-to-end
        apply_interval_def(&mut r, "end_to_start", 0);
        assert_eq!(r.elapsed_seconds, vec![-1, 99]); // 100_000 - 300 ms -> 99 s

        let mut r = raw2();
        r.review_time = Vec::new();
        apply_interval_def(&mut r, "end_to_end", 0);
        assert_eq!(r.elapsed_seconds, vec![-1, 100]); // no-op
    }

    #[test]
    fn floor_raises_short_gaps_but_never_the_sentinel() {
        // The whole point: with a floor both definitions keep every non-sentinel row, so the two
        // arms evaluate identical row sets and the comparison is properly paired.
        let mut r = raw2();
        r.review_time = vec![1000, 1600]; // end-to-start gap = -200 ms -> clamps to 0 s
        r.elapsed_seconds = vec![-1, 0];
        apply_interval_def(&mut r, "end_to_start", 1);
        assert_eq!(r.elapsed_seconds, vec![-1, 1]); // sentinel untouched, 0 raised to 1

        // ...and a genuinely long gap is not disturbed.
        let mut r = raw2();
        apply_interval_def(&mut r, "end_to_start", 1);
        assert_eq!(r.elapsed_seconds, vec![-1, 100]);
    }

    #[test]
    fn floor_applies_to_the_stored_column_too() {
        let mut r = raw2();
        r.elapsed_seconds = vec![-1, 0];
        apply_interval_def(&mut r, "stored", 1);
        assert_eq!(r.elapsed_seconds, vec![-1, 1]);
    }

    #[test]
    fn sentinel_is_kept_for_the_first_review_of_a_card() {
        let mut r = raw2();
        r.card_id = vec![7, 8]; // two different cards -> neither has a predecessor
        apply_interval_def(&mut r, "end_to_start", 0);
        assert_eq!(r.elapsed_seconds, vec![-1, -1]);
    }
}
