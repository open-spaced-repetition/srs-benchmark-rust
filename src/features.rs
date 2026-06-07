//! Feature engineering — a faithful port of `features/base.py` (the common pipeline shared
//! by every model). Correctness of the surviving row set here is what makes `size` exact
//! (rule #6), so this mirrors the Python operation order precisely.
//!
//! Implements the `--secs` path (and the non-secs path WITHOUT the outlier /
//! non-continuous-row removal, added later). `--short` and `--two_buttons` are handled.
//!
//! Histories (the prior `(Δt, rating)` sequence each model needs) are NOT materialized as
//! per-row strings — that was O(Σ prefix length) with heavy allocation/float-formatting and
//! dominated runtime on large users. Instead each card keeps compact numeric arrays
//! ([`Card`]) and each [`Row`] references them via `(card_idx, pos)`; models slice
//! `cards[card_idx].*[0..pos]` for the prior sequence.

use std::collections::{HashMap, HashSet};

use crate::config::Config;

/// Per-card review sequence (post-preprocessing: after rating filter + `i>128` drop, before
/// the postprocessing short/`delta_t>0` filters). Indexed by `Row::pos`.
#[derive(Debug, Clone)]
pub struct Card {
    /// Rating per review (after `--two_buttons` remap).
    pub ratings: Vec<i64>,
    /// Active interval per review (days): secs→`elapsed_seconds/86400` or days→`elapsed_days`,
    /// clamped ≥ 0. This is the value used in the FSRS input tensor / t_history.
    pub dt_active: Vec<f64>,
    /// `max(0, elapsed_days)` per review (used for last_rating and non-secs intervals).
    pub dt_int: Vec<i64>,
}

/// One review row after feature engineering, in final `review_th` order.
#[derive(Debug, Clone)]
pub struct Row {
    pub card_idx: u32,
    /// Index of this review within its card's [`Card`] arrays. The prior sequence is `0..pos`.
    pub pos: u32,
    pub card_id: i64,
    pub review_th: i64,
    pub rating: i64,
    pub y: i64,
    /// Active interval in days, > 0 in returned rows (= `cards[card_idx].dt_active[pos]`).
    pub delta_t: f64,
    pub elapsed_days: i64,
    pub elapsed_seconds: i64,
    pub duration: i64,
    /// Review-count feature: per-card running count of `elapsed_days>0` (incl. current), +1.
    pub i: i64,
    pub rmse_bins_lapse: i64,
    pub last_rating: i64,
    pub first_rating: i64,
    pub partition: i64,
}

/// Feature-engineered dataset for one user.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub rows: Vec<Row>,
    pub cards: Vec<Card>,
}

impl Dataset {
    /// Prior ratings of `row`'s card (the reviews before it), in order.
    pub fn prior_ratings(&self, row: &Row) -> &[i64] {
        &self.cards[row.card_idx as usize].ratings[..row.pos as usize]
    }
    /// Prior active intervals of `row`'s card, in order.
    pub fn prior_dt_active(&self, row: &Row) -> &[f64] {
        &self.cards[row.card_idx as usize].dt_active[..row.pos as usize]
    }
    /// Intervals `dt_active[1..=pos]` (length `pos`) — DASH's `t_history` is `t_item[1:]`.
    pub fn intervals_from_second(&self, row: &Row) -> &[f64] {
        &self.cards[row.card_idx as usize].dt_active[1..=row.pos as usize]
    }
    /// Intervals `dt_active[0..=pos]` (length `pos+1`, incl. current) — ACT-R's cumulative
    /// time tensor is `cumsum` of this.
    pub fn dt_active_incl(&self, row: &Row) -> &[f64] {
        &self.cards[row.card_idx as usize].dt_active[..=row.pos as usize]
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Intermediate per-row state after `_common_preprocessing`.
struct PreRow {
    src: usize, // index into raw, = file order; review_th = src + 1
    card_id: i64,
    rating: i64,
    delta_t: f64,     // active interval (secs->days if --secs, else days), clamped >=0
    delta_t_int: i64, // max(0, elapsed_days)
    elapsed_days: i64,
    elapsed_seconds: i64,
    duration: i64,
}

#[inline]
fn label(rating: i64) -> i64 {
    if rating == 1 {
        0
    } else {
        1
    }
}

/// Port of `BaseFeatureEngineer.create_features` (common path).
pub fn create_features(raw: &crate::data::RawRevlogs, cfg: &Config) -> Result<Dataset, String> {
    let n = raw.len();
    let max_i = (cfg.max_seq_len as i64) * 2; // i > max_seq_len*2 dropped

    // --- _common_preprocessing ---
    // review_th = 1..n in file order; sort by (card_id, review_th). review_th is the file
    // index, so sorting by (card_id, src) reproduces sort_values(["card_id","review_th"]).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| raw.card_id[a].cmp(&raw.card_id[b]).then(a.cmp(&b)));

    let mut pre: Vec<PreRow> = Vec::with_capacity(n);
    let mut cur_card = i64::MIN;
    let mut pos = 0i64; // within-card cumcount over rating-valid rows
    for &idx in &order {
        let mut rating = raw.rating[idx];
        if !(1..=4).contains(&rating) {
            continue; // drop invalid ratings
        }
        if raw.card_id[idx] != cur_card {
            cur_card = raw.card_id[idx];
            pos = 0;
        }
        if cfg.two_buttons && (rating == 2 || rating == 4) {
            rating = 3;
        }
        pos += 1; // i = cumcount + 1
        if pos > max_i {
            continue; // drop i > max_seq_len*2
        }

        let elapsed_days = raw.elapsed_days[idx];
        let elapsed_seconds = raw.elapsed_seconds[idx];
        let delta_t_int = elapsed_days.max(0);
        let delta_t = if cfg.use_secs_intervals {
            (elapsed_seconds as f64 / 86400.0).max(0.0)
        } else {
            delta_t_int as f64
        };

        pre.push(PreRow {
            src: idx,
            card_id: raw.card_id[idx],
            rating,
            delta_t,
            delta_t_int,
            elapsed_days,
            elapsed_seconds,
            duration: raw.duration[idx],
        });
    }

    // Non-short: drop same-day reviews (elapsed_days==0) before histories (base.py).
    if !cfg.include_short_term {
        pre.retain(|r| r.elapsed_days != 0);
    }

    // --- _compute_histories + _common_postprocessing, per card group ---
    let mut cards: Vec<Card> = Vec::new();
    let mut rows: Vec<Row> = Vec::with_capacity(pre.len());
    // Per card: whether it has an `i==1` row, i.e. its first review is NOT a positive-interval
    // review (`elapsed_days <= 0` → `i = cumsum(elapsed_days>0)+1 == 1`; new cards log -1).
    // Needed by the non-secs outlier/continuity removal to decide whole-card vs i==2-only drops.
    let mut card_has_i1: Vec<bool> = Vec::new();

    let mut start = 0usize;
    while start < pre.len() {
        let card = pre[start].card_id;
        let mut end = start + 1;
        while end < pre.len() && pre[end].card_id == card {
            end += 1;
        }
        let group = &pre[start..end];
        let card_idx = cards.len() as u32;
        card_has_i1.push(group[0].elapsed_days <= 0);

        let first_rating = group[0].rating;
        let ratings: Vec<i64> = group.iter().map(|r| r.rating).collect();
        let dt_active: Vec<f64> = group.iter().map(|r| r.delta_t).collect();
        let dt_int: Vec<i64> = group.iter().map(|r| r.delta_t_int).collect();

        let mut lapse_prefix = 0i64; // exclusive prefix sum of is_lapse
        let mut i_run = 0i64; // running count of elapsed_days>0
        let mut last_rating_with_dt = first_rating; // most recent prior review with dt_int>0

        for (j, r) in group.iter().enumerate() {
            // last_rating: most recent prior review with elapsed_days>0, else first rating.
            let last_rating = if j == 0 { first_rating } else { last_rating_with_dt };

            let is_lapse = (r.rating == 1 && interval_str_nonzero(r.delta_t, cfg)) as i64;
            let rmse_bins_lapse = lapse_prefix;
            lapse_prefix += is_lapse;

            if r.elapsed_days > 0 {
                i_run += 1;
            }
            let i_val = i_run + 1;

            // --short keep filter is `(delta_t != 0) | (i_pre == 1)`, but the final filter
            // `delta_t > 0` then drops the delta_t==0 rows anyway; any elapsed_days>0 row has
            // delta_t>0, so the net surviving set is exactly delta_t > 0.
            if r.delta_t > 0.0 {
                rows.push(Row {
                    card_idx,
                    pos: j as u32,
                    card_id: r.card_id,
                    review_th: (r.src as i64) + 1,
                    rating: r.rating,
                    y: label(r.rating),
                    delta_t: r.delta_t,
                    elapsed_days: r.elapsed_days,
                    elapsed_seconds: r.elapsed_seconds,
                    duration: r.duration,
                    i: i_val,
                    rmse_bins_lapse,
                    last_rating,
                    first_rating,
                    partition: 0,
                });
            }

            // Update "most recent prior review with dt_int>0" for the NEXT review.
            if r.delta_t_int > 0 {
                last_rating_with_dt = r.rating;
            }
        }

        cards.push(Card {
            ratings,
            dt_active,
            dt_int,
        });
        start = end;
    }

    // Non-secs only: outlier + non-continuous-row removal (`remove_outliers` /
    // `remove_non_continuous_rows`, run in `_common_postprocessing` only when NOT --secs).
    if !cfg.use_secs_intervals {
        apply_outlier_continuity_filter(&mut rows, &card_has_i1);
    }

    rows.sort_by_key(|r| r.review_th);

    if rows.is_empty() {
        return Err("no data after feature engineering".into());
    }
    Ok(Dataset { rows, cards })
}

/// `remove_outliers` (per `first_rating`): return the set of `delta_t` (=elapsed_days) cells
/// removed from the `i==2` rows. `dtmap` maps each delta_t to its row count for this rating.
fn outlier_removed_deltas(dtmap: &HashMap<i64, i64>, first_rating: i64) -> HashSet<i64> {
    let total: i64 = dtmap.values().sum();
    // Iterate cells smallest-count-first, ties broken by larger delta_t first (matches
    // pandas `sort_values(by=[count, delta_t], ascending=[True, False])`; delta_t is unique
    // per cell so the order is fully determined).
    let mut cells: Vec<(i64, i64)> = dtmap.iter().map(|(&dt, &c)| (dt, c)).collect();
    cells.sort_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
    // Threshold = max(total*0.05, 20). Compute total*0.05 in f64 exactly as Python does.
    let threshold = (total as f64 * 0.05).max(20.0);
    let limit = if first_rating != 4 { 100 } else { 365 };
    let mut removed = HashSet::new();
    let mut has_been_removed: i64 = 0;
    for (dt, count) in cells {
        if (has_been_removed + count) as f64 >= threshold {
            // Budget reached: only drop "real" outlier cells (rare or extreme interval).
            if count < 6 || dt > limit {
                removed.insert(dt);
                has_been_removed += count;
            }
        } else {
            removed.insert(dt);
            has_been_removed += count;
        }
    }
    removed
}

/// Apply the non-secs outlier + continuity removal to `rows` in place.
///
/// Only `i==2` rows (each card's first positive-interval review) are eligible for outlier
/// removal, and continuity truncation can therefore only act at that one position. So a card
/// whose first-positive `delta_t` is an outlier loses either the whole card (if it has an
/// `i==1` first review (`elapsed_days <= 0`)) or only the `i==2` row (if its first review
/// already had a positive interval — then `i==3,4,…` stay continuous and survive).
fn apply_outlier_continuity_filter(rows: &mut Vec<Row>, card_has_i1: &[bool]) {
    // Count i==2 rows per (first_rating, delta_t).
    let mut cells: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
    for r in rows.iter() {
        if r.i == 2 {
            *cells
                .entry(r.first_rating)
                .or_default()
                .entry(r.elapsed_days)
                .or_default() += 1;
        }
    }
    let removed: HashMap<i64, HashSet<i64>> = cells
        .iter()
        .map(|(&fr, dtmap)| (fr, outlier_removed_deltas(dtmap, fr)))
        .collect();

    // Decide, per card, whether to drop the whole card or just its i==2 row.
    let mut drop_card: HashSet<u32> = HashSet::new();
    let mut drop_i2_card: HashSet<u32> = HashSet::new();
    for r in rows.iter() {
        if r.i == 2
            && removed
                .get(&r.first_rating)
                .is_some_and(|s| s.contains(&r.elapsed_days))
        {
            if card_has_i1[r.card_idx as usize] {
                drop_card.insert(r.card_idx);
            } else {
                drop_i2_card.insert(r.card_idx);
            }
        }
    }

    rows.retain(|r| {
        !drop_card.contains(&r.card_idx) && !(r.i == 2 && drop_i2_card.contains(&r.card_idx))
    });
}

/// Whether `str(delta_t) != "0"`, matching Python. For --secs, delta_t is a float whose
/// str() is never exactly "0"; for days it's an int, str(0)=="0".
#[inline]
fn interval_str_nonzero(delta_t: f64, cfg: &Config) -> bool {
    if cfg.use_secs_intervals {
        true
    } else {
        delta_t != 0.0
    }
}
