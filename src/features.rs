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

    let mut start = 0usize;
    while start < pre.len() {
        let card = pre[start].card_id;
        let mut end = start + 1;
        while end < pre.len() && pre[end].card_id == card {
            end += 1;
        }
        let group = &pre[start..end];
        let card_idx = cards.len() as u32;

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

    rows.sort_by_key(|r| r.review_th);

    if rows.is_empty() {
        return Err("no data after feature engineering".into());
    }
    Ok(Dataset { rows, cards })
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
