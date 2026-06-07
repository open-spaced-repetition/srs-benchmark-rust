//! Feature engineering — a faithful port of `features/base.py` (the common pipeline shared
//! by every model). Correctness of the surviving row set here is what makes `size` exact
//! (rule #6), so this mirrors the Python operation order precisely.
//!
//! Currently implements the path for `--secs` configs (and the non-secs path WITHOUT the
//! outlier / non-continuous-row removal, which is added in a later phase). The `--short`
//! flag and `--two_buttons` are handled.

use crate::config::Config;

/// One review row after feature engineering, in final `review_th` order.
#[derive(Debug, Clone)]
pub struct Row {
    pub card_id: i64,
    pub review_th: i64,
    pub rating: i64,
    pub y: i64,
    /// Active interval in days: `elapsed_seconds/86400` (secs) or `elapsed_days` (days),
    /// clamped to ≥ 0. Always > 0 in the returned rows.
    pub delta_t: f64,
    pub elapsed_days: i64,
    pub elapsed_seconds: i64,
    pub duration: i64,
    /// Review count feature: per-card running count of `elapsed_days > 0` (incl. current), +1.
    pub i: i64,
    pub rmse_bins_lapse: i64,
    pub last_rating: i64,
    pub first_rating: i64,
    /// Comma-joined ratings of prior reviews of this card (e.g. "3,1,3").
    pub r_history: String,
    /// Comma-joined active-interval values of prior reviews (string form, model-dependent).
    pub t_history: String,
    pub partition: i64,
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

/// Map a rating to a binary recall label: {1→0, 2,3,4→1}.
#[inline]
fn label(rating: i64) -> i64 {
    if rating == 1 {
        0
    } else {
        1
    }
}

/// Port of `BaseFeatureEngineer.create_features` (common path).
pub fn create_features(raw: &crate::data::RawRevlogs, cfg: &Config) -> Result<Vec<Row>, String> {
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

    // For the non-secs, non-short path the upstream also drops elapsed_days==0 here and
    // recomputes i; for --short it keeps them. (Non-short non-secs handled later.)
    if !cfg.include_short_term {
        // Drop same-day reviews (elapsed_days == 0) and recompute within-card position.
        let mut kept: Vec<PreRow> = Vec::with_capacity(pre.len());
        for r in pre.into_iter() {
            if r.elapsed_days == 0 {
                continue;
            }
            kept.push(r);
        }
        pre = kept;
    }

    // --- _compute_histories + _common_postprocessing, done per card group ---
    // `pre` is in (card_id, review_th) order: cards are contiguous. Build histories then
    // apply postprocessing filters; collect surviving rows, finally sort by review_th.
    let mut out: Vec<Row> = Vec::with_capacity(pre.len());

    let mut start = 0usize;
    while start < pre.len() {
        let card = pre[start].card_id;
        let mut end = start + 1;
        while end < pre.len() && pre[end].card_id == card {
            end += 1;
        }
        let group = &pre[start..end];

        // first_rating = first review's rating in this card.
        let first_rating = group[0].rating;

        // Exclusive prefix sum of is_lapse for rmse_bins_lapse.
        // is_lapse = (rating==1) & (str(delta_t) != "0"). For --secs delta_t is a float so
        // str(delta_t) is never "0" (0.0 -> "0.0"); for days, str(0)=="0".
        let mut lapse_prefix = 0i64;
        // Running review count i (elapsed_days>0 cumsum) over the post-short-filter rows.
        let mut i_run = 0i64;

        for (j, r) in group.iter().enumerate() {
            // histories: prior reviews [0..j]
            let r_history = join_ratings(&group[..j]);
            let t_history = join_intervals(&group[..j], cfg.use_secs_intervals);
            let last_rating = compute_last_rating(&group[..=j]);

            // rmse_bins_lapse (exclusive prefix of is_lapse)
            let is_lapse = if r.rating == 1 && interval_str_nonzero(r.delta_t, cfg) {
                1
            } else {
                0
            };
            let rmse_bins_lapse = lapse_prefix;
            lapse_prefix += is_lapse;

            // i recompute (cumsum of elapsed_days>0, +1)
            if r.elapsed_days > 0 {
                i_run += 1;
            }
            let i_val = i_run + 1;

            let _ = j;

            // --short keep filter: (delta_t != 0) | (i_pre == 1). We don't keep delta_t==0
            // rows anyway because the final filter is delta_t > 0; and any elapsed_days>0
            // row has delta_t>0. So the net surviving set is exactly delta_t > 0.
            if r.delta_t > 0.0 {
                out.push(Row {
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
                    r_history,
                    t_history,
                    partition: 0,
                });
            }
        }

        start = end;
    }

    // Final: sort by review_th (restores global review order).
    out.sort_by_key(|r| r.review_th);

    if out.is_empty() {
        return Err("no data after feature engineering".into());
    }
    Ok(out)
}

/// last_rating = rating of the most recent prior review with elapsed_days>0 (i.e. the
/// non-secs interval > 0), else the first review's rating. `slice` is group[..=j].
fn compute_last_rating(slice: &[PreRow]) -> i64 {
    let cur = slice.len() - 1;
    // iterate reversed over prior reviews [0..cur)
    for r in slice[..cur].iter().rev() {
        if r.delta_t_int > 0 {
            return r.rating;
        }
    }
    slice[0].rating
}

fn join_ratings(prior: &[PreRow]) -> String {
    let mut s = String::new();
    for (k, r) in prior.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        s.push_str(itoa(r.rating).as_str());
    }
    s
}

fn join_intervals(prior: &[PreRow], secs: bool) -> String {
    let mut s = String::new();
    for (k, r) in prior.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        if secs {
            // NOTE: Python uses str(np.float64(elapsed_seconds/86400)). Exact float
            // formatting is only needed by RMSE-BINS-EXPLOIT's count_lapse; refined later.
            s.push_str(&format!("{}", r.delta_t));
        } else {
            s.push_str(itoa(r.delta_t_int).as_str());
        }
    }
    s
}

#[inline]
fn itoa(v: i64) -> String {
    v.to_string()
}

/// Whether `str(delta_t) != "0"`, matching Python. For --secs, delta_t is a float whose
/// str() is never exactly "0"; for days it's an int, str(0)=="0".
#[inline]
fn interval_str_nonzero(delta_t: f64, cfg: &Config) -> bool {
    if cfg.use_secs_intervals {
        true // float repr never equals "0"
    } else {
        delta_t != 0.0
    }
}
