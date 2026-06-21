//! Orchestration: enumerate users, process them in parallel (rayon, threads instead of the
//! Python process pool), time each user (rule #3), and write `result/<name>.jsonl` sorted
//! by user (mirrors `script.py` main + `utils.sort_jsonl`), with resume support.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;
use serde_json::Value;

use crate::config::Config;
use crate::data::{read_user_partition_map, read_user_revlogs};
use crate::eval::{evaluate, Params};
use crate::features::create_features;
use crate::models;

/// Process one user end-to-end (timed). Returns the result JSON object, or an error string.
fn process_user(cfg: &Config, user_id: i64) -> Result<Value, String> {
    let t0 = Instant::now();
    let profile = std::env::var_os("FSRS_PROFILE").is_some();

    let raw = read_user_revlogs(&cfg.data_path, user_id)?;
    let t_read = t0.elapsed();
    let mut ds = create_features(&raw, cfg)?;
    let t_feat = t0.elapsed();
    if ds.len() < 6 {
        return Err(format!("{user_id} does not have enough data."));
    }
    // `--secs --equalize_test_with_non_secs`: the train/test split is defined by the non-secs
    // pipeline (test only on reviews a non-`--secs` run would test). Features stay the secs ones.
    if cfg.use_secs_intervals && cfg.equalize_test_with_non_secs {
        ds.equalize_splits = Some(crate::features::build_equalize_splits(&raw, cfg, &ds)?);
    }
    // `--partitions deck|preset|smart`: tag each row with its card's deck/preset partition
    // (smart clusters decks, so it uses the deck map).
    if cfg.partitions != "none" {
        let kind = if cfg.partitions == "smart" { "deck" } else { cfg.partitions.as_str() };
        let map = read_user_partition_map(&cfg.data_path, user_id, kind)?;
        for r in &mut ds.rows {
            // Python left-merges cards/decks then `fillna(-1)`, so a revlog card with no `cards`
            // row gets partition -1 (NOT 0, which is a real deck id). Match that.
            r.partition = *map.get(&r.card_id).unwrap_or(&-1);
        }
    }

    let out = match cfg.model_name.as_str() {
        "AVG" => models::avg::process(&ds, cfg),
        "SM2" => models::sm2::process(&ds, cfg),
        "SM2-trainable" => models::sm2_trainable::process(&ds, cfg),
        "MOVING-AVG" => models::moving_avg::process(&ds, cfg),
        "HLR" => models::hlr::process(&ds, cfg),
        "LogisticRegression" => models::logistic_regression::process(&ds, cfg),
        "DASH" | "DASH[MCM]" => models::dash::process(&ds, cfg),
        "DASH[ACT-R]" => models::dash_act_r::process(&ds, cfg),
        "ACT-R" => models::act_r::process(&ds, cfg),
        "Anki" => models::anki::process(&ds, cfg),
        "Ebisu-v2" => models::ebisu::process(&ds, cfg),
        "RMSE-BINS-EXPLOIT" => models::rmse_bins_exploit::process(&ds, cfg),
        "FSRSv1" => models::fsrs_v1::process(&ds, cfg),
        "FSRSv2" => models::fsrs_v2::process(&ds, cfg),
        "FSRSv3" => models::fsrs_v3::process(&ds, cfg),
        "FSRSv4" => models::fsrs_v4::process(&ds, cfg),
        "FSRS-4.5" => models::fsrs_v4dot5::process(&ds, cfg),
        "FSRS-5" => models::fsrs_v5::process(&ds, cfg),
        "FSRS-6" => models::fsrs_v6::process(&ds, cfg),
        "FSRS-6-one-step" => models::fsrs_v6_one_step::process(&ds, cfg),
        "FSRS-7" => models::fsrs_v7::process(&ds, cfg),
        #[cfg(feature = "fsrs-rs")]
        "FSRS-rs" => models::fsrs_rs::process(&ds, cfg),
        #[cfg(not(feature = "fsrs-rs"))]
        "FSRS-rs" => {
            return Err("FSRS-rs requires building with `--features fsrs-rs`".into())
        }
        #[cfg(feature = "neural")]
        "GRU" => models::gru::process(&ds, cfg),
        #[cfg(feature = "neural")]
        "LSTM" => models::lstm::process(&ds, cfg),
        #[cfg(not(feature = "neural"))]
        "GRU" => return Err("GRU requires building with `--features neural`".into()),
        #[cfg(not(feature = "neural"))]
        "LSTM" => return Err("LSTM requires building with `--features neural`".into()),
        other => return Err(format!("model '{other}' not yet ported")),
    };

    let time_s = t0.elapsed().as_secs_f64();
    if profile {
        let t_model = t0.elapsed();
        let read_ms = t_read.as_secs_f64() * 1e3;
        let feat_ms = (t_feat - t_read).as_secs_f64() * 1e3;
        let model_ms = (t_model - t_feat).as_secs_f64() * 1e3;
        eprintln!(
            "PROFILE user={user_id} rows={} read_ms={read_ms:.1} feat_ms={feat_ms:.1} model_ms={model_ms:.1}",
            ds.len()
        );
    }
    let _ = Params::None;
    Ok(evaluate(&out.eval_rows, &out.p, cfg, user_id, out.params, time_s))
}

/// Enumerate user ids from `<data>/revlogs/user_id=*` directories.
fn enumerate_users(data_path: &Path, max_user_id: Option<i64>) -> Result<Vec<i64>, String> {
    let dir = data_path.join("revlogs");
    let mut users = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("user_id=") {
            if let Ok(id) = rest.parse::<i64>() {
                if max_user_id.map(|m| id <= m).unwrap_or(true) {
                    users.push(id);
                }
            }
        }
    }
    users.sort_unstable();
    Ok(users)
}

/// Python `json.dumps` default separators are `(", ", ": ")`; serde's compact output omits
/// the spaces. This formatter reproduces the spacing so output is byte-compatible.
struct PyFormatter;
impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

fn to_py_json(value: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFormatter);
    value.serialize(&mut ser).expect("serialize");
    String::from_utf8(buf).expect("utf8")
}

/// Read an existing jsonl into (parsed values, set of user ids) for resume.
fn read_existing(path: &Path) -> (Vec<Value>, std::collections::HashSet<i64>) {
    let mut vals = Vec::new();
    let mut set = std::collections::HashSet::new();
    if let Ok(content) = fs::read_to_string(path) {
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if let Some(u) = v.get("user").and_then(|x| x.as_i64()) {
                    set.insert(u);
                }
                vals.push(v);
            }
        }
    }
    (vals, set)
}

fn write_sorted(path: &Path, mut values: Vec<Value>) -> Result<(), String> {
    values.sort_by_key(|v| v.get("user").and_then(|x| x.as_i64()).unwrap_or(i64::MAX));
    let mut f = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    for v in &values {
        f.write_all(to_py_json(v).as_bytes())
            .and_then(|_| f.write_all(b"\n"))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Algos whose trained gradient comes from forward-mode `Dual` autodiff. They run in f64: torch
/// computes gradients in reverse-mode, and only the f64 forward-mode gradient faithfully proxies
/// it (matched upstream in the all-f64 era) — a f32 forward-mode gradient diverges badly on
/// chaotic `--secs` training. All other algos (analytic/reverse-mode: HLR, DASH, LogReg, FSRS-7;
/// and the non-trained models) run in f32 to match torch / the f32 upstream references.
fn algo_uses_f64(model_name: &str) -> bool {
    matches!(
        model_name,
        "ACT-R"
            | "Anki"
            | "DASH[ACT-R]"
            | "FSRSv1"
            | "FSRSv2"
            | "FSRSv3"
            | "FSRSv4"
            | "FSRS-4.5"
            | "FSRS-5"
            | "FSRS-6"
            | "FSRS-6-one-step"
            | "SM2-trainable"
    )
}

/// Main benchmark run.
pub fn run(cfg: &Config) -> Result<(), String> {
    // Per-algo precision (see `autodiff::ROUND_F32`): f32 for analytic/reverse-mode algos (match
    // the f32 upstream refs), f64 for forward-mode-`Dual` algos. Set once before the parallel loop.
    crate::autodiff::ROUND_F32.store(
        !algo_uses_f64(&cfg.model_name),
        std::sync::atomic::Ordering::Relaxed,
    );

    // `--partitions smart` (FSRS-7 only): load the precomputed covariance once, up front, so a
    // missing/bad file fails cleanly instead of panicking inside a rayon worker.
    if cfg.partitions == "smart" {
        if cfg.model_name != "FSRS-7" {
            return Err(format!(
                "--partitions smart only supports FSRS-7 (got {})",
                cfg.model_name
            ));
        }
        crate::smart::load_global("_smart/smart_preset_cov.json")?;
        if cfg.cluster_sweep {
            return run_smart_sweep(cfg);
        }
        if cfg.cluster_method == "hdbscan" || cfg.cluster_method == "optimal" {
            return Err(format!(
                "--cluster_method {} is only supported with --cluster_sweep",
                cfg.cluster_method
            ));
        }
    }

    let users = enumerate_users(&cfg.data_path, cfg.max_user_id)?;

    fs::create_dir_all("result").map_err(|e| e.to_string())?;
    let result_file = PathBuf::from(format!("result/{}.jsonl", cfg.evaluation_file_name()));

    let (existing, processed) = read_existing(&result_file);
    let todo: Vec<i64> = users.into_iter().filter(|u| !processed.contains(u)).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.num_processes)
        .build()
        .map_err(|e| e.to_string())?;

    let t_start = Instant::now();
    let results: Vec<Value> = pool.install(|| {
        todo.par_iter()
            .filter_map(|&user| match process_user(cfg, user) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("User {user}: {e}");
                    None
                }
            })
            .collect()
    });
    let makespan = t_start.elapsed().as_secs_f64();

    let mut all = existing;
    all.extend(results);
    let n = all.len();
    write_sorted(&result_file, all)?;

    eprintln!(
        "wrote {} users to {} (makespan {:.3}s, {} workers)",
        n,
        result_file.display(),
        makespan,
        cfg.num_processes
    );
    Ok(())
}

/// `--partitions smart --cluster_sweep`: run a whole clustering-experiment matrix in one pass per
/// user (sharing the per-deck training), writing one `result/<base>-smart-<suffix>.jsonl` each.
/// `--cluster_method hdbscan` selects the 16-config HDBSCAN matrix; otherwise the 30 hierarchical.
/// Resume is per-experiment: a user is recomputed unless present in EVERY experiment file.
fn run_smart_sweep(cfg: &Config) -> Result<(), String> {
    let hdbscan = cfg.cluster_method == "hdbscan";
    let optimal = cfg.cluster_method == "optimal";
    let kl = cfg.cluster_distance == "kl";
    let suffixes = if optimal {
        crate::models::fsrs_v7::opt_sweep_suffixes()
    } else if hdbscan {
        crate::models::fsrs_v7::hdbscan_sweep_suffixes(kl)
    } else {
        crate::models::fsrs_v7::hier_sweep_suffixes(kl)
    };
    let users = enumerate_users(&cfg.data_path, cfg.max_user_id)?;
    fs::create_dir_all("result").map_err(|e| e.to_string())?;

    let base = cfg.evaluation_file_name();
    let names: Vec<String> = suffixes.iter().map(|s| format!("{base}-smart-{s}")).collect();
    let n_exp = names.len();
    let paths: Vec<PathBuf> =
        names.iter().map(|n| PathBuf::from(format!("result/{n}.jsonl"))).collect();

    // Resume: a user is done iff present in EVERY experiment file (all written together).
    let existing: Vec<(Vec<Value>, std::collections::HashSet<i64>)> =
        paths.iter().map(|p| read_existing(p)).collect();
    let done: std::collections::HashSet<i64> = match existing.split_first() {
        Some(((_, first), rest)) => first
            .iter()
            .copied()
            .filter(|u| rest.iter().all(|(_, s)| s.contains(u)))
            .collect(),
        None => std::collections::HashSet::new(),
    };
    let todo: Vec<i64> = users.into_iter().filter(|u| !done.contains(u)).collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.num_processes)
        .build()
        .map_err(|e| e.to_string())?;
    let t_start = Instant::now();
    let results: Vec<Vec<Value>> = pool.install(|| {
        todo.par_iter()
            .map(|&user| match sweep_user(cfg, user, hdbscan) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("User {user}: {e}");
                    Vec::new()
                }
            })
            .collect()
    });
    let makespan = t_start.elapsed().as_secs_f64();

    let mut per_exp: Vec<Vec<Value>> = existing.into_iter().map(|(v, _)| v).collect();
    let mut new_users = 0usize;
    for user_vals in results {
        if user_vals.len() == n_exp {
            new_users += 1;
            for (i, v) in user_vals.into_iter().enumerate() {
                per_exp[i].push(v);
            }
        }
    }
    for (path, vals) in paths.iter().zip(per_exp) {
        write_sorted(path, vals)?;
    }
    eprintln!(
        "sweep: {} experiments, {} new users -> result/{}-smart-*.jsonl (makespan {:.1}s, {} workers)",
        n_exp, new_users, base, makespan, cfg.num_processes
    );
    Ok(())
}

/// Process one user through a whole sweep matrix → one result Value per experiment (matrix order).
fn sweep_user(cfg: &Config, user_id: i64, hdbscan: bool) -> Result<Vec<Value>, String> {
    let t0 = Instant::now();
    let raw = read_user_revlogs(&cfg.data_path, user_id)?;
    let mut ds = create_features(&raw, cfg)?;
    if ds.len() < 6 {
        return Err(format!("{user_id} does not have enough data."));
    }
    // smart clusters decks → use the deck partition map (missing cards ⇒ -1).
    let map = read_user_partition_map(&cfg.data_path, user_id, "deck")?;
    for r in &mut ds.rows {
        r.partition = *map.get(&r.card_id).unwrap_or(&-1);
    }
    let prep_s = t0.elapsed().as_secs_f64();

    let outs = if cfg.cluster_method == "optimal" {
        crate::models::fsrs_v7::process_optimal_sweep(&ds, cfg)
    } else if hdbscan {
        crate::models::fsrs_v7::process_hdbscan_sweep(&ds, cfg)
    } else {
        crate::models::fsrs_v7::process_smart_sweep(&ds, cfg)
    };
    let prep_share = prep_s / outs.len().max(1) as f64;
    let mut vals = Vec::with_capacity(outs.len());
    for (out, time_s) in outs {
        vals.push(evaluate(&out.eval_rows, &out.p, cfg, user_id, out.params, prep_share + time_s));
    }
    Ok(vals)
}
