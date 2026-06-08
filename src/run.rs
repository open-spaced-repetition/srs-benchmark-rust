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

    let raw = read_user_revlogs(&cfg.data_path, user_id)?;
    let mut ds = create_features(&raw, cfg)?;
    if ds.len() < 6 {
        return Err(format!("{user_id} does not have enough data."));
    }
    // `--secs --equalize_test_with_non_secs`: the train/test split is defined by the non-secs
    // pipeline (test only on reviews a non-`--secs` run would test). Features stay the secs ones.
    if cfg.use_secs_intervals && cfg.equalize_test_with_non_secs {
        ds.equalize_splits = Some(crate::features::build_equalize_splits(&raw, cfg, &ds)?);
    }
    // `--partitions deck|preset`: tag each row with its card's deck/preset partition.
    if cfg.partitions != "none" {
        let map = read_user_partition_map(&cfg.data_path, user_id, &cfg.partitions)?;
        for r in &mut ds.rows {
            r.partition = *map.get(&r.card_id).unwrap_or(&0);
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
        other => return Err(format!("model '{other}' not yet ported")),
    };

    let time_s = t0.elapsed().as_secs_f64();
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

/// Main benchmark run.
pub fn run(cfg: &Config) -> Result<(), String> {
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
