//! Builds the per-user result JSON object, mirroring `utils.evaluate`'s `stats` dict.
//!
//! Adds a `time` field (per-user wall seconds) per rule #3. ICI/smECE are omitted until
//! ported (they don't affect the LogLoss/size verification).

use serde_json::{Map, Value};

use crate::config::Config;
use crate::features::Row;
use crate::metrics;

/// Trained parameters to record under `"parameters"`: either a flat list or a
/// per-partition map (`{"0": [...]}`), matching `utils.result_parameters`.
pub enum Params {
    None,
    Flat(Vec<f64>),
    Partitioned(Vec<(String, Vec<f64>)>),
}

fn round6_value(x: f64) -> Value {
    // round(x,6); serde_json prints the shortest round-tripping repr (matches json.dumps).
    Value::from(metrics::round6(x))
}

/// Compute metrics and assemble the result object for one user.
pub fn evaluate(
    rows: &[Row],
    p: &[f64],
    cfg: &Config,
    user_id: i64,
    params: Params,
    time_s: f64,
) -> Value {
    let y: Vec<i64> = rows.iter().map(|r| r.y).collect();

    let mut m = Map::new();
    m.insert("RMSE".into(), round6_value(metrics::rmse(&y, p)));
    m.insert("LogLoss".into(), round6_value(metrics::log_loss(&y, p)));
    m.insert(
        "RMSE(bins)".into(),
        round6_value(metrics::rmse_bins(rows, p, None)),
    );
    match metrics::auc(&y, p) {
        Some(a) => m.insert("AUC".into(), round6_value(a)),
        None => m.insert("AUC".into(), Value::Null),
    };
    let (prec, rec) = metrics::precision_recall_at_90(&y, p);
    m.insert("precision@90".into(), round6_value(prec));
    m.insert("recall@90".into(), round6_value(rec));
    m.insert("MBE".into(), round6_value(metrics::mean_bias_error(&y, p)));

    let mut obj = Map::new();
    obj.insert("metrics".into(), Value::Object(m));
    obj.insert("user".into(), Value::from(user_id));
    obj.insert("size".into(), Value::from(y.len() as i64));

    match params {
        Params::None => {}
        Params::Flat(v) => {
            obj.insert(
                "parameters".into(),
                Value::Array(v.into_iter().map(round6_value).collect()),
            );
        }
        Params::Partitioned(parts) => {
            let mut pm = Map::new();
            for (k, v) in parts {
                pm.insert(k, Value::Array(v.into_iter().map(round6_value).collect()));
            }
            obj.insert("parameters".into(), Value::Object(pm));
        }
    }

    // rule #3: per-user processing time (seconds).
    obj.insert("time".into(), Value::from((time_s * 1e6).round() / 1e6));

    let _ = cfg;
    Value::Object(obj)
}
