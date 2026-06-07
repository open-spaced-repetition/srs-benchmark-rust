//! Per-user parquet loading.
//!
//! Mirrors `data_loader.py::UserDataLoader.load_user_data` for the revlogs read. Rows are
//! returned in the parquet's physical row order, which is the order `pandas.read_parquet`
//! preserves and from which `create_features` assigns `review_th = 1..n`.

use std::path::Path;

use arrow::array::Int64Array;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Raw revlog columns for one user, in file row order.
#[derive(Debug, Default, Clone)]
pub struct RawRevlogs {
    pub card_id: Vec<i64>,
    pub day_offset: Vec<i64>,
    pub rating: Vec<i64>,
    pub state: Vec<i64>,
    pub duration: Vec<i64>,
    pub elapsed_days: Vec<i64>,
    pub elapsed_seconds: Vec<i64>,
}

impl RawRevlogs {
    pub fn len(&self) -> usize {
        self.card_id.len()
    }
    pub fn is_empty(&self) -> bool {
        self.card_id.is_empty()
    }
}

/// Read all `*.parquet` files under `<data_path>/revlogs/user_id=<user_id>/`.
pub fn read_user_revlogs(data_path: &Path, user_id: i64) -> Result<RawRevlogs, String> {
    let dir = data_path
        .join("revlogs")
        .join(format!("user_id={}", user_id));
    if !dir.is_dir() {
        return Err(format!("revlog partition not found: {}", dir.display()));
    }

    // Collect parquet files; sort by name so multi-file partitions read deterministically
    // (matches pyarrow's lexicographic fragment ordering).
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read_dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "parquet").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no parquet files in {}", dir.display()));
    }

    let mut out = RawRevlogs::default();
    for f in files {
        read_one_parquet(&f, &mut out)?;
    }
    Ok(out)
}

fn read_one_parquet(path: &Path, out: &mut RawRevlogs) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("parquet builder {}: {e}", path.display()))?;
    let reader = builder
        .build()
        .map_err(|e| format!("parquet reader {}: {e}", path.display()))?;

    for batch in reader {
        let batch = batch.map_err(|e| format!("read batch {}: {e}", path.display()))?;
        push_col(&batch, "card_id", &mut out.card_id)?;
        push_col(&batch, "day_offset", &mut out.day_offset)?;
        push_col(&batch, "rating", &mut out.rating)?;
        push_col(&batch, "state", &mut out.state)?;
        push_col(&batch, "duration", &mut out.duration)?;
        push_col(&batch, "elapsed_days", &mut out.elapsed_days)?;
        push_col(&batch, "elapsed_seconds", &mut out.elapsed_seconds)?;
    }
    Ok(())
}

fn push_col(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    dst: &mut Vec<i64>,
) -> Result<(), String> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing column {name}"))?;
    let arr = col
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("column {name} is not Int64"))?;
    // No nulls expected in revlogs; values() is the raw buffer.
    dst.extend_from_slice(arr.values());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn reads_user1_3k() {
        let data = PathBuf::from(r"C:\Users\Andrew\anki-revlogs-3k");
        if !data.join("revlogs").is_dir() {
            eprintln!("dataset not present; skipping");
            return;
        }
        let r = read_user_revlogs(&data, 1).expect("read user 1");
        assert_eq!(r.len(), 22430, "row count for user 1");
        // Spot-check first rows from the schema dump: card_id 0,0,1 / rating 3,1,3.
        assert_eq!(&r.card_id[..3], &[0, 0, 1]);
        assert_eq!(&r.rating[..3], &[3, 1, 3]);
        assert_eq!(&r.elapsed_seconds[..3], &[-1, 85849, -1]);
    }
}
