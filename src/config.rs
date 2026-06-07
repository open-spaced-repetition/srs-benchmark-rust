//! CLI argument parsing and derived configuration.
//!
//! This mirrors `config.py` in the upstream Python `srs-benchmark` exactly: the same
//! flags (rule #4 — commands stay the same) and the same output-filename derivation, so
//! a Rust run writes to `result/<name>.jsonl` with the identical `<name>` as Python.

use clap::Parser;
use std::path::PathBuf;

/// Command-line options. Names and defaults match `config.py::create_parser`.
#[derive(Parser, Debug, Clone)]
#[command(name = "script", about = "Rust port of srs-benchmark")]
pub struct Cli {
    /// Number of worker threads (Python: process count).
    #[arg(long, default_value_t = 8)]
    pub processes: usize,

    /// Comma/space-separated CUDA device IDs (e.g. "0,1" or "all"). Unused by CPU models.
    #[arg(long)]
    pub gpus: Option<String>,

    /// For local development.
    #[arg(long, default_value_t = false)]
    pub dev: bool,

    /// Maximum user ID to process (inclusive).
    #[arg(long = "max-user-id")]
    pub max_user_id: Option<i64>,

    /// Use partitions instead of presets: none | deck | preset.
    #[arg(long, default_value = "none", value_parser = ["none", "deck", "preset"])]
    pub partitions: String,

    /// Enable recency weighting during training.
    #[arg(long, default_value_t = false)]
    pub recency: bool,

    /// Evaluate default parameters (no training).
    #[arg(long, default_value_t = false)]
    pub default: bool,

    /// FSRS-5/6 with only S0 initialization.
    #[arg(long = "S0", default_value_t = false)]
    pub s0: bool,

    /// FSRS-7 scheduling penalties.
    #[arg(long, default_value_t = false)]
    pub sched_penalties: bool,

    /// Treat Hard and Easy as Good.
    #[arg(long, default_value_t = false)]
    pub two_buttons: bool,

    /// Path to revlogs/*.parquet.
    #[arg(long, default_value = "../anki-revlogs-10k")]
    pub data: PathBuf,

    /// Use elapsed_seconds as the interval instead of days.
    #[arg(long, default_value_t = false)]
    pub secs: bool,

    /// Enable duration feature (LSTM only).
    #[arg(long, default_value_t = false)]
    pub duration: bool,

    /// Exclude reviews with elapsed_days=0 from the test set.
    #[arg(long, default_value_t = false)]
    pub no_test_same_day: bool,

    /// Exclude reviews with elapsed_days=0 from the train set.
    #[arg(long, default_value_t = false)]
    pub no_train_same_day: bool,

    /// Only test with reviews that would be included in non-secs tests.
    #[arg(long, default_value_t = false)]
    pub equalize_test_with_non_secs: bool,

    /// Save raw predictions to raw/<name>.jsonl.
    #[arg(long, default_value_t = false)]
    pub raw: bool,

    /// Save per-user evaluation TSVs.
    #[arg(long, default_value_t = false)]
    pub file: bool,

    /// Save evaluation plots.
    #[arg(long, default_value_t = false)]
    pub plot: bool,

    /// Algorithm name.
    #[arg(long, default_value = "FSRSv3")]
    pub algo: String,

    /// Include short-term (same-day) reviews.
    #[arg(long, default_value_t = false)]
    pub short: bool,

    /// Save model weights.
    #[arg(long, default_value_t = false)]
    pub weights: bool,

    /// Train and test on the same data.
    #[arg(long, default_value_t = false)]
    pub train_equals_test: bool,

    /// Number of TimeSeriesSplit folds.
    #[arg(long, default_value_t = 5)]
    pub n_splits: usize,

    /// Batch size for training models.
    #[arg(long, default_value_t = 512)]
    pub batch_size: usize,

    /// Max sequence length for batching inputs.
    #[arg(long, default_value_t = 64)]
    pub max_seq_len: usize,

    /// PyTorch intra-op threads (parity flag; affects nothing in Rust yet).
    #[arg(long, default_value_t = 1)]
    pub torch_num_threads: usize,
}

/// Resolved configuration derived from [`Cli`]. Mirrors `config.py::Config`.
#[derive(Debug, Clone)]
pub struct Config {
    pub model_name: String,
    pub default_params: bool,
    pub only_s0: bool,
    pub sched_penalties: bool,
    pub two_buttons: bool,
    pub include_short_term: bool,
    pub use_secs_intervals: bool,
    pub lstm_use_duration: bool,
    pub use_recency_weighting: bool,
    pub no_test_same_day: bool,
    pub no_train_same_day: bool,
    pub equalize_test_with_non_secs: bool,
    pub train_equals_test: bool,
    pub partitions: String,
    pub dev_mode: bool,

    pub max_user_id: Option<i64>,
    pub num_processes: usize,
    pub data_path: PathBuf,
    pub n_splits: usize,
    pub batch_size: usize,
    pub max_seq_len: usize,

    pub save_raw_output: bool,
    pub save_evaluation_file: bool,
    pub generate_plots: bool,
    pub save_weights: bool,

    // Stability bounds (config.py).
    pub s_min: f64,
    pub init_s_max: f64,
    pub s_max: f64,
    pub seed: u64,

    base_file_name: String,
}

impl Config {
    pub fn from_cli(cli: &Cli) -> Self {
        let model_name = cli.algo.clone();

        // base_file_name derivation — must match config.py byte-for-byte.
        let mut parts: Vec<String> = vec![model_name.clone()];
        if cli.default {
            parts.push("-default".into());
        }
        if cli.s0 {
            parts.push("-S0".into());
        }
        if cli.sched_penalties {
            parts.push("-sched_penalties".into());
        }
        if cli.two_buttons {
            parts.push("-binary".into());
        }
        if cli.short {
            parts.push("-short".into());
        }
        if cli.secs {
            parts.push("-secs".into());
        }
        if model_name == "LSTM" && cli.duration {
            parts.push("-duration".into());
        }
        if cli.recency {
            parts.push("-recency".into());
        }
        if cli.no_test_same_day {
            parts.push("-no_test_same_day".into());
        }
        if cli.no_train_same_day {
            parts.push("-no_train_same_day".into());
        }
        if cli.equalize_test_with_non_secs {
            parts.push("-equalize_test_with_non_secs".into());
        }
        if cli.train_equals_test {
            parts.push("-train_equals_test".into());
        }
        if cli.partitions != "none" {
            parts.push(format!("-{}", cli.partitions));
        }
        if cli.dev {
            parts.push("-dev".into());
        }
        let base_file_name = parts.concat();

        // s_min logic from config.py.
        let s_min_base = if cli.secs { 0.0001 } else { 0.01 };
        let s_min = if model_name.starts_with("FSRS-6") {
            if !cli.secs {
                0.001
            } else {
                s_min_base
            }
        } else {
            s_min_base
        };

        Config {
            model_name,
            default_params: cli.default,
            only_s0: cli.s0,
            sched_penalties: cli.sched_penalties,
            two_buttons: cli.two_buttons,
            include_short_term: cli.short,
            use_secs_intervals: cli.secs,
            lstm_use_duration: cli.duration,
            use_recency_weighting: cli.recency,
            no_test_same_day: cli.no_test_same_day,
            no_train_same_day: cli.no_train_same_day,
            equalize_test_with_non_secs: cli.equalize_test_with_non_secs,
            train_equals_test: cli.train_equals_test,
            partitions: cli.partitions.clone(),
            dev_mode: cli.dev,
            max_user_id: cli.max_user_id,
            num_processes: cli.processes,
            data_path: cli.data.clone(),
            n_splits: cli.n_splits,
            batch_size: cli.batch_size,
            max_seq_len: cli.max_seq_len,
            save_raw_output: cli.raw,
            save_evaluation_file: cli.file,
            generate_plots: cli.plot,
            save_weights: cli.weights,
            s_min,
            init_s_max: 100.0,
            s_max: 36500.0,
            seed: 42,
            base_file_name,
        }
    }

    /// Base name for output files, e.g. "FSRS-7-default-short-secs".
    pub fn evaluation_file_name(&self) -> &str {
        &self.base_file_name
    }
}
