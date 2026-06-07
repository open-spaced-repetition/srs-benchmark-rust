//! Rust port of srs-benchmark `script.py`.
//!
//! Entry point. Phase 0 scaffold: parse the CLI and report the resolved config. The
//! per-user pipeline (data load → features → split → train → eval → jsonl) is built out in
//! subsequent phases — see CLAUDE.md.

mod config;

use clap::Parser;
use config::{Cli, Config};

fn main() {
    let cli = Cli::parse();
    let config = Config::from_cli(&cli);

    eprintln!("srs-benchmark-rust (scaffold)");
    eprintln!("  algo         = {}", config.model_name);
    eprintln!("  output       = result/{}.jsonl", config.evaluation_file_name());
    eprintln!("  data_path    = {}", config.data_path.display());
    eprintln!("  processes    = {}", config.num_processes);
    eprintln!("  n_splits     = {}", config.n_splits);
    eprintln!("  short/secs   = {}/{}", config.include_short_term, config.use_secs_intervals);
    eprintln!("  default      = {}", config.default_params);

    // Touch a rayon symbol so the dependency is exercised even in the scaffold.
    let _threads = rayon::current_num_threads();
}
