//! Rust port of srs-benchmark `script.py`.

mod config;
mod data;
mod eval;
mod features;
mod metrics;
mod models;
mod run;
mod split;

use clap::Parser;
use config::{Cli, Config};

fn main() {
    let cli = Cli::parse();
    let config = Config::from_cli(&cli);

    if let Err(e) = run::run(&config) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
