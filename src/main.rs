//! Rust port of srs-benchmark `script.py`.

mod autodiff;
mod cluster;
mod config;
mod data;
mod eval;
mod features;
mod hdbscan;
mod interval;
mod metrics;
mod models;
#[cfg(feature = "neural")]
mod neural;
mod run;
mod smart;
mod split;
mod train;

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
