#![recursion_limit = "256"]

//! RustGPT is organized into a few maintainable research-oriented layers:
//! `app` for CLI entrypoints, `core` for shared types and errors,
//! `data` for corpora/tokenizers/training examples,
//! `model` for the decoder-only transformer,
//! `train` for optimization and validation,
//! `infer` for sampling/chat, and `runtime`
//! for device/checkpoint/profile concerns.

mod app;
mod core;
mod data;
mod infer;
mod model;
mod runtime;
mod train;

use crate::core::error::Result;

pub fn run<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    match app::cli::parse_args(args)? {
        app::cli::Command::Help(text) => {
            println!("{text}");
            Ok(())
        }
        app::cli::Command::Train(command) => app::commands::train::run_train(command),
        app::cli::Command::BenchTrain(command) => app::commands::bench::run_bench_train(command),
        app::cli::Command::BenchCompareTrain(command) => {
            app::commands::bench::run_bench_compare_train(command)
        }
        app::cli::Command::InspectVocab(command) => {
            app::commands::train::run_inspect_vocab(command)
        }
        app::cli::Command::PrepareData(command) => {
            app::commands::prepare_data::run_prepare_data(command)
        }
        app::cli::Command::TrainTokenizer(command) => {
            app::commands::train_tokenizer::run_train_tokenizer(command)
        }
        app::cli::Command::Sample(command) => app::commands::sample::run_sample(command),
        app::cli::Command::BenchSample(command) => app::commands::bench::run_bench_sample(command),
        app::cli::Command::Chat(command) => app::commands::chat::run_chat(command),
        app::cli::Command::Eval(command) => app::commands::eval::run_eval(command),
        app::cli::Command::GpuInfo(command) => app::commands::gpu_info::run_gpu_info(command),
    }
}
