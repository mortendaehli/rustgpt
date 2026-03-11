use std::path::PathBuf;

use crate::app::cli::help::{bench_sample_help, sample_help};
use crate::app::cli::{BenchSampleCommand, Command, SampleCommand};
use crate::core::config::SampleConfig;
use crate::core::error::{Result, RustGptError};

use super::parse_shared::{
    parse_device_kind, parse_f32, parse_u64, parse_usize, strip_benchmark_flags, take_value,
};

pub(super) fn parse_sample(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut checkpoint = None;
    let mut sample = SampleConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(sample_help(bin))),
            "--checkpoint" => {
                idx += 1;
                checkpoint = Some(PathBuf::from(take_value(&args, idx, "--checkpoint")?));
            }
            "--prompt" => {
                idx += 1;
                sample.prompt = take_value(&args, idx, "--prompt")?;
            }
            "--temperature" => {
                idx += 1;
                sample.temperature = parse_f32(&args, idx, "--temperature")?;
            }
            "--top-k" => {
                idx += 1;
                sample.top_k = parse_usize(&args, idx, "--top-k")?;
            }
            "--top-p" => {
                idx += 1;
                sample.top_p = parse_f32(&args, idx, "--top-p")?;
            }
            "--repetition-penalty" => {
                idx += 1;
                sample.repetition_penalty = parse_f32(&args, idx, "--repetition-penalty")?;
            }
            "--presence-penalty" => {
                idx += 1;
                sample.presence_penalty = parse_f32(&args, idx, "--presence-penalty")?;
            }
            "--frequency-penalty" => {
                idx += 1;
                sample.frequency_penalty = parse_f32(&args, idx, "--frequency-penalty")?;
            }
            "--max-new-tokens" => {
                idx += 1;
                sample.max_new_tokens = parse_usize(&args, idx, "--max-new-tokens")?;
            }
            "--samples" => {
                idx += 1;
                sample.samples = parse_usize(&args, idx, "--samples")?;
            }
            "--seed" => {
                idx += 1;
                sample.seed = parse_u64(&args, idx, "--seed")?;
            }
            "--device" => {
                idx += 1;
                sample.device = parse_device_kind(&args, idx, "--device")?;
            }
            "--profile" => sample.profile = true,
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown sample argument {other:?}\n\n{}",
                    sample_help(bin)
                )));
            }
        }
        idx += 1;
    }

    let checkpoint = checkpoint.ok_or_else(|| {
        RustGptError::Cli(format!(
            "missing required --checkpoint for sample\n\n{}",
            sample_help(bin)
        ))
    })?;

    Ok(Command::Sample(SampleCommand { checkpoint, sample }))
}

pub(super) fn parse_bench_sample(bin: &str, args: Vec<String>) -> Result<Command> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Command::Help(bench_sample_help(bin)));
    }
    let (forwarded, bench) = strip_benchmark_flags(&args)?;
    let Command::Sample(sample) = parse_sample(bin, forwarded)? else {
        unreachable!("parse_sample always returns Command::Sample");
    };
    Ok(Command::BenchSample(BenchSampleCommand { sample, bench }))
}
