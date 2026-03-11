use std::time::{Duration, Instant};

use crate::app::cli::{
    BenchCompareTrainCommand, BenchSampleCommand, BenchTrainCommand, TrainCommand,
};
use crate::core::config::{BenchmarkConfig, DeviceKind};
use crate::core::error::Result;
use crate::core::rng::Rng;
use crate::data::checkpoint::load_checkpoint;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::profile::RuntimeProfile;
use crate::runtime::sampling::{SamplingStrategy, generate_sample_with_backend};
use crate::runtime::training::{prepare_training_run, run_training_steps};

#[derive(Debug)]
struct TrainBenchSummary {
    durations: Vec<Duration>,
    aggregate_profile: RuntimeProfile,
    steps: usize,
    batch_size: usize,
}

impl TrainBenchSummary {
    fn avg_duration(&self) -> Duration {
        average_duration(&self.durations)
    }
}

pub fn run_bench_train(command: BenchTrainCommand) -> Result<()> {
    let summary = bench_train_iterations(&command.train, &command.bench, None)?;
    print_bench_summary("train", &summary, &command.bench);
    Ok(())
}

pub fn run_bench_compare_train(command: BenchCompareTrainCommand) -> Result<()> {
    let cpu_summary =
        bench_train_iterations(&command.train, &command.bench, Some(DeviceKind::Cpu))?;
    let gpu_summary =
        bench_train_iterations(&command.train, &command.bench, Some(DeviceKind::Gpu))?;

    println!("RustGPT benchmark  mode=train-compare");
    println!(
        "cpu_avg={:.4?}  gpu_avg={:.4?}  speedup_vs_cpu={:.2}x",
        cpu_summary.avg_duration(),
        gpu_summary.avg_duration(),
        speedup_vs_cpu(&cpu_summary, &gpu_summary)
    );
    println!();
    print_bench_summary("train cpu", &cpu_summary, &command.bench);
    println!();
    print_bench_summary("train gpu", &gpu_summary, &command.bench);
    Ok(())
}

pub fn run_bench_sample(command: BenchSampleCommand) -> Result<()> {
    let checkpoint = load_checkpoint(&command.sample.checkpoint)?;
    let backend = ComputeBackend::from_model(&checkpoint.model, command.sample.sample.device)?;
    let mut durations = Vec::with_capacity(command.bench.iterations);
    let aggregate_profile = RuntimeProfile::default();
    let strategy = SamplingStrategy {
        temperature: command.sample.sample.temperature,
        top_k: command.sample.sample.top_k,
        top_p: command.sample.sample.top_p,
        repetition_penalty: command.sample.sample.repetition_penalty,
        presence_penalty: command.sample.sample.presence_penalty,
        frequency_penalty: command.sample.sample.frequency_penalty,
    };

    for bench_idx in 0..(command.bench.warmup + command.bench.iterations) {
        let profile = RuntimeProfile::default();
        let mut rng = Rng::from_seed(command.sample.sample.seed + bench_idx as u64);
        let started_at = Instant::now();
        let _sample = generate_sample_with_backend(
            &checkpoint.model,
            &backend,
            &checkpoint.tokenizer,
            &command.sample.sample.prompt,
            command.sample.sample.max_new_tokens,
            &strategy,
            Some(&profile),
            &mut rng,
        )?;
        let elapsed = started_at.elapsed();
        if bench_idx >= command.bench.warmup {
            durations.push(elapsed);
            aggregate_profile.merge_from(&profile);
        }
    }

    print_bench_header(
        "sample",
        &command.bench,
        &durations,
        None,
        None,
        Some(&aggregate_profile),
    );
    Ok(())
}

fn bench_train_iterations(
    command: &TrainCommand,
    bench: &BenchmarkConfig,
    device_override: Option<DeviceKind>,
) -> Result<TrainBenchSummary> {
    let mut durations = Vec::with_capacity(bench.iterations);
    let aggregate_profile = RuntimeProfile::default();

    for bench_idx in 0..(bench.warmup + bench.iterations) {
        let profile = RuntimeProfile::default();
        let mut prepared = prepare_training_run(
            &command.data,
            &command.train,
            command.validation_data.as_ref(),
            command.resume.as_deref(),
        )?;
        let mut backend = ComputeBackend::from_model(
            &prepared.model,
            device_override.unwrap_or(command.train.device),
        )?;

        let started_at = Instant::now();
        let _summary = run_training_steps(
            &mut prepared.model,
            &prepared.tokenizer,
            &prepared.training_data,
            &mut backend,
            &command.train,
            prepared.starting_step,
            Some(&profile),
            false,
            prepared.validation_data.as_ref(),
            None,
        )?;
        let elapsed = started_at.elapsed();
        if bench_idx >= bench.warmup {
            durations.push(elapsed);
            aggregate_profile.merge_from(&profile);
        }
    }

    Ok(TrainBenchSummary {
        durations,
        aggregate_profile,
        steps: command.train.steps,
        batch_size: usize::max(1, command.train.batch_size),
    })
}

fn print_bench_summary(label: &str, summary: &TrainBenchSummary, bench: &BenchmarkConfig) {
    print_bench_header(
        label,
        bench,
        &summary.durations,
        Some(summary.steps),
        Some(summary.batch_size),
        Some(&summary.aggregate_profile),
    );
}

fn print_bench_header(
    label: &str,
    bench: &BenchmarkConfig,
    durations: &[Duration],
    steps_per_iter: Option<usize>,
    batch_size: Option<usize>,
    aggregate_profile: Option<&RuntimeProfile>,
) {
    let total = durations.iter().copied().sum::<Duration>();
    let avg = average_duration(durations);
    let min = durations.iter().copied().min().unwrap_or(Duration::ZERO);
    let max = durations.iter().copied().max().unwrap_or(Duration::ZERO);

    println!("RustGPT benchmark  mode={label}");
    println!("iterations={}  warmup={}", bench.iterations, bench.warmup);
    println!("avg={avg:.4?}  min={min:.4?}  max={max:.4?}  total={total:.4?}");
    if let Some(steps_per_iter) = steps_per_iter {
        let batch_size = batch_size.unwrap_or(1);
        let steps_per_second = if avg.is_zero() {
            0.0
        } else {
            steps_per_iter as f64 / avg.as_secs_f64()
        };
        let sequences_per_second = steps_per_second * batch_size as f64;
        println!(
            "steps_per_iter={}  batch_size={}  steps_per_second={steps_per_second:.2}  sequences_per_second={sequences_per_second:.2}",
            steps_per_iter, batch_size,
        );
    }
    if let Some(aggregate_profile) = aggregate_profile {
        println!("stage timings:");
        println!("{}", aggregate_profile.format_table());
    }
}

fn average_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        Duration::ZERO
    } else {
        let total = durations.iter().copied().sum::<Duration>();
        Duration::from_secs_f64(total.as_secs_f64() / durations.len() as f64)
    }
}

fn speedup_vs_cpu(cpu: &TrainBenchSummary, gpu: &TrainBenchSummary) -> f64 {
    let cpu_avg = cpu.avg_duration().as_secs_f64();
    let gpu_avg = gpu.avg_duration().as_secs_f64();
    if gpu_avg == 0.0 {
        0.0
    } else {
        cpu_avg / gpu_avg
    }
}
