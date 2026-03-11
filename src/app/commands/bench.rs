use std::time::{Duration, Instant};

use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::app::cli::{
    BenchCompareTrainCommand, BenchSampleCommand, BenchTrainCommand, TrainCommand,
};
use crate::core::config::{BenchmarkConfig, DeviceKind};
use crate::core::error::Result;
use crate::core::rng::Rng;
use crate::engine::checkpoint::{load_inference_checkpoint, load_training_checkpoint};
use crate::engine::device::{
    CpuAutodiffBackend, CpuBackend, GpuAutodiffBackend, GpuBackend, ResolvedDeviceKind, cpu_device,
    gpu_device,
};
use crate::engine::generate::{SamplingStrategy, generate_sample};
use crate::engine::profile::RuntimeProfile;
use crate::engine::train::{TrainingArtifacts, prepare_training_run, run_training_steps};

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
    let resolved = ResolvedDeviceKind::resolve(command.train.train.device)?;
    let summary = match resolved {
        ResolvedDeviceKind::Cpu => bench_train_iterations::<CpuBackend, CpuAutodiffBackend>(
            &command.train,
            &command.bench,
            cpu_device(),
        )?,
        ResolvedDeviceKind::Gpu => bench_train_iterations::<GpuBackend, GpuAutodiffBackend>(
            &command.train,
            &command.bench,
            gpu_device(),
        )?,
    };
    print_bench_summary("train", &summary, &command.bench);
    Ok(())
}

pub fn run_bench_compare_train(command: BenchCompareTrainCommand) -> Result<()> {
    let cpu_summary = bench_train_iterations::<CpuBackend, CpuAutodiffBackend>(
        &command.train,
        &command.bench,
        cpu_device(),
    )?;
    let gpu_summary = bench_train_iterations::<GpuBackend, GpuAutodiffBackend>(
        &TrainCommand {
            train: crate::core::config::TrainConfig {
                device: DeviceKind::Gpu,
                ..command.train.train.clone()
            },
            ..command.train.clone()
        },
        &command.bench,
        gpu_device(),
    )?;

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
    let resolved = ResolvedDeviceKind::resolve(command.sample.sample.device)?;
    match resolved {
        ResolvedDeviceKind::Cpu => {
            run_bench_sample_impl::<CpuBackend>(command, cpu_device(), resolved)
        }
        ResolvedDeviceKind::Gpu => {
            run_bench_sample_impl::<GpuBackend>(command, gpu_device(), resolved)
        }
    }
}

fn run_bench_sample_impl<B: Backend>(
    command: BenchSampleCommand,
    device: B::Device,
    _resolved: ResolvedDeviceKind,
) -> Result<()> {
    let checkpoint = load_inference_checkpoint::<B>(&command.sample.checkpoint, &device)?;
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
        let _sample = generate_sample(
            &checkpoint.model,
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

fn bench_train_iterations<B, AD>(
    command: &TrainCommand,
    bench: &BenchmarkConfig,
    device: AD::Device,
) -> Result<TrainBenchSummary>
where
    B: Backend<Device = AD::Device>,
    AD: AutodiffBackend<InnerBackend = B>,
{
    let mut durations = Vec::with_capacity(bench.iterations);
    let aggregate_profile = RuntimeProfile::default();

    for bench_idx in 0..(bench.warmup + bench.iterations) {
        let profile = RuntimeProfile::default();
        let prepared = prepare_training_run(
            &command.data,
            &command.train,
            command.validation_data.as_ref(),
            command.resume.as_deref(),
        )?;
        let artifacts = if let Some(resume_path) = command.resume.as_deref() {
            let loaded = load_training_checkpoint::<AD>(resume_path, &device, &command.train)?;
            TrainingArtifacts {
                model: loaded.model,
                optimizer: loaded.optimizer,
            }
        } else {
            TrainingArtifacts::new(&prepared.model_config, &command.train, &device)?
        };

        let started_at = Instant::now();
        let _summary = run_training_steps::<B, AD>(
            artifacts,
            &prepared.tokenizer,
            &prepared.training_data,
            &command.train,
            prepared.starting_step,
            &device,
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
