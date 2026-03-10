use std::path::PathBuf;

use crate::core::config::{
    BenchmarkConfig, BoundaryMode, ChatConfig, DataConfig, DataFormat, DeviceKind, GpuInfoConfig,
    InspectVocabConfig, SampleConfig, TrainConfig,
};
use crate::core::error::{Result, RustGptError};

mod help;

#[cfg(test)]
mod tests;

use self::help::{
    bench_compare_train_help, bench_sample_help, bench_train_help, chat_help, global_help,
    gpu_info_help, inspect_vocab_help, sample_help, train_help,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Help(String),
    Train(TrainCommand),
    BenchTrain(BenchTrainCommand),
    BenchCompareTrain(BenchCompareTrainCommand),
    InspectVocab(InspectVocabCommand),
    Sample(SampleCommand),
    BenchSample(BenchSampleCommand),
    Chat(ChatCommand),
    GpuInfo(GpuInfoCommand),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainCommand {
    pub data: DataConfig,
    pub train: TrainConfig,
    pub checkpoint_out: Option<PathBuf>,
    pub resume: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectVocabCommand {
    pub data: DataConfig,
    pub inspect: InspectVocabConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchTrainCommand {
    pub train: TrainCommand,
    pub bench: BenchmarkConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchCompareTrainCommand {
    pub train: TrainCommand,
    pub bench: BenchmarkConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampleCommand {
    pub checkpoint: PathBuf,
    pub sample: SampleConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchSampleCommand {
    pub sample: SampleCommand,
    pub bench: BenchmarkConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatCommand {
    pub checkpoint: PathBuf,
    pub chat: ChatConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfoCommand {
    pub gpu: GpuInfoConfig,
}

pub fn parse_args<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let bin = args.next().unwrap_or_else(|| "rustgpt".to_string());
    let Some(command) = args.next() else {
        return Ok(Command::Help(global_help(&bin)));
    };
    let rest = args.collect::<Vec<_>>();

    match command.as_str() {
        "help" | "-h" | "--help" => Ok(Command::Help(global_help(&bin))),
        "train" => parse_train(&bin, rest),
        "bench-train" => parse_bench_train(&bin, rest),
        "bench-compare-train" => parse_bench_compare_train(&bin, rest),
        "inspect-vocab" => parse_inspect_vocab(&bin, rest),
        "sample" => parse_sample(&bin, rest),
        "bench-sample" => parse_bench_sample(&bin, rest),
        "chat" => parse_chat(&bin, rest),
        "gpu-info" => parse_gpu_info(&bin, rest),
        other => Err(RustGptError::Cli(format!(
            "unknown command {other:?}\n\n{}",
            global_help(&bin)
        ))),
    }
}

fn parse_train(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut data = DataConfig::default();
    let mut train = TrainConfig::default();
    let mut checkpoint_out = None;
    let mut resume = None;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(train_help(bin))),
            "--data" => {
                idx += 1;
                data.data_path = take_value(&args, idx, "--data")?.into();
            }
            "--checkpoint-out" => {
                idx += 1;
                checkpoint_out = Some(PathBuf::from(take_value(&args, idx, "--checkpoint-out")?));
            }
            "--resume" => {
                idx += 1;
                resume = Some(PathBuf::from(take_value(&args, idx, "--resume")?));
            }
            "--format" => {
                idx += 1;
                let value = take_value(&args, idx, "--format")?;
                data.format = DataFormat::parse(&value).ok_or_else(|| {
                    RustGptError::Cli(format!(
                        "invalid value for --format: {value:?}. Expected lines or text."
                    ))
                })?;
            }
            "--no-shuffle" => data.shuffle = false,
            "--lowercase" => data.lowercase = true,
            "--steps" => {
                idx += 1;
                train.steps = parse_usize(&args, idx, "--steps")?;
            }
            "--batch-size" => {
                idx += 1;
                train.batch_size = parse_usize(&args, idx, "--batch-size")?;
            }
            "--seed" => {
                idx += 1;
                train.seed = parse_u64(&args, idx, "--seed")?;
            }
            "--sample-every" => {
                idx += 1;
                train.sample_every = parse_usize(&args, idx, "--sample-every")?;
            }
            "--block-size" => {
                idx += 1;
                train.block_size = parse_usize(&args, idx, "--block-size")?;
            }
            "--n-layer" => {
                idx += 1;
                train.n_layer = parse_usize(&args, idx, "--n-layer")?;
            }
            "--n-embd" => {
                idx += 1;
                train.n_embd = parse_usize(&args, idx, "--n-embd")?;
            }
            "--n-head" => {
                idx += 1;
                train.n_head = parse_usize(&args, idx, "--n-head")?;
            }
            "--lr" => {
                idx += 1;
                train.learning_rate = parse_f32(&args, idx, "--lr")?;
            }
            "--beta1" => {
                idx += 1;
                train.beta1 = parse_f32(&args, idx, "--beta1")?;
            }
            "--beta2" => {
                idx += 1;
                train.beta2 = parse_f32(&args, idx, "--beta2")?;
            }
            "--eps" => {
                idx += 1;
                train.eps_adam = parse_f32(&args, idx, "--eps")?;
            }
            "--separate-eos" => train.boundary_mode = BoundaryMode::SeparateBosEos,
            "--device" => {
                idx += 1;
                train.device = parse_device_kind(&args, idx, "--device")?;
            }
            "--profile" => train.profile = true,
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown train argument {other:?}\n\n{}",
                    train_help(bin)
                )));
            }
        }
        idx += 1;
    }

    Ok(Command::Train(TrainCommand {
        data,
        train,
        checkpoint_out,
        resume,
    }))
}

fn parse_bench_train(bin: &str, args: Vec<String>) -> Result<Command> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Command::Help(bench_train_help(bin)));
    }
    let (forwarded, bench) = strip_benchmark_flags(&args)?;
    let Command::Train(train) = parse_train(bin, forwarded)? else {
        unreachable!("parse_train always returns Command::Train");
    };
    Ok(Command::BenchTrain(BenchTrainCommand { train, bench }))
}

fn parse_bench_compare_train(bin: &str, args: Vec<String>) -> Result<Command> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Command::Help(bench_compare_train_help(bin)));
    }
    let (forwarded, bench) = strip_benchmark_flags(&args)?;
    let Command::Train(train) = parse_train(bin, forwarded)? else {
        unreachable!("parse_train always returns Command::Train");
    };
    Ok(Command::BenchCompareTrain(BenchCompareTrainCommand {
        train,
        bench,
    }))
}

fn parse_inspect_vocab(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut data = DataConfig::default();
    let mut inspect = InspectVocabConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(inspect_vocab_help(bin))),
            "--data" => {
                idx += 1;
                data.data_path = take_value(&args, idx, "--data")?.into();
            }
            "--format" => {
                idx += 1;
                let value = take_value(&args, idx, "--format")?;
                data.format = DataFormat::parse(&value).ok_or_else(|| {
                    RustGptError::Cli(format!(
                        "invalid value for --format: {value:?}. Expected lines or text."
                    ))
                })?;
            }
            "--lowercase" => data.lowercase = true,
            "--show-tokens" => {
                idx += 1;
                inspect.show_tokens = parse_usize(&args, idx, "--show-tokens")?;
            }
            "--separate-eos" => inspect.boundary_mode = BoundaryMode::SeparateBosEos,
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown inspect-vocab argument {other:?}\n\n{}",
                    inspect_vocab_help(bin)
                )));
            }
        }
        idx += 1;
    }

    Ok(Command::InspectVocab(InspectVocabCommand { data, inspect }))
}

fn parse_sample(bin: &str, args: Vec<String>) -> Result<Command> {
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

fn parse_bench_sample(bin: &str, args: Vec<String>) -> Result<Command> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Command::Help(bench_sample_help(bin)));
    }
    let (forwarded, bench) = strip_benchmark_flags(&args)?;
    let Command::Sample(sample) = parse_sample(bin, forwarded)? else {
        unreachable!("parse_sample always returns Command::Sample");
    };
    Ok(Command::BenchSample(BenchSampleCommand { sample, bench }))
}

fn parse_chat(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut checkpoint = None;
    let mut chat = ChatConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(chat_help(bin))),
            "--checkpoint" => {
                idx += 1;
                checkpoint = Some(PathBuf::from(take_value(&args, idx, "--checkpoint")?));
            }
            "--system" => {
                idx += 1;
                chat.system_prompt = take_value(&args, idx, "--system")?;
            }
            "--temperature" => {
                idx += 1;
                chat.temperature = parse_f32(&args, idx, "--temperature")?;
            }
            "--max-new-tokens" => {
                idx += 1;
                chat.max_new_tokens = parse_usize(&args, idx, "--max-new-tokens")?;
            }
            "--seed" => {
                idx += 1;
                chat.seed = parse_u64(&args, idx, "--seed")?;
            }
            "--device" => {
                idx += 1;
                chat.device = parse_device_kind(&args, idx, "--device")?;
            }
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown chat argument {other:?}\n\n{}",
                    chat_help(bin)
                )));
            }
        }
        idx += 1;
    }

    let checkpoint = checkpoint.ok_or_else(|| {
        RustGptError::Cli(format!(
            "missing required --checkpoint for chat\n\n{}",
            chat_help(bin)
        ))
    })?;

    Ok(Command::Chat(ChatCommand { checkpoint, chat }))
}

fn parse_gpu_info(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut gpu = GpuInfoConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(gpu_info_help(bin))),
            "--device" => {
                idx += 1;
                gpu.device = parse_device_kind(&args, idx, "--device")?;
            }
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown gpu-info argument {other:?}\n\n{}",
                    gpu_info_help(bin)
                )));
            }
        }
        idx += 1;
    }

    Ok(Command::GpuInfo(GpuInfoCommand { gpu }))
}

fn take_value(args: &[String], idx: usize, flag: &str) -> Result<String> {
    args.get(idx).cloned().ok_or_else(|| {
        RustGptError::Cli(format!(
            "missing value for {flag}. Run `rustgpt help` for usage."
        ))
    })
}

fn parse_usize(args: &[String], idx: usize, flag: &str) -> Result<usize> {
    let value = take_value(args, idx, flag)?;
    value
        .parse::<usize>()
        .map_err(|_| RustGptError::Cli(format!("invalid integer for {flag}: {value:?}")))
}

fn parse_u64(args: &[String], idx: usize, flag: &str) -> Result<u64> {
    let value = take_value(args, idx, flag)?;
    value
        .parse::<u64>()
        .map_err(|_| RustGptError::Cli(format!("invalid integer for {flag}: {value:?}")))
}

fn parse_f32(args: &[String], idx: usize, flag: &str) -> Result<f32> {
    let value = take_value(args, idx, flag)?;
    value
        .parse::<f32>()
        .map_err(|_| RustGptError::Cli(format!("invalid float for {flag}: {value:?}")))
}

fn parse_device_kind(args: &[String], idx: usize, flag: &str) -> Result<DeviceKind> {
    let value = take_value(args, idx, flag)?;
    DeviceKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected cpu, auto, or gpu."
        ))
    })
}

fn strip_benchmark_flags(args: &[String]) -> Result<(Vec<String>, BenchmarkConfig)> {
    let mut forwarded = Vec::new();
    let mut bench = BenchmarkConfig::default();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--iters" => {
                idx += 1;
                bench.iterations = parse_usize(args, idx, "--iters")?;
            }
            "--warmup" => {
                idx += 1;
                bench.warmup = parse_usize(args, idx, "--warmup")?;
            }
            other => forwarded.push(other.to_string()),
        }
        idx += 1;
    }
    Ok((forwarded, bench))
}
