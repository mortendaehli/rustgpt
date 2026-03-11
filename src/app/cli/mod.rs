use std::path::PathBuf;

use crate::core::config::{
    BenchmarkConfig, ChatConfig, DataConfig, EvalConfig, GpuInfoConfig, InspectVocabConfig,
    PrepareDataConfig, SampleConfig, TrainConfig, TrainTokenizerConfig,
};
use crate::core::error::Result;

mod help;
mod parse_chat;
mod parse_eval;
mod parse_gpu_info;
mod parse_inspect_vocab;
mod parse_prepare_data;
mod parse_sample;
mod parse_shared;
mod parse_train;
mod parse_train_tokenizer;

#[cfg(test)]
mod tests;

use self::help::global_help;

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Help(String),
    Train(TrainCommand),
    BenchTrain(BenchTrainCommand),
    BenchCompareTrain(BenchCompareTrainCommand),
    InspectVocab(InspectVocabCommand),
    PrepareData(PrepareDataCommand),
    TrainTokenizer(TrainTokenizerCommand),
    Sample(SampleCommand),
    BenchSample(BenchSampleCommand),
    Chat(ChatCommand),
    Eval(EvalCommand),
    GpuInfo(GpuInfoCommand),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainCommand {
    pub data: DataConfig,
    pub train: TrainConfig,
    pub validation_data: Option<DataConfig>,
    pub checkpoint_out: Option<PathBuf>,
    pub best_checkpoint_out: Option<PathBuf>,
    pub resume: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InspectVocabCommand {
    pub data: DataConfig,
    pub inspect: InspectVocabConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrepareDataCommand {
    pub data: DataConfig,
    pub prepare: PrepareDataConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainTokenizerCommand {
    pub data: DataConfig,
    pub tokenizer: TrainTokenizerConfig,
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
pub struct EvalCommand {
    pub checkpoint: PathBuf,
    pub data: Option<DataConfig>,
    pub eval: EvalConfig,
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
        "train" => parse_train::parse_train(&bin, rest),
        "bench-train" => parse_train::parse_bench_train(&bin, rest),
        "bench-compare-train" => parse_train::parse_bench_compare_train(&bin, rest),
        "inspect-vocab" => parse_inspect_vocab::parse_inspect_vocab(&bin, rest),
        "prepare-data" => parse_prepare_data::parse_prepare_data(&bin, rest),
        "train-tokenizer" => parse_train_tokenizer::parse_train_tokenizer(&bin, rest),
        "sample" => parse_sample::parse_sample(&bin, rest),
        "bench-sample" => parse_sample::parse_bench_sample(&bin, rest),
        "chat" => parse_chat::parse_chat(&bin, rest),
        "eval" => parse_eval::parse_eval(&bin, rest),
        "gpu-info" => parse_gpu_info::parse_gpu_info(&bin, rest),
        other => parse_shared::unknown_command(other, &bin),
    }
}
