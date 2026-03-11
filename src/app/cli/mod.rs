use std::path::PathBuf;

use crate::core::config::{
    ActivationKind, BenchmarkConfig, BoundaryMode, ChatConfig, ChatTemplateKind, DataConfig,
    DataFormat, DeviceKind, EvalConfig, GpuInfoConfig, InspectVocabConfig, LrScheduleKind,
    PositionEncodingKind, PrepareDataConfig, SampleConfig, TokenizerModelKind, TrainConfig,
    TrainMode, TrainTokenizerConfig,
};
use crate::core::error::{Result, RustGptError};

mod help;

#[cfg(test)]
mod tests;

use self::help::{
    bench_compare_train_help, bench_sample_help, bench_train_help, chat_help, eval_help,
    global_help, gpu_info_help, inspect_vocab_help, prepare_data_help, sample_help, train_help,
    train_tokenizer_help,
};

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
        "train" => parse_train(&bin, rest),
        "bench-train" => parse_bench_train(&bin, rest),
        "bench-compare-train" => parse_bench_compare_train(&bin, rest),
        "inspect-vocab" => parse_inspect_vocab(&bin, rest),
        "prepare-data" => parse_prepare_data(&bin, rest),
        "train-tokenizer" => parse_train_tokenizer(&bin, rest),
        "sample" => parse_sample(&bin, rest),
        "bench-sample" => parse_bench_sample(&bin, rest),
        "chat" => parse_chat(&bin, rest),
        "eval" => parse_eval(&bin, rest),
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
    let mut validation_data = None;
    let mut checkpoint_out = None;
    let mut best_checkpoint_out = None;
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
            "--best-checkpoint-out" => {
                idx += 1;
                best_checkpoint_out = Some(PathBuf::from(take_value(
                    &args,
                    idx,
                    "--best-checkpoint-out",
                )?));
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
                        "invalid value for --format: {value:?}. Expected lines, text, jsonl-text, jsonl-chat, parquet-text, or parquet-chat."
                    ))
                })?;
            }
            "--tokenizer" => {
                idx += 1;
                data.tokenizer_path = Some(PathBuf::from(take_value(&args, idx, "--tokenizer")?));
            }
            "--bos-token" => {
                idx += 1;
                data.tokenizer_bos = Some(take_value(&args, idx, "--bos-token")?);
            }
            "--eos-token" => {
                idx += 1;
                data.tokenizer_eos = Some(take_value(&args, idx, "--eos-token")?);
            }
            "--chat-template" => {
                idx += 1;
                data.chat_template = parse_chat_template_kind(&args, idx, "--chat-template")?;
            }
            "--no-shuffle" => data.shuffle = false,
            "--lowercase" => data.lowercase = true,
            "--steps" => {
                idx += 1;
                train.steps = parse_usize(&args, idx, "--steps")?;
            }
            "--valid-data" => {
                idx += 1;
                validation_data
                    .get_or_insert_with(DataConfig::default)
                    .data_path = take_value(&args, idx, "--valid-data")?.into();
            }
            "--valid-format" => {
                idx += 1;
                let value = take_value(&args, idx, "--valid-format")?;
                validation_data.get_or_insert_with(DataConfig::default).format =
                    DataFormat::parse(&value).ok_or_else(|| {
                        RustGptError::Cli(format!(
                            "invalid value for --valid-format: {value:?}. Expected lines, text, jsonl-text, jsonl-chat, parquet-text, or parquet-chat."
                        ))
                    })?;
            }
            "--valid-lowercase" => {
                validation_data
                    .get_or_insert_with(DataConfig::default)
                    .lowercase = true;
            }
            "--valid-chat-template" => {
                idx += 1;
                validation_data
                    .get_or_insert_with(DataConfig::default)
                    .chat_template = parse_chat_template_kind(&args, idx, "--valid-chat-template")?;
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
            "--n-kv-head" => {
                idx += 1;
                train.n_kv_head = parse_usize(&args, idx, "--n-kv-head")?;
            }
            "--tied-embeddings" => train.tie_embeddings = true,
            "--activation" => {
                idx += 1;
                train.activation = parse_activation_kind(&args, idx, "--activation")?;
            }
            "--position" | "--position-encoding" => {
                idx += 1;
                train.position_encoding = parse_position_encoding_kind(&args, idx, "--position")?;
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
            "--weight-decay" => {
                idx += 1;
                train.weight_decay = parse_f32(&args, idx, "--weight-decay")?;
            }
            "--warmup-steps" => {
                idx += 1;
                train.warmup_steps = parse_usize(&args, idx, "--warmup-steps")?;
            }
            "--lr-schedule" => {
                idx += 1;
                train.lr_schedule = parse_lr_schedule_kind(&args, idx, "--lr-schedule")?;
            }
            "--grad-clip" => {
                idx += 1;
                train.grad_clip = parse_f32(&args, idx, "--grad-clip")?;
            }
            "--valid-ratio" => {
                idx += 1;
                train.validation_ratio = parse_f32(&args, idx, "--valid-ratio")?;
            }
            "--valid-max-examples" => {
                idx += 1;
                train.validation_max_examples = parse_usize(&args, idx, "--valid-max-examples")?;
            }
            "--separate-eos" => train.boundary_mode = BoundaryMode::SeparateBosEos,
            "--mode" => {
                idx += 1;
                let value = take_value(&args, idx, "--mode")?;
                train.mode = TrainMode::parse(&value).ok_or_else(|| {
                    RustGptError::Cli(format!(
                        "invalid value for --mode: {value:?}. Expected auto, pretrain, or sft."
                    ))
                })?;
            }
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

    if let Some(valid) = &mut validation_data {
        if valid.format == DataFormat::Lines {
            valid.format = data.format;
        }
        if valid.chat_template == ChatTemplateKind::SimpleTranscript
            && data.chat_template != ChatTemplateKind::SimpleTranscript
        {
            valid.chat_template = data.chat_template;
        }
        if !valid.lowercase && data.lowercase {
            valid.lowercase = true;
        }
    }

    Ok(Command::Train(TrainCommand {
        data,
        train,
        validation_data,
        checkpoint_out,
        best_checkpoint_out,
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
                        "invalid value for --format: {value:?}. Expected lines, text, jsonl-text, jsonl-chat, parquet-text, or parquet-chat."
                    ))
                })?;
            }
            "--tokenizer" => {
                idx += 1;
                data.tokenizer_path = Some(PathBuf::from(take_value(&args, idx, "--tokenizer")?));
            }
            "--bos-token" => {
                idx += 1;
                data.tokenizer_bos = Some(take_value(&args, idx, "--bos-token")?);
            }
            "--eos-token" => {
                idx += 1;
                data.tokenizer_eos = Some(take_value(&args, idx, "--eos-token")?);
            }
            "--chat-template" => {
                idx += 1;
                data.chat_template = parse_chat_template_kind(&args, idx, "--chat-template")?;
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

fn parse_prepare_data(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut data = DataConfig::default();
    let mut prepare = PrepareDataConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(prepare_data_help(bin))),
            "--data" => {
                idx += 1;
                data.data_path = take_value(&args, idx, "--data")?.into();
            }
            "--format" => {
                idx += 1;
                let value = take_value(&args, idx, "--format")?;
                data.format = parse_data_format(&value, "--format")?;
            }
            "--chat-template" => {
                idx += 1;
                data.chat_template = parse_chat_template_kind(&args, idx, "--chat-template")?;
            }
            "--lowercase" => data.lowercase = true,
            "--out" => {
                idx += 1;
                prepare.output_path = PathBuf::from(take_value(&args, idx, "--out")?);
            }
            "--out-format" => {
                idx += 1;
                let value = take_value(&args, idx, "--out-format")?;
                prepare.output_format = parse_data_format(&value, "--out-format")?;
            }
            "--pretty" => prepare.pretty = true,
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown prepare-data argument {other:?}\n\n{}",
                    prepare_data_help(bin)
                )));
            }
        }
        idx += 1;
    }

    Ok(Command::PrepareData(PrepareDataCommand { data, prepare }))
}

fn parse_train_tokenizer(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut data = DataConfig::default();
    let mut tokenizer = TrainTokenizerConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(train_tokenizer_help(bin))),
            "--data" => {
                idx += 1;
                data.data_path = take_value(&args, idx, "--data")?.into();
            }
            "--format" => {
                idx += 1;
                let value = take_value(&args, idx, "--format")?;
                data.format = parse_data_format(&value, "--format")?;
            }
            "--chat-template" => {
                idx += 1;
                data.chat_template = parse_chat_template_kind(&args, idx, "--chat-template")?;
            }
            "--lowercase" => data.lowercase = true,
            "--out" => {
                idx += 1;
                tokenizer.output_path = PathBuf::from(take_value(&args, idx, "--out")?);
            }
            "--model" => {
                idx += 1;
                tokenizer.model = parse_tokenizer_model_kind(&args, idx, "--model")?;
            }
            "--vocab-size" => {
                idx += 1;
                tokenizer.vocab_size = parse_usize(&args, idx, "--vocab-size")?;
            }
            "--min-frequency" => {
                idx += 1;
                tokenizer.min_frequency = parse_u64(&args, idx, "--min-frequency")?;
            }
            "--show-progress" => tokenizer.show_progress = true,
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown train-tokenizer argument {other:?}\n\n{}",
                    train_tokenizer_help(bin)
                )));
            }
        }
        idx += 1;
    }

    Ok(Command::TrainTokenizer(TrainTokenizerCommand {
        data,
        tokenizer,
    }))
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
            "--top-k" => {
                idx += 1;
                chat.top_k = parse_usize(&args, idx, "--top-k")?;
            }
            "--top-p" => {
                idx += 1;
                chat.top_p = parse_f32(&args, idx, "--top-p")?;
            }
            "--repetition-penalty" => {
                idx += 1;
                chat.repetition_penalty = parse_f32(&args, idx, "--repetition-penalty")?;
            }
            "--presence-penalty" => {
                idx += 1;
                chat.presence_penalty = parse_f32(&args, idx, "--presence-penalty")?;
            }
            "--frequency-penalty" => {
                idx += 1;
                chat.frequency_penalty = parse_f32(&args, idx, "--frequency-penalty")?;
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
            "--stream" => chat.stream = true,
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

fn parse_eval(bin: &str, args: Vec<String>) -> Result<Command> {
    let mut checkpoint = None;
    let mut data = None::<DataConfig>;
    let mut eval = EvalConfig::default();

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--help" | "-h" => return Ok(Command::Help(eval_help(bin))),
            "--checkpoint" => {
                idx += 1;
                checkpoint = Some(PathBuf::from(take_value(&args, idx, "--checkpoint")?));
            }
            "--data" => {
                idx += 1;
                data.get_or_insert_with(DataConfig::default).data_path =
                    take_value(&args, idx, "--data")?.into();
            }
            "--format" => {
                idx += 1;
                let value = take_value(&args, idx, "--format")?;
                data.get_or_insert_with(DataConfig::default).format =
                    parse_data_format(&value, "--format")?;
            }
            "--lowercase" => data.get_or_insert_with(DataConfig::default).lowercase = true,
            "--chat-template" => {
                idx += 1;
                data.get_or_insert_with(DataConfig::default).chat_template =
                    parse_chat_template_kind(&args, idx, "--chat-template")?;
            }
            "--max-examples" => {
                idx += 1;
                eval.max_examples = parse_usize(&args, idx, "--max-examples")?;
            }
            "--prompt" => {
                idx += 1;
                eval.prompts.push(take_value(&args, idx, "--prompt")?);
            }
            "--prompt-file" => {
                idx += 1;
                eval.prompt_files
                    .push(PathBuf::from(take_value(&args, idx, "--prompt-file")?));
            }
            "--temperature" => {
                idx += 1;
                eval.temperature = parse_f32(&args, idx, "--temperature")?;
            }
            "--top-k" => {
                idx += 1;
                eval.top_k = parse_usize(&args, idx, "--top-k")?;
            }
            "--top-p" => {
                idx += 1;
                eval.top_p = parse_f32(&args, idx, "--top-p")?;
            }
            "--repetition-penalty" => {
                idx += 1;
                eval.repetition_penalty = parse_f32(&args, idx, "--repetition-penalty")?;
            }
            "--presence-penalty" => {
                idx += 1;
                eval.presence_penalty = parse_f32(&args, idx, "--presence-penalty")?;
            }
            "--frequency-penalty" => {
                idx += 1;
                eval.frequency_penalty = parse_f32(&args, idx, "--frequency-penalty")?;
            }
            "--max-new-tokens" => {
                idx += 1;
                eval.max_new_tokens = parse_usize(&args, idx, "--max-new-tokens")?;
            }
            "--device" => {
                idx += 1;
                eval.device = parse_device_kind(&args, idx, "--device")?;
            }
            other => {
                return Err(RustGptError::Cli(format!(
                    "unknown eval argument {other:?}\n\n{}",
                    eval_help(bin)
                )));
            }
        }
        idx += 1;
    }

    let checkpoint = checkpoint.ok_or_else(|| {
        RustGptError::Cli(format!(
            "missing required --checkpoint for eval\n\n{}",
            eval_help(bin)
        ))
    })?;

    Ok(Command::Eval(EvalCommand {
        checkpoint,
        data,
        eval,
    }))
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

fn parse_data_format(value: &str, flag: &str) -> Result<DataFormat> {
    DataFormat::parse(value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected lines, text, jsonl-text, jsonl-chat, parquet-text, or parquet-chat."
        ))
    })
}

fn parse_device_kind(args: &[String], idx: usize, flag: &str) -> Result<DeviceKind> {
    let value = take_value(args, idx, flag)?;
    DeviceKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected cpu, auto, or gpu."
        ))
    })
}

fn parse_chat_template_kind(args: &[String], idx: usize, flag: &str) -> Result<ChatTemplateKind> {
    let value = take_value(args, idx, flag)?;
    ChatTemplateKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected simple or chatml."
        ))
    })
}

fn parse_activation_kind(args: &[String], idx: usize, flag: &str) -> Result<ActivationKind> {
    let value = take_value(args, idx, flag)?;
    ActivationKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected relu, gelu, or swiglu."
        ))
    })
}

fn parse_position_encoding_kind(
    args: &[String],
    idx: usize,
    flag: &str,
) -> Result<PositionEncodingKind> {
    let value = take_value(args, idx, flag)?;
    PositionEncodingKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected learned or rope."
        ))
    })
}

fn parse_lr_schedule_kind(args: &[String], idx: usize, flag: &str) -> Result<LrScheduleKind> {
    let value = take_value(args, idx, flag)?;
    LrScheduleKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected linear or cosine."
        ))
    })
}

fn parse_tokenizer_model_kind(
    args: &[String],
    idx: usize,
    flag: &str,
) -> Result<TokenizerModelKind> {
    let value = take_value(args, idx, flag)?;
    TokenizerModelKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected bpe."
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
