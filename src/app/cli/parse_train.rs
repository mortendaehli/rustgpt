use std::path::PathBuf;

use crate::app::cli::help::{bench_compare_train_help, bench_train_help, train_help};
use crate::app::cli::{BenchCompareTrainCommand, BenchTrainCommand, Command, TrainCommand};
use crate::core::config::{
    BoundaryMode, ChatTemplateKind, DataConfig, DataFormat, TrainConfig, TrainMode,
};
use crate::core::error::{Result, RustGptError};

use super::parse_shared::{
    parse_activation_kind, parse_chat_template_kind, parse_data_format, parse_device_kind,
    parse_f32, parse_lr_schedule_kind, parse_position_encoding_kind, parse_train_preset, parse_u64,
    parse_usize, strip_benchmark_flags, take_value, validate_train_surface,
};

pub(super) fn parse_train(bin: &str, args: Vec<String>) -> Result<Command> {
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
            "--preset" => {
                idx += 1;
                train = parse_train_preset(&args, idx, "--preset")?.train_config();
            }
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
                data.format = parse_data_format(&value, "--format")?;
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
                validation_data
                    .get_or_insert_with(DataConfig::default)
                    .format = parse_data_format(&value, "--valid-format")?;
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
            "--grad-accum" | "--grad-accum-steps" => {
                idx += 1;
                train.gradient_accumulation_steps = parse_usize(&args, idx, "--grad-accum-steps")?;
            }
            "--activation-checkpointing" | "--checkpoint-activations" => {
                train.activation_checkpointing = true;
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

    validate_train_surface(&train)?;

    Ok(Command::Train(TrainCommand {
        data,
        train,
        validation_data,
        checkpoint_out,
        best_checkpoint_out,
        resume,
    }))
}

pub(super) fn parse_bench_train(bin: &str, args: Vec<String>) -> Result<Command> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Ok(Command::Help(bench_train_help(bin)));
    }
    let (forwarded, bench) = strip_benchmark_flags(&args)?;
    let Command::Train(train) = parse_train(bin, forwarded)? else {
        unreachable!("parse_train always returns Command::Train");
    };
    Ok(Command::BenchTrain(BenchTrainCommand { train, bench }))
}

pub(super) fn parse_bench_compare_train(bin: &str, args: Vec<String>) -> Result<Command> {
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
