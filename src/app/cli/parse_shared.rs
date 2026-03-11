use crate::app::cli::help::global_help;
use crate::core::config::{
    ActivationKind, BenchmarkConfig, ChatTemplateKind, DataFormat, DeviceKind, LrScheduleKind,
    PositionEncodingKind, TokenizerModelKind, TrainConfig,
};
use crate::core::error::{Result, RustGptError};
use crate::core::presets::TrainPreset;

pub(super) fn unknown_command(command: &str, bin: &str) -> Result<crate::app::cli::Command> {
    Err(RustGptError::Cli(format!(
        "unknown command {command:?}\n\n{}",
        global_help(bin)
    )))
}

pub(super) fn take_value(args: &[String], idx: usize, flag: &str) -> Result<String> {
    args.get(idx).cloned().ok_or_else(|| {
        RustGptError::Cli(format!(
            "missing value for {flag}. Run `rustgpt help` for usage."
        ))
    })
}

pub(super) fn parse_usize(args: &[String], idx: usize, flag: &str) -> Result<usize> {
    let value = take_value(args, idx, flag)?;
    value
        .parse::<usize>()
        .map_err(|_| RustGptError::Cli(format!("invalid integer for {flag}: {value:?}")))
}

pub(super) fn parse_u64(args: &[String], idx: usize, flag: &str) -> Result<u64> {
    let value = take_value(args, idx, flag)?;
    value
        .parse::<u64>()
        .map_err(|_| RustGptError::Cli(format!("invalid integer for {flag}: {value:?}")))
}

pub(super) fn parse_f32(args: &[String], idx: usize, flag: &str) -> Result<f32> {
    let value = take_value(args, idx, flag)?;
    value
        .parse::<f32>()
        .map_err(|_| RustGptError::Cli(format!("invalid float for {flag}: {value:?}")))
}

pub(super) fn parse_data_format(value: &str, flag: &str) -> Result<DataFormat> {
    DataFormat::parse(value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected lines, text, jsonl-text, jsonl-chat, parquet-text, or parquet-chat."
        ))
    })
}

pub(super) fn parse_device_kind(args: &[String], idx: usize, flag: &str) -> Result<DeviceKind> {
    let value = take_value(args, idx, flag)?;
    DeviceKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected cpu, auto, or gpu."
        ))
    })
}

pub(super) fn parse_chat_template_kind(
    args: &[String],
    idx: usize,
    flag: &str,
) -> Result<ChatTemplateKind> {
    let value = take_value(args, idx, flag)?;
    ChatTemplateKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected simple or chatml."
        ))
    })
}

pub(super) fn parse_activation_kind(
    args: &[String],
    idx: usize,
    flag: &str,
) -> Result<ActivationKind> {
    let value = take_value(args, idx, flag)?;
    ActivationKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected relu, gelu, or swiglu."
        ))
    })
}

pub(super) fn parse_position_encoding_kind(
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

pub(super) fn parse_lr_schedule_kind(
    args: &[String],
    idx: usize,
    flag: &str,
) -> Result<LrScheduleKind> {
    let value = take_value(args, idx, flag)?;
    LrScheduleKind::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected linear or cosine."
        ))
    })
}

pub(super) fn parse_tokenizer_model_kind(
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

pub(super) fn parse_train_preset(args: &[String], idx: usize, flag: &str) -> Result<TrainPreset> {
    let value = take_value(args, idx, flag)?;
    TrainPreset::parse(&value).ok_or_else(|| {
        RustGptError::Cli(format!(
            "invalid value for {flag}: {value:?}. Expected {}.",
            TrainPreset::names()
        ))
    })
}

pub(super) fn validate_train_surface(train: &TrainConfig) -> Result<()> {
    if train.sample_every == 0 {
        return Err(RustGptError::Cli(
            "invalid value for --sample-every: expected an integer >= 1".to_string(),
        ));
    }
    if train.gradient_accumulation_steps == 0 {
        return Err(RustGptError::Cli(
            "invalid value for --grad-accum-steps: expected an integer >= 1".to_string(),
        ));
    }
    if train.validation_max_examples == 0 {
        return Err(RustGptError::Cli(
            "invalid value for --valid-max-examples: expected an integer >= 1".to_string(),
        ));
    }
    if train.n_kv_head != 0 && train.n_kv_head > train.n_head {
        return Err(RustGptError::Cli(format!(
            "--n-kv-head must be less than or equal to --n-head (got {} vs {})",
            train.n_kv_head, train.n_head
        )));
    }
    if train.n_kv_head != 0 && !train.n_head.is_multiple_of(train.n_kv_head) {
        return Err(RustGptError::Cli(format!(
            "--n-head must be divisible by --n-kv-head for grouped-query attention (got {} vs {})",
            train.n_head, train.n_kv_head
        )));
    }

    Ok(())
}

pub(super) fn strip_benchmark_flags(args: &[String]) -> Result<(Vec<String>, BenchmarkConfig)> {
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
