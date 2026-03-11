use std::path::PathBuf;

use crate::app::cli::help::eval_help;
use crate::app::cli::{Command, EvalCommand};
use crate::core::config::{DataConfig, EvalConfig};
use crate::core::error::{Result, RustGptError};

use super::parse_shared::{
    parse_chat_template_kind, parse_data_format, parse_device_kind, parse_f32, parse_usize,
    take_value,
};

pub(super) fn parse_eval(bin: &str, args: Vec<String>) -> Result<Command> {
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
