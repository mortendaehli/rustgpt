use std::path::PathBuf;

use crate::app::cli::help::chat_help;
use crate::app::cli::{ChatCommand, Command};
use crate::core::config::ChatConfig;
use crate::core::error::{Result, RustGptError};

use super::parse_shared::{parse_device_kind, parse_f32, parse_u64, parse_usize, take_value};

pub(super) fn parse_chat(bin: &str, args: Vec<String>) -> Result<Command> {
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
