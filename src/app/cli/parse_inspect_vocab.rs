use crate::app::cli::help::inspect_vocab_help;
use crate::app::cli::{Command, InspectVocabCommand};
use crate::core::config::{BoundaryMode, DataConfig, InspectVocabConfig};
use crate::core::error::{Result, RustGptError};

use super::parse_shared::{parse_chat_template_kind, parse_data_format, parse_usize, take_value};

pub(super) fn parse_inspect_vocab(bin: &str, args: Vec<String>) -> Result<Command> {
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
                data.format = parse_data_format(&value, "--format")?;
            }
            "--tokenizer" => {
                idx += 1;
                data.tokenizer_path = Some(take_value(&args, idx, "--tokenizer")?.into());
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
