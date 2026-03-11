use std::path::PathBuf;

use crate::app::cli::help::prepare_data_help;
use crate::app::cli::{Command, PrepareDataCommand};
use crate::core::config::{DataConfig, PrepareDataConfig};
use crate::core::error::{Result, RustGptError};

use super::parse_shared::{parse_chat_template_kind, parse_data_format, parse_usize, take_value};

pub(super) fn parse_prepare_data(bin: &str, args: Vec<String>) -> Result<Command> {
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
            "--dedup" => prepare.dedup = true,
            "--min-chars" => {
                idx += 1;
                prepare.min_chars = parse_usize(&args, idx, "--min-chars")?;
            }
            "--max-chars" => {
                idx += 1;
                prepare.max_chars = parse_usize(&args, idx, "--max-chars")?;
            }
            "--min-messages" => {
                idx += 1;
                prepare.min_messages = parse_usize(&args, idx, "--min-messages")?;
            }
            "--require-assistant" => prepare.require_assistant = true,
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
