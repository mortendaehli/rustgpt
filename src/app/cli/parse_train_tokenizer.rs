use std::path::PathBuf;

use crate::app::cli::help::train_tokenizer_help;
use crate::app::cli::{Command, TrainTokenizerCommand};
use crate::core::config::{DataConfig, TrainTokenizerConfig};
use crate::core::error::{Result, RustGptError};

use super::parse_shared::{
    parse_chat_template_kind, parse_data_format, parse_tokenizer_model_kind, parse_u64,
    parse_usize, take_value,
};

pub(super) fn parse_train_tokenizer(bin: &str, args: Vec<String>) -> Result<Command> {
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
