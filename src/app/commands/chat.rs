use std::io::{self, Write};

use crate::app::cli::ChatCommand;
use crate::core::error::Result;
use crate::core::rng::Rng;
use crate::data::checkpoint::load_checkpoint;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::chat::{ChatDirective, ChatSession, parse_chat_directive};
use crate::runtime::sampling::generate_completion_with_backend;

pub fn run_chat(command: ChatCommand) -> Result<()> {
    let checkpoint = load_checkpoint(&command.checkpoint)?;
    let backend = ComputeBackend::from_model(&checkpoint.model, command.chat.device)?;
    let mut rng = Rng::from_seed(command.chat.seed);
    let mut session = ChatSession::new(command.chat.system_prompt.clone());

    println!(
        "RustGPT chat  checkpoint={}  trained_steps={}  context={}  max_new_tokens={}  backend={}",
        command.checkpoint.display(),
        checkpoint.trained_steps,
        checkpoint.model.cfg.block_size,
        command.chat.max_new_tokens,
        backend.description()
    );
    println!("Commands: /exit  /reset  /history");
    println!("Submit a message with an empty line. Multi-line UTF-8 input is supported.");

    let stdin = io::stdin();
    loop {
        let Some(input) = read_user_message(&stdin)? else {
            println!();
            break;
        };

        if let Some(directive) = parse_chat_directive(&input) {
            match directive {
                ChatDirective::Exit => break,
                ChatDirective::Reset => {
                    session.reset();
                    println!("history cleared");
                }
                ChatDirective::History => {
                    println!("{}", session.history());
                }
            }
            continue;
        }

        session.push_user_turn(&input);
        let reply = generate_completion_with_backend(
            &checkpoint.model,
            &backend,
            &checkpoint.tokenizer,
            &session.prompt(),
            command.chat.max_new_tokens,
            command.chat.temperature,
            session.stop_sequences(),
            None,
            &mut rng,
        )?;
        println!("assistant> {}", reply);
        session.push_assistant_turn(&reply);
    }

    Ok(())
}

fn read_user_message(stdin: &io::Stdin) -> Result<Option<String>> {
    let mut lines = Vec::new();

    loop {
        if lines.is_empty() {
            print!("user> ");
        } else {
            print!("...> ");
        }
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes_read = stdin.read_line(&mut line)?;
        if bytes_read == 0 {
            return if lines.is_empty() {
                Ok(None)
            } else {
                Ok(Some(lines.join("\n")))
            };
        }

        let line = line.trim_end_matches(['\n', '\r']);
        if lines.is_empty() {
            if line.trim().is_empty() {
                continue;
            }
            if parse_chat_directive(line).is_some() {
                return Ok(Some(line.trim().to_string()));
            }
        }

        if line.is_empty() {
            break;
        }
        lines.push(line.to_string());
    }

    Ok(Some(lines.join("\n")))
}

#[cfg(test)]
mod tests {
    use crate::runtime::chat::{ChatDirective, parse_chat_directive};

    #[test]
    fn directives_are_detected_before_multiline_collection() {
        assert_eq!(parse_chat_directive("/reset"), Some(ChatDirective::Reset));
    }
}
