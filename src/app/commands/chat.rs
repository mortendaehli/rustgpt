use std::io::{self, Write};

use crate::app::cli::ChatCommand;
use crate::core::error::Result;
use crate::core::rng::Rng;
use crate::data::checkpoint::load_checkpoint;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::chat::{ChatDirective, ChatSession, parse_chat_directive};
use crate::runtime::sampling::{
    SamplingStrategy, generate_completion_from_tokens_streaming_with_backend,
    generate_completion_from_tokens_with_backend,
};

pub fn run_chat(command: ChatCommand) -> Result<()> {
    let checkpoint = load_checkpoint(&command.checkpoint)?;
    let backend = ComputeBackend::from_model(&checkpoint.model, command.chat.device)?;
    let mut rng = Rng::from_seed(command.chat.seed);
    let mut session =
        ChatSession::new(checkpoint.chat_template, command.chat.system_prompt.clone());
    let strategy = SamplingStrategy {
        temperature: command.chat.temperature,
        top_k: command.chat.top_k,
        top_p: command.chat.top_p,
        repetition_penalty: command.chat.repetition_penalty,
        presence_penalty: command.chat.presence_penalty,
        frequency_penalty: command.chat.frequency_penalty,
    };

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
        let prepared_prompt = session.prepare_prompt(
            &checkpoint.tokenizer,
            checkpoint.model.cfg.block_size,
            command.chat.max_new_tokens,
        )?;
        if prepared_prompt.dropped_turns > 0 {
            println!(
                "note: dropped {} old turn(s) to fit the context window",
                prepared_prompt.dropped_turns
            );
        }
        let reply = if command.chat.stream {
            print!("assistant> ");
            io::stdout().flush()?;
            let mut stream_stdout = |delta: &str| {
                print!("{delta}");
                let _ = io::stdout().flush();
            };
            let reply = generate_completion_from_tokens_streaming_with_backend(
                &checkpoint.model,
                &backend,
                &checkpoint.tokenizer,
                &prepared_prompt.prompt_tokens,
                command.chat.max_new_tokens,
                &strategy,
                &prepared_prompt.stop_condition,
                None,
                &mut rng,
                &mut stream_stdout,
            )?;
            println!();
            reply
        } else {
            let reply = generate_completion_from_tokens_with_backend(
                &checkpoint.model,
                &backend,
                &checkpoint.tokenizer,
                &prepared_prompt.prompt_tokens,
                command.chat.max_new_tokens,
                &strategy,
                &prepared_prompt.stop_condition,
                None,
                &mut rng,
            )?;
            println!("assistant> {}", reply);
            reply
        };
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
