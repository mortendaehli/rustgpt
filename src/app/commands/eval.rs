use burn::tensor::backend::Backend;

use crate::app::cli::EvalCommand;
use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::data::corpus::Dataset;
use crate::data::eval_suite::{PromptEvalCase, load_prompt_eval_suite};
use crate::data::training_data::TrainingData;
use crate::engine::checkpoint::load_inference_checkpoint;
use crate::engine::device::{CpuBackend, GpuBackend, ResolvedDeviceKind, cpu_device, gpu_device};
use crate::engine::generate::{SamplingStrategy, StopCondition, generate_completion};
use crate::engine::train::evaluate_training_data;

pub fn run_eval(command: EvalCommand) -> Result<()> {
    let resolved = ResolvedDeviceKind::resolve(command.eval.device)?;
    match resolved {
        ResolvedDeviceKind::Cpu => run_eval_impl::<CpuBackend>(command, cpu_device(), resolved),
        ResolvedDeviceKind::Gpu => run_eval_impl::<GpuBackend>(command, gpu_device(), resolved),
    }
}

fn run_eval_impl<B: Backend>(
    command: EvalCommand,
    device: B::Device,
    resolved: ResolvedDeviceKind,
) -> Result<()> {
    let checkpoint = load_inference_checkpoint::<B>(&command.checkpoint, &device)?;
    let strategy = SamplingStrategy {
        temperature: command.eval.temperature,
        top_k: command.eval.top_k,
        top_p: command.eval.top_p,
        repetition_penalty: command.eval.repetition_penalty,
        presence_penalty: command.eval.presence_penalty,
        frequency_penalty: command.eval.frequency_penalty,
    };

    println!(
        "RustGPT eval  checkpoint={}  trained_steps={}  backend={}",
        command.checkpoint.display(),
        checkpoint.trained_steps,
        resolved.description()
    );

    if let Some(data_config) = &command.data {
        let dataset = Dataset::from_path(data_config)?;
        let training_data = TrainingData::from_dataset(
            &dataset,
            &checkpoint.tokenizer,
            checkpoint_train_mode(data_config.format),
            data_config.chat_template,
        )?;
        let metrics = evaluate_training_data(
            &checkpoint.model,
            &checkpoint.tokenizer,
            &training_data,
            checkpoint.model.config().block_size,
            command.eval.max_examples,
            &device,
        )?;
        println!(
            "eval_data={}  format={}  examples={}  mean_loss={:.4}  perplexity={:.4}",
            data_config.data_path.display(),
            data_config.format,
            metrics.examples,
            metrics.mean_loss,
            metrics.perplexity
        );
    }

    if !command.eval.prompts.is_empty() {
        println!("inline_prompts={}", command.eval.prompts.len());
    }

    let mut prompt_cases = command
        .eval
        .prompts
        .iter()
        .enumerate()
        .map(|(idx, prompt)| PromptEvalCase {
            name: format!("inline-prompt-{}", idx + 1),
            prompt: prompt.clone(),
            notes: None,
            must_contain: Vec::new(),
            must_not_contain: Vec::new(),
            max_new_tokens: None,
        })
        .collect::<Vec<_>>();

    for path in &command.eval.prompt_files {
        let suite = load_prompt_eval_suite(path)?;
        println!("prompt_suite={}  cases={}", path.display(), suite.len());
        prompt_cases.extend(suite);
    }

    if !prompt_cases.is_empty() {
        let mut rng = Rng::from_seed(42);
        let mut failed_cases = Vec::new();
        for (prompt_idx, case) in prompt_cases.iter().enumerate() {
            let completion = generate_completion(
                &checkpoint.model,
                &checkpoint.tokenizer,
                &case.prompt,
                case.max_new_tokens.unwrap_or(command.eval.max_new_tokens),
                &strategy,
                &StopCondition::none(),
                None,
                &mut rng,
            )?;
            let failures = check_prompt_case(&completion, case);
            let status = if failures.is_empty() { "pass" } else { "fail" };
            println!(
                "case   {:>2}: {}  status={}",
                prompt_idx + 1,
                case.name,
                status
            );
            if let Some(notes) = &case.notes {
                println!("notes  {:>2}: {}", prompt_idx + 1, notes);
            }
            println!("prompt {:>2}: {:?}", prompt_idx + 1, case.prompt);
            println!("output {:>2}: {}", prompt_idx + 1, completion);
            if !failures.is_empty() {
                println!("checks {:>2}: {}", prompt_idx + 1, failures.join("; "));
                failed_cases.push(case.name.clone());
            }
        }
        if !failed_cases.is_empty() {
            return Err(RustGptError::Data(format!(
                "prompt eval failed for {} case(s): {}",
                failed_cases.len(),
                failed_cases.join(", ")
            )));
        }
    }

    if command.data.is_none()
        && command.eval.prompts.is_empty()
        && command.eval.prompt_files.is_empty()
    {
        return Err(RustGptError::Cli(
            "eval needs either --data for held-out loss, --prompt for generation checks, --prompt-file for checked prompt suites, or a combination of them".to_string(),
        ));
    }

    Ok(())
}

fn check_prompt_case(completion: &str, case: &PromptEvalCase) -> Vec<String> {
    let mut failures = Vec::new();
    for needle in &case.must_contain {
        if !completion.contains(needle) {
            failures.push(format!("missing required substring {:?}", needle));
        }
    }
    for needle in &case.must_not_contain {
        if completion.contains(needle) {
            failures.push(format!("found forbidden substring {:?}", needle));
        }
    }
    failures
}

fn checkpoint_train_mode(
    format: crate::core::config::DataFormat,
) -> crate::core::config::TrainMode {
    if format.is_chat() {
        crate::core::config::TrainMode::Sft
    } else {
        crate::core::config::TrainMode::Pretrain
    }
}

#[cfg(test)]
mod tests {
    use crate::data::eval_suite::PromptEvalCase;

    use super::check_prompt_case;

    #[test]
    fn prompt_case_checks_forbidden_and_required_substrings() {
        let case = PromptEvalCase {
            name: "stop-cleanly".to_string(),
            prompt: "User: hi\nAssistant: ".to_string(),
            notes: None,
            must_contain: vec!["hi".to_string()],
            must_not_contain: vec!["\nUser:".to_string()],
            max_new_tokens: None,
        };

        let failures = check_prompt_case("hello\nUser: again", &case);
        assert_eq!(failures.len(), 2);
        assert!(failures[0].contains("missing required substring"));
        assert!(failures[1].contains("found forbidden substring"));
    }
}
