use std::collections::BTreeSet;

use crate::app::cli::{InspectVocabCommand, TrainCommand};
use crate::core::error::Result;
use crate::data::checkpoint::save_checkpoint;
use crate::data::corpus::Dataset;
use crate::data::tokenizer::Tokenizer;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::profile::RuntimeProfile;
use crate::runtime::training::{prepare_training_run, run_training_steps};

pub fn run_train(command: TrainCommand) -> Result<()> {
    let mut prepared =
        prepare_training_run(&command.data, &command.train, command.resume.as_deref())?;
    let mut backend = ComputeBackend::from_model(&prepared.model, command.train.device)?;

    println!(
        "RustGPT  vocab={}  params={}  layers={}  embd={}  heads={}",
        prepared.tokenizer.vocab_size(),
        prepared.model.num_parameters(),
        prepared.model.cfg.n_layer,
        prepared.model.cfg.n_embd,
        prepared.model.cfg.n_head
    );
    println!(
        "data={}  format={}  docs={}  total_bytes={}  start_step={}",
        command.data.data_path.display(),
        command.data.format,
        prepared.dataset.len(),
        prepared.dataset.total_bytes(),
        prepared.starting_step
    );
    println!(
        "steps={}  batch_size={}  lr={}  beta1={}  beta2={}  eps={}  seed={}  boundary={}  backend={}",
        command.train.steps,
        command.train.batch_size,
        command.train.learning_rate,
        command.train.beta1,
        command.train.beta2,
        command.train.eps_adam,
        command.train.seed,
        prepared.tokenizer.boundary_mode(),
        backend.description()
    );

    let runtime_profile = command.train.profile.then(RuntimeProfile::default);
    let summary = run_training_steps(
        &mut prepared.model,
        &prepared.tokenizer,
        &prepared.training_data,
        &mut backend,
        &command.train,
        prepared.starting_step,
        runtime_profile.as_ref(),
        true,
    )?;

    for entry in &summary.log_entries {
        println!(
            "step {:>5} / {:>5}  loss={:.4}  batch={}  t={:.2?}  doc={:?}",
            entry.step,
            command.train.steps,
            entry.loss,
            entry.batch_size,
            entry.elapsed,
            entry.first_doc,
        );
    }

    let elapsed = summary
        .log_entries
        .last()
        .map(|entry| entry.elapsed)
        .unwrap_or_default();
    println!(
        "done  final_loss={:.4}  elapsed={:.2?}",
        summary.final_loss, elapsed
    );
    if let Some(runtime_profile) = runtime_profile.as_ref() {
        if !runtime_profile.is_empty() {
            println!("stage timings:");
            println!("{}", runtime_profile.format_table());
        }
    }

    if let Some(checkpoint_out) = &command.checkpoint_out {
        backend.download_model(&mut prepared.model)?;
        save_checkpoint(
            checkpoint_out,
            &prepared.model,
            &prepared.tokenizer,
            prepared.starting_step + command.train.steps,
            prepared.checkpoint_seed,
        )?;
        println!("saved checkpoint={}", checkpoint_out.display());
    }

    Ok(())
}

pub fn run_inspect_vocab(command: InspectVocabCommand) -> Result<()> {
    let dataset = Dataset::from_path(&command.data)?;
    let tokenizer = Tokenizer::from_docs(dataset.docs(), command.inspect.boundary_mode)?;

    println!("RustGPT vocabulary");
    println!("data path        {}", command.data.data_path.display());
    println!("data format      {}", command.data.format);
    println!("docs             {}", dataset.len());
    println!("boundary mode    {}", tokenizer.boundary_mode());
    println!("vocab size       {}", tokenizer.vocab_size());
    println!("bos id           {}", tokenizer.bos_id());
    match tokenizer.eos_id() {
        Some(eos_id) => println!("eos id           {}", eos_id),
        None => println!("eos id           <shared with BOS>"),
    }
    println!();

    let mut shown = 0;
    let used_token_ids = dataset
        .docs()
        .iter()
        .flat_map(|doc| tokenizer.encode_text(doc))
        .collect::<BTreeSet<_>>();
    for token_id in used_token_ids
        .iter()
        .copied()
        .take(command.inspect.show_tokens)
    {
        println!("{token_id:>4}  {}", tokenizer.symbol(token_id)?);
        shown += 1;
    }

    if shown < command.inspect.show_tokens {
        for token_id in [
            tokenizer.bos_id(),
            tokenizer.eos_id().unwrap_or(tokenizer.bos_id()),
        ] {
            if shown >= command.inspect.show_tokens {
                break;
            }
            println!("{token_id:>4}  {}", tokenizer.symbol(token_id)?);
            shown += 1;
        }
    }

    if let Some(first_doc) = dataset.docs().first() {
        let encoded = tokenizer.encode_with_boundaries(first_doc)?;
        println!();
        println!("first doc        {:?}", truncate_string(first_doc, 48));
        println!("encoded          {:?}", encoded);
        println!("decoded          {:?}", tokenizer.decode(&encoded, false)?);
    }

    Ok(())
}

fn truncate_string(input: &str, max_chars: usize) -> String {
    let mut out = input.chars().take(max_chars).collect::<String>();
    if input.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}
