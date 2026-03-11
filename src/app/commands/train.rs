use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::app::cli::{InspectVocabCommand, TrainCommand};
use crate::core::error::Result;
use crate::data::checkpoint::save_checkpoint;
use crate::data::corpus::Dataset;
use crate::data::tokenizer::Tokenizer;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::profile::RuntimeProfile;
use crate::runtime::training::{prepare_training_run, run_training_steps};

pub fn run_train(command: TrainCommand) -> Result<()> {
    let mut prepared = prepare_training_run(
        &command.data,
        &command.train,
        command.validation_data.as_ref(),
        command.resume.as_deref(),
    )?;
    let mut backend = ComputeBackend::from_model(&prepared.model, command.train.device)?;
    let best_checkpoint_path = command.best_checkpoint_out.clone().or_else(|| {
        prepared.validation_data.as_ref().and_then(|_| {
            command
                .checkpoint_out
                .as_ref()
                .map(|path| derive_best_checkpoint_path(path))
        })
    });

    println!(
        "RustGPT  vocab={}  params={}  layers={}  embd={}  heads={}  kv_heads={}  tied_embeddings={}  activation={}  position={}  tokenizer={}",
        prepared.tokenizer.vocab_size(),
        prepared.model.num_parameters(),
        prepared.model.cfg.n_layer,
        prepared.model.cfg.n_embd,
        prepared.model.cfg.n_head,
        prepared.model.cfg.n_kv_head,
        prepared.model.uses_tied_embeddings(),
        prepared.model.cfg.activation,
        prepared.model.cfg.position_encoding,
        prepared.tokenizer.kind_name()
    );
    println!(
        "data={}  format={}  docs={}  total_bytes={}  chat_template={}  start_step={}",
        command.data.data_path.display(),
        command.data.format,
        prepared.dataset.len(),
        prepared
            .dataset
            .total_bytes_with_template(prepared.chat_template),
        prepared.chat_template,
        prepared.starting_step
    );
    let train_windows = prepared
        .training_data
        .window_summary(prepared.model.cfg.block_size);
    println!(
        "train_windows={}  source_sequences={}  multi_window_sequences={}  max_windows_per_sequence={}",
        train_windows.total_windows,
        train_windows.source_sequences,
        train_windows.multi_window_sequences,
        train_windows.max_windows_per_sequence
    );
    println!(
        "steps={}  batch_size={}  lr={}  beta1={}  beta2={}  eps={}  weight_decay={}  warmup={}  schedule={}  grad_clip={}  seed={}  mode={}  boundary={}  backend={}",
        command.train.steps,
        command.train.batch_size,
        command.train.learning_rate,
        command.train.beta1,
        command.train.beta2,
        command.train.eps_adam,
        command.train.weight_decay,
        command.train.warmup_steps,
        command.train.lr_schedule,
        command.train.grad_clip,
        command.train.seed,
        command.train.mode,
        prepared.tokenizer.boundary_mode(),
        backend.description()
    );
    if let Some(validation_data) = prepared.validation_data.as_ref() {
        let validation_windows = validation_data.window_summary(prepared.model.cfg.block_size);
        println!(
            "validation_examples={}  validation_source_sequences={}  validation_multi_window_sequences={}  validation_max_windows_per_sequence={}  validation_max_examples={}  best_checkpoint={}",
            validation_windows.total_windows,
            validation_windows.source_sequences,
            validation_windows.multi_window_sequences,
            validation_windows.max_windows_per_sequence,
            command.train.validation_max_examples,
            best_checkpoint_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<disabled>".to_string())
        );
    }

    let runtime_profile = command.train.profile.then(RuntimeProfile::default);
    let tokenizer_for_best = prepared.tokenizer.clone();
    let chat_template = prepared.chat_template;
    let checkpoint_seed = prepared.checkpoint_seed;
    let mut best_validation_loss = None::<f32>;
    let mut save_best = |trained_steps: usize,
                         validation_loss: f32,
                         model: &mut crate::model::Model,
                         backend: &mut ComputeBackend|
     -> Result<()> {
        if best_validation_loss
            .map(|best| validation_loss < best)
            .unwrap_or(true)
        {
            best_validation_loss = Some(validation_loss);
            if let Some(path) = best_checkpoint_path.as_ref() {
                backend.download_model(model)?;
                save_checkpoint(
                    path,
                    model,
                    &tokenizer_for_best,
                    chat_template,
                    trained_steps,
                    checkpoint_seed,
                )?;
            }
        }
        Ok(())
    };
    let summary = run_training_steps(
        &mut prepared.model,
        &prepared.tokenizer,
        &prepared.training_data,
        &mut backend,
        &command.train,
        prepared.starting_step,
        runtime_profile.as_ref(),
        true,
        prepared.validation_data.as_ref(),
        Some(&mut save_best),
    )?;

    for entry in &summary.log_entries {
        if let Some(validation_loss) = entry.validation_loss {
            println!(
                "step {:>5} / {:>5}  loss={:.4}  val={:.4}  best_val={:.4}  lr={:.5}  batch={}  t={:.2?}  doc={:?}",
                entry.step,
                command.train.steps,
                entry.loss,
                validation_loss,
                entry.best_validation_loss.unwrap_or(validation_loss),
                entry.learning_rate,
                entry.batch_size,
                entry.elapsed,
                entry.first_doc,
            );
        } else {
            println!(
                "step {:>5} / {:>5}  loss={:.4}  lr={:.5}  batch={}  t={:.2?}  doc={:?}",
                entry.step,
                command.train.steps,
                entry.loss,
                entry.learning_rate,
                entry.batch_size,
                entry.elapsed,
                entry.first_doc,
            );
        }
    }

    let elapsed = summary
        .log_entries
        .last()
        .map(|entry| entry.elapsed)
        .unwrap_or_default();
    if let Some(validation_loss) = summary.final_validation_loss {
        println!(
            "done  final_loss={:.4}  final_val={:.4}  best_val={:.4}  elapsed={:.2?}",
            summary.final_loss,
            validation_loss,
            summary.best_validation_loss.unwrap_or(validation_loss),
            elapsed
        );
    } else {
        println!(
            "done  final_loss={:.4}  elapsed={:.2?}",
            summary.final_loss, elapsed
        );
    }
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
            prepared.chat_template,
            prepared.starting_step + command.train.steps,
            prepared.checkpoint_seed,
        )?;
        println!("saved checkpoint={}", checkpoint_out.display());
    }

    Ok(())
}

fn derive_best_checkpoint_path(path: &Path) -> PathBuf {
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("checkpoint");
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("ckpt");
    parent.join(format!("{stem}.best.{extension}"))
}

pub fn run_inspect_vocab(command: InspectVocabCommand) -> Result<()> {
    let dataset = Dataset::from_path(&command.data)?;
    let tokenizer = if let Some(path) = &command.data.tokenizer_path {
        Tokenizer::from_tokenizer_file(
            path,
            command.data.tokenizer_bos.as_deref(),
            command.data.tokenizer_eos.as_deref(),
        )?
    } else {
        let docs = dataset.docs_with_template(command.data.chat_template);
        Tokenizer::from_docs(&docs, command.inspect.boundary_mode)?
    };

    println!("RustGPT vocabulary");
    println!("data path        {}", command.data.data_path.display());
    println!("data format      {}", command.data.format);
    println!("docs             {}", dataset.len());
    println!("chat template    {}", command.data.chat_template);
    println!("tokenizer kind   {}", tokenizer.kind_name());
    println!("boundary mode    {}", tokenizer.boundary_mode());
    println!("vocab size       {}", tokenizer.vocab_size());
    println!("bos id           {}", tokenizer.bos_id());
    match tokenizer.eos_id() {
        Some(eos_id) => println!("eos id           {}", eos_id),
        None => println!("eos id           <shared with BOS>"),
    }
    println!();

    let mut shown = 0;
    let rendered_docs = dataset.docs_with_template(command.data.chat_template);
    let used_token_ids = rendered_docs
        .iter()
        .flat_map(|doc| tokenizer.encode_text(doc))
        .collect::<BTreeSet<_>>();
    for token_id in used_token_ids
        .iter()
        .copied()
        .take(command.inspect.show_tokens)
    {
        println!("{token_id:>4}  {}", tokenizer.token_label(token_id)?);
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
            println!("{token_id:>4}  {}", tokenizer.token_label(token_id)?);
            shown += 1;
        }
    }

    if let Some(first_doc) = rendered_docs.first() {
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
