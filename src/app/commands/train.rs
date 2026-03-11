use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::app::cli::{InspectVocabCommand, TrainCommand};
use crate::core::error::Result;
use crate::data::corpus::Dataset;
use crate::data::tokenizer::Tokenizer;
use crate::model::lm::{LanguageModel, LanguageModelOptimizer};
use crate::runtime::checkpoint::{
    CheckpointRunManifest, SaveTrainingCheckpointRequest, load_training_checkpoint,
    save_training_checkpoint,
};
use crate::runtime::device::{
    CpuAutodiffBackend, CpuBackend, CpuCheckpointAutodiffBackend, GpuAutodiffBackend, GpuBackend,
    GpuCheckpointAutodiffBackend, ResolvedDeviceKind, cpu_device, gpu_device,
};
use crate::runtime::profile::RuntimeProfile;
use crate::train::training::{
    TrainingArtifacts, TrainingRunRequest, prepare_training_run, run_training_steps,
};

pub fn run_train(command: TrainCommand) -> Result<()> {
    let resolved = ResolvedDeviceKind::resolve(command.train.device)?;
    match (resolved, command.train.activation_checkpointing) {
        (ResolvedDeviceKind::Cpu, false) => {
            run_train_impl::<CpuBackend, CpuAutodiffBackend>(command, cpu_device(), resolved)
        }
        (ResolvedDeviceKind::Cpu, true) => {
            run_train_impl::<CpuBackend, CpuCheckpointAutodiffBackend>(
                command,
                cpu_device(),
                resolved,
            )
        }
        (ResolvedDeviceKind::Gpu, false) => {
            run_train_impl::<GpuBackend, GpuAutodiffBackend>(command, gpu_device(), resolved)
        }
        (ResolvedDeviceKind::Gpu, true) => {
            run_train_impl::<GpuBackend, GpuCheckpointAutodiffBackend>(
                command,
                gpu_device(),
                resolved,
            )
        }
    }
}

fn run_train_impl<B, AD>(
    command: TrainCommand,
    device: AD::Device,
    resolved: ResolvedDeviceKind,
) -> Result<()>
where
    B: Backend<Device = AD::Device>,
    AD: AutodiffBackend<InnerBackend = B>,
{
    let prepared = prepare_training_run(
        &command.data,
        &command.train,
        command.validation_data.as_ref(),
        command.resume.as_deref(),
    )?;
    let artifacts = if let Some(resume_path) = command.resume.as_deref() {
        let loaded = load_training_checkpoint::<AD>(resume_path, &device, &command.train)?;
        TrainingArtifacts {
            model: loaded.model,
            optimizer: loaded.optimizer,
        }
    } else {
        TrainingArtifacts::new(&prepared.model_config, &command.train, &device)?
    };
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
        artifacts.model.num_parameters(),
        prepared.model_config.n_layer,
        prepared.model_config.n_embd,
        prepared.model_config.n_head,
        prepared.model_config.n_kv_head,
        prepared.model_config.tie_embeddings,
        prepared.model_config.activation,
        prepared.model_config.position_encoding,
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
    if let Some(run_manifest) = prepared.resume_manifest.as_ref() {
        println!(
            "resume_manifest_data={}  resume_manifest_format={}  resume_manifest_device={}",
            run_manifest.training_data_path.display(),
            run_manifest.training_data_format,
            run_manifest.resolved_device
        );
    }
    let train_windows = prepared
        .training_data
        .window_summary(prepared.model_config.block_size);
    println!(
        "train_windows={}  source_sequences={}  multi_window_sequences={}  max_windows_per_sequence={}",
        train_windows.total_windows,
        train_windows.source_sequences,
        train_windows.multi_window_sequences,
        train_windows.max_windows_per_sequence
    );
    println!(
        "steps={}  batch_size={}  grad_accum={}  effective_batch={}  act_ckpt={}  lr={}  beta1={}  beta2={}  eps={}  weight_decay={}  warmup={}  schedule={}  grad_clip={}  seed={}  mode={}  boundary={}  backend={}",
        command.train.steps,
        command.train.batch_size,
        command.train.gradient_accumulation_steps,
        command.train.batch_size * command.train.gradient_accumulation_steps,
        command.train.activation_checkpointing,
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
        resolved.description()
    );
    if let Some(validation_data) = prepared.validation_data.as_ref() {
        let validation_windows = validation_data.window_summary(prepared.model_config.block_size);
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
    let mut save_best = |trained_steps: usize,
                         _validation_loss: f32,
                         model: &LanguageModel<AD>,
                         optimizer: &LanguageModelOptimizer<AD>|
     -> Result<()> {
        if let Some(path) = best_checkpoint_path.as_ref() {
            save_training_checkpoint(SaveTrainingCheckpointRequest {
                path,
                model,
                optimizer: Some(optimizer),
                tokenizer: &tokenizer_for_best,
                chat_template,
                trained_steps,
                seed: checkpoint_seed,
                run_manifest: Some(build_checkpoint_run_manifest(
                    &command,
                    resolved,
                    None,
                    Some(_validation_loss),
                    Some(_validation_loss),
                    None,
                    None,
                )),
            })?;
        }
        Ok(())
    };
    let (artifacts, summary) = run_training_steps::<B, AD>(
        artifacts,
        TrainingRunRequest {
            tokenizer: &prepared.tokenizer,
            training_data: &prepared.training_data,
            train_config: &command.train,
            starting_step: prepared.starting_step,
            device: &device,
            profile: runtime_profile.as_ref(),
            capture_logs: true,
            validation_data: prepared.validation_data.as_ref(),
            on_validation: Some(&mut save_best),
        },
    )?;

    for entry in &summary.log_entries {
        if let Some(validation_loss) = entry.validation_loss {
            println!(
                "step {:>5} / {:>5}  total_step={:>5}  loss={:.4}  val={:.4}  best_val={:.4}  lr={:.5}  batch={}  toks={}  tok/s={:.0}  dt={:.2?}  t={:.2?}  doc={:?}",
                entry.run_step,
                command.train.steps,
                entry.total_step,
                entry.loss,
                validation_loss,
                entry.best_validation_loss.unwrap_or(validation_loss),
                entry.learning_rate,
                entry.batch_size,
                entry.processed_tokens,
                entry.tokens_per_second,
                entry.step_elapsed,
                entry.elapsed,
                entry.first_doc,
            );
        } else {
            println!(
                "step {:>5} / {:>5}  total_step={:>5}  loss={:.4}  lr={:.5}  batch={}  toks={}  tok/s={:.0}  dt={:.2?}  t={:.2?}  doc={:?}",
                entry.run_step,
                command.train.steps,
                entry.total_step,
                entry.loss,
                entry.learning_rate,
                entry.batch_size,
                entry.processed_tokens,
                entry.tokens_per_second,
                entry.step_elapsed,
                entry.elapsed,
                entry.first_doc,
            );
        }
    }

    if let Some(validation_loss) = summary.final_validation_loss {
        println!(
            "done  final_loss={:.4}  final_val={:.4}  best_val={:.4}  tokens={}  avg_tok/s={:.0}  elapsed={:.2?}",
            summary.final_loss,
            validation_loss,
            summary.best_validation_loss.unwrap_or(validation_loss),
            summary.total_tokens,
            summary.average_tokens_per_second,
            summary.elapsed
        );
    } else {
        println!(
            "done  final_loss={:.4}  tokens={}  avg_tok/s={:.0}  elapsed={:.2?}",
            summary.final_loss,
            summary.total_tokens,
            summary.average_tokens_per_second,
            summary.elapsed
        );
    }
    if let Some(runtime_profile) = runtime_profile.as_ref()
        && !runtime_profile.is_empty()
    {
        println!("stage timings:");
        println!("{}", runtime_profile.format_table());
    }

    if let Some(checkpoint_out) = &command.checkpoint_out {
        save_training_checkpoint(SaveTrainingCheckpointRequest {
            path: checkpoint_out,
            model: &artifacts.model,
            optimizer: Some(&artifacts.optimizer),
            tokenizer: &prepared.tokenizer,
            chat_template: prepared.chat_template,
            trained_steps: prepared.starting_step + command.train.steps,
            seed: prepared.checkpoint_seed,
            run_manifest: Some(build_checkpoint_run_manifest(
                &command,
                resolved,
                Some(summary.final_loss),
                summary.final_validation_loss,
                summary.best_validation_loss,
                Some(summary.total_tokens),
                Some(summary.average_tokens_per_second),
            )),
        })?;
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

fn build_checkpoint_run_manifest(
    command: &TrainCommand,
    resolved: ResolvedDeviceKind,
    final_loss: Option<f32>,
    validation_loss: Option<f32>,
    best_validation_loss: Option<f32>,
    total_tokens: Option<usize>,
    average_tokens_per_second: Option<f32>,
) -> CheckpointRunManifest {
    let saved_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    CheckpointRunManifest {
        saved_at_ms,
        train_config: command.train.clone(),
        training_data_path: command.data.data_path.clone(),
        training_data_format: command.data.format,
        validation_data_path: command
            .validation_data
            .as_ref()
            .map(|config| config.data_path.clone()),
        validation_data_format: command.validation_data.as_ref().map(|config| config.format),
        resolved_device: resolved.description().to_string(),
        final_loss,
        validation_loss,
        best_validation_loss,
        total_tokens,
        average_tokens_per_second,
    }
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
    if shown == 0 {
        println!("no tokens found");
    }

    Ok(())
}
