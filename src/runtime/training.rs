use std::path::Path;
use std::time::{Duration, Instant};

use crate::core::config::{ChatTemplateKind, DataConfig, LrScheduleKind, TrainConfig};
use crate::core::error::Result;
use crate::core::rng::Rng;
use crate::data::checkpoint::load_checkpoint;
use crate::data::corpus::Dataset;
use crate::data::tokenizer::Tokenizer;
use crate::data::training_data::TrainingData;
use crate::model::Model;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::backward::{
    accumulate_sequence_gradients_profiled, apply_optimizer_profiled, begin_gradient_accumulation,
    clip_gradients_profiled,
};
use crate::runtime::eval::evaluate_training_data;
use crate::runtime::forward::forward_example_profiled_with_loss;
use crate::runtime::profile::{RuntimeProfile, measure};

#[derive(Clone, Debug)]
pub struct PreparedTrainingRun {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub dataset: Dataset,
    pub training_data: TrainingData,
    pub validation_data: Option<TrainingData>,
    pub starting_step: usize,
    pub checkpoint_seed: u64,
    pub chat_template: ChatTemplateKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingLogEntry {
    pub step: usize,
    pub loss: f32,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub validation_loss: Option<f32>,
    pub best_validation_loss: Option<f32>,
    pub elapsed: Duration,
    pub first_doc: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingRunSummary {
    pub final_loss: f32,
    pub final_validation_loss: Option<f32>,
    pub best_validation_loss: Option<f32>,
    pub log_entries: Vec<TrainingLogEntry>,
}

pub type ValidationHook<'a> =
    dyn FnMut(usize, f32, &mut Model, &mut ComputeBackend) -> Result<()> + 'a;

pub fn prepare_training_run(
    data_config: &DataConfig,
    train_config: &TrainConfig,
    validation_config: Option<&DataConfig>,
    resume_path: Option<&Path>,
) -> Result<PreparedTrainingRun> {
    let dataset = Dataset::from_path(data_config)?;
    let (model, tokenizer, starting_step, checkpoint_seed, chat_template) =
        if let Some(resume_path) = resume_path {
            let checkpoint = load_checkpoint(resume_path)?;
            (
                checkpoint.model,
                checkpoint.tokenizer,
                checkpoint.trained_steps,
                checkpoint.seed,
                checkpoint.chat_template,
            )
        } else {
            let tokenizer = build_tokenizer(data_config, &dataset, train_config)?;
            let mut init_rng = Rng::from_seed(train_config.seed);
            let mut model_cfg = train_config.to_model_config(tokenizer.vocab_size());
            model_cfg.boundary_mode = tokenizer.boundary_mode();
            let model = Model::new(model_cfg, &mut init_rng)?;
            (
                model,
                tokenizer,
                0,
                train_config.seed,
                data_config.chat_template,
            )
        };
    let mut data_rng = Rng::from_seed(checkpoint_seed);
    let dataset = if data_config.shuffle {
        dataset.shuffled(&mut data_rng)
    } else {
        dataset
    };
    let training_data =
        TrainingData::from_dataset(&dataset, &tokenizer, train_config.mode, chat_template)?;
    let (training_data, validation_data) = if let Some(validation_config) = validation_config {
        let validation_dataset = Dataset::from_path(validation_config)?;
        let validation_data = TrainingData::from_dataset(
            &validation_dataset,
            &tokenizer,
            train_config.mode,
            validation_config.chat_template,
        )?;
        (training_data, Some(validation_data))
    } else if train_config.validation_ratio > 0.0 {
        let (training_data, validation_data) =
            training_data.split_validation(train_config.validation_ratio)?;
        (training_data, Some(validation_data))
    } else {
        (training_data, None)
    };

    Ok(PreparedTrainingRun {
        model,
        tokenizer,
        dataset,
        training_data,
        validation_data,
        starting_step,
        checkpoint_seed,
        chat_template,
    })
}

pub fn run_training_steps(
    model: &mut Model,
    tokenizer: &Tokenizer,
    training_data: &TrainingData,
    backend: &mut ComputeBackend,
    train_config: &TrainConfig,
    starting_step: usize,
    profile: Option<&RuntimeProfile>,
    capture_logs: bool,
    validation_data: Option<&TrainingData>,
    mut on_validation: Option<&mut ValidationHook<'_>>,
) -> Result<TrainingRunSummary> {
    let started_at = Instant::now();
    let batch_size = usize::max(1, train_config.batch_size);
    let mut summary = TrainingRunSummary::default();
    let mut best_validation_loss = None::<f32>;

    for step in 0..train_config.steps {
        let absolute_step = starting_step + step;
        let should_log = step == 0
            || (step + 1) == train_config.steps
            || ((step + 1) % train_config.sample_every == 0);
        let grad_scale = 1.0 / batch_size as f32;
        let mut batch_loss = 0.0;
        let mut batch_loss_weight_sum = 0.0;
        if !backend.uses_device_optimizer() {
            measure(profile, "sync.weights", || backend.sync_model(model))?;
        }
        begin_gradient_accumulation(model, backend);
        let example_start_idx = absolute_step
            .checked_mul(batch_size)
            .unwrap_or(absolute_step);
        let batch_examples =
            training_data.build_batch(example_start_idx, batch_size, model.cfg.block_size);
        let first_doc = tokenizer.decode(&batch_examples[0].tokens_with_boundaries(), true)?;

        let batch_mean_loss = if let Some(loss) = measure(profile, "train.grouped_batch", || {
            backend.accumulate_training_batch(
                model,
                &batch_examples,
                should_log,
                grad_scale,
                profile,
            )
        })? {
            loss
        } else {
            for example in &batch_examples {
                let sequence = forward_example_profiled_with_loss(
                    model, example, backend, profile, should_log, grad_scale,
                )?;
                if should_log {
                    batch_loss += sequence.mean_loss * sequence.loss_weight_sum;
                    batch_loss_weight_sum += sequence.loss_weight_sum;
                }
                accumulate_sequence_gradients_profiled(model, &sequence, backend, profile)?;
            }
            if should_log && batch_loss_weight_sum > 0.0 {
                batch_loss / batch_loss_weight_sum
            } else {
                0.0
            }
        };
        summary.final_loss = batch_mean_loss;

        if train_config.grad_clip > 0.0 {
            clip_gradients_profiled(model, backend, train_config.grad_clip, profile)?;
        }

        let lr_t = learning_rate_at_step(train_config, step);
        apply_optimizer_profiled(
            model,
            backend,
            lr_t,
            train_config.beta1,
            train_config.beta2,
            train_config.eps_adam,
            train_config.weight_decay,
            absolute_step + 1,
            profile,
        )?;

        let validation_loss = if should_log {
            if let Some(validation_data) = validation_data {
                let metrics = measure(profile, "eval.validation", || {
                    evaluate_training_data(
                        model,
                        validation_data,
                        backend,
                        model.cfg.block_size,
                        train_config.validation_max_examples,
                        profile,
                    )
                })?;
                summary.final_validation_loss = Some(metrics.mean_loss);
                if best_validation_loss
                    .map(|best| metrics.mean_loss < best)
                    .unwrap_or(true)
                {
                    best_validation_loss = Some(metrics.mean_loss);
                    summary.best_validation_loss = best_validation_loss;
                    if let Some(callback) = &mut on_validation {
                        callback(absolute_step + 1, metrics.mean_loss, model, backend)?;
                    }
                }
                Some(metrics.mean_loss)
            } else {
                None
            }
        } else {
            None
        };

        if should_log && capture_logs {
            summary.log_entries.push(TrainingLogEntry {
                step: step + 1,
                loss: batch_mean_loss,
                batch_size,
                learning_rate: lr_t,
                validation_loss,
                best_validation_loss,
                elapsed: started_at.elapsed(),
                first_doc: truncate_string(&first_doc, 24),
            });
        }
    }

    Ok(summary)
}

fn build_tokenizer(
    data_config: &DataConfig,
    dataset: &Dataset,
    train_config: &TrainConfig,
) -> Result<Tokenizer> {
    if let Some(path) = &data_config.tokenizer_path {
        Tokenizer::from_tokenizer_file(
            path,
            data_config.tokenizer_bos.as_deref(),
            data_config.tokenizer_eos.as_deref(),
        )
    } else {
        let docs = dataset.docs_with_template(data_config.chat_template);
        Tokenizer::from_docs(&docs, train_config.boundary_mode)
    }
}

fn truncate_string(input: &str, max_chars: usize) -> String {
    let mut out = input.chars().take(max_chars).collect::<String>();
    if input.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn learning_rate_at_step(train_config: &TrainConfig, step: usize) -> f32 {
    if train_config.steps <= 1 {
        return train_config.learning_rate;
    }

    if train_config.warmup_steps > 0 && step < train_config.warmup_steps {
        let warmup_progress = (step + 1) as f32 / train_config.warmup_steps as f32;
        return train_config.learning_rate * warmup_progress;
    }

    let decay_start = train_config
        .warmup_steps
        .min(train_config.steps.saturating_sub(1));
    let decay_steps = train_config.steps.saturating_sub(decay_start).max(1);
    let decay_progress =
        (step.saturating_sub(decay_start)) as f32 / decay_steps.saturating_sub(1).max(1) as f32;
    let schedule_scale = match train_config.lr_schedule {
        LrScheduleKind::Linear => 1.0 - decay_progress,
        LrScheduleKind::Cosine => 0.5 * (1.0 + (std::f32::consts::PI * decay_progress).cos()),
    };
    train_config.learning_rate * schedule_scale.max(0.0)
}
