use std::path::Path;
use std::time::{Duration, Instant};

use crate::core::config::{DataConfig, TrainConfig};
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
};
use crate::runtime::forward::forward_sequence_profiled_with_loss;
use crate::runtime::profile::{RuntimeProfile, measure};

#[derive(Clone, Debug)]
pub struct PreparedTrainingRun {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub dataset: Dataset,
    pub training_data: TrainingData,
    pub starting_step: usize,
    pub checkpoint_seed: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingLogEntry {
    pub step: usize,
    pub loss: f32,
    pub batch_size: usize,
    pub elapsed: Duration,
    pub first_doc: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingRunSummary {
    pub final_loss: f32,
    pub log_entries: Vec<TrainingLogEntry>,
}

pub fn prepare_training_run(
    data_config: &DataConfig,
    train_config: &TrainConfig,
    resume_path: Option<&Path>,
) -> Result<PreparedTrainingRun> {
    let dataset = Dataset::from_path(data_config)?;
    let (model, tokenizer, starting_step, checkpoint_seed) = if let Some(resume_path) = resume_path
    {
        let checkpoint = load_checkpoint(resume_path)?;
        (
            checkpoint.model,
            checkpoint.tokenizer,
            checkpoint.trained_steps,
            checkpoint.seed,
        )
    } else {
        let tokenizer = Tokenizer::from_docs(dataset.docs(), train_config.boundary_mode)?;
        let mut init_rng = Rng::from_seed(train_config.seed);
        let model = Model::new(
            train_config.to_model_config(tokenizer.vocab_size()),
            &mut init_rng,
        )?;
        (model, tokenizer, 0, train_config.seed)
    };
    let mut data_rng = Rng::from_seed(checkpoint_seed);
    let dataset = if data_config.shuffle {
        dataset.shuffled(&mut data_rng)
    } else {
        dataset
    };
    let training_data = TrainingData::from_dataset(&dataset, &tokenizer)?;

    Ok(PreparedTrainingRun {
        model,
        tokenizer,
        dataset,
        training_data,
        starting_step,
        checkpoint_seed,
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
) -> Result<TrainingRunSummary> {
    let started_at = Instant::now();
    let batch_size = usize::max(1, train_config.batch_size);
    let mut summary = TrainingRunSummary::default();

    for step in 0..train_config.steps {
        let absolute_step = starting_step + step;
        let should_log = step == 0
            || (step + 1) == train_config.steps
            || ((step + 1) % train_config.sample_every == 0);
        let grad_scale = 1.0 / batch_size as f32;
        let mut batch_loss = 0.0;
        if !backend.uses_device_optimizer() {
            measure(profile, "sync.weights", || backend.sync_model(model))?;
        }
        begin_gradient_accumulation(model, backend);
        let example_start_idx = absolute_step
            .checked_mul(batch_size)
            .unwrap_or(absolute_step);
        let batch_tokens =
            training_data.build_batch(example_start_idx, batch_size, model.cfg.block_size);
        let first_doc = tokenizer.decode(&batch_tokens[0], true)?;

        let batch_mean_loss = if let Some(loss) = measure(profile, "train.grouped_batch", || {
            backend.accumulate_training_batch(model, &batch_tokens, should_log, grad_scale, profile)
        })? {
            loss
        } else {
            for tokens in &batch_tokens {
                let sequence = forward_sequence_profiled_with_loss(
                    model, tokens, backend, profile, should_log, grad_scale,
                )?;
                if should_log {
                    batch_loss += sequence.mean_loss;
                }
                accumulate_sequence_gradients_profiled(model, &sequence, backend, profile)?;
            }
            batch_loss / batch_size as f32
        };
        summary.final_loss = batch_mean_loss;

        let lr_t = train_config.learning_rate * (1.0 - step as f32 / train_config.steps as f32);
        apply_optimizer_profiled(
            model,
            backend,
            lr_t,
            train_config.beta1,
            train_config.beta2,
            train_config.eps_adam,
            absolute_step + 1,
            profile,
        )?;

        if should_log && capture_logs {
            summary.log_entries.push(TrainingLogEntry {
                step: step + 1,
                loss: batch_mean_loss,
                batch_size,
                elapsed: started_at.elapsed(),
                first_doc: truncate_string(&first_doc, 24),
            });
        }
    }

    Ok(summary)
}

fn truncate_string(input: &str, max_chars: usize) -> String {
    let mut out = input.chars().take(max_chars).collect::<String>();
    if input.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}
