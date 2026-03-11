use std::path::Path;
use std::time::{Duration, Instant};

use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Int, Tensor, TensorData};

use crate::core::config::{ChatTemplateKind, DataConfig, LrScheduleKind, TrainConfig};
use crate::core::error::{Result, RustGptError};
use crate::data::corpus::Dataset;
use crate::data::tokenizer::Tokenizer;
use crate::data::training_data::{SequenceExample, TrainingData};
use crate::model::lm::{LanguageModel, LanguageModelOptimizer, build_optimizer, init_model};
use crate::runtime::checkpoint::{CheckpointRunManifest, read_checkpoint_metadata};
use crate::runtime::profile::{RuntimeProfile, measure};

#[derive(Clone, Debug)]
pub struct PreparedTrainingRun {
    pub model_config: crate::core::config::ModelConfig,
    pub tokenizer: Tokenizer,
    pub dataset: Dataset,
    pub training_data: TrainingData,
    pub validation_data: Option<TrainingData>,
    pub starting_step: usize,
    pub checkpoint_seed: u64,
    pub chat_template: ChatTemplateKind,
    pub resume_manifest: Option<CheckpointRunManifest>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingLogEntry {
    pub run_step: usize,
    pub total_step: usize,
    pub loss: f32,
    pub batch_size: usize,
    pub processed_tokens: usize,
    pub learning_rate: f32,
    pub tokens_per_second: f32,
    pub step_elapsed: Duration,
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
    pub total_tokens: usize,
    pub average_tokens_per_second: f32,
    pub elapsed: Duration,
    pub log_entries: Vec<TrainingLogEntry>,
}

pub struct TrainingArtifacts<AD: AutodiffBackend> {
    pub model: LanguageModel<AD>,
    pub optimizer: LanguageModelOptimizer<AD>,
}

pub type ValidationHook<'a, AD> =
    dyn FnMut(usize, f32, &LanguageModel<AD>, &LanguageModelOptimizer<AD>) -> Result<()> + 'a;

pub struct TrainingRunRequest<'a, AD: AutodiffBackend> {
    pub tokenizer: &'a Tokenizer,
    pub training_data: &'a TrainingData,
    pub train_config: &'a TrainConfig,
    pub starting_step: usize,
    pub device: &'a AD::Device,
    pub profile: Option<&'a RuntimeProfile>,
    pub capture_logs: bool,
    pub validation_data: Option<&'a TrainingData>,
    pub on_validation: Option<&'a mut ValidationHook<'a, AD>>,
}

pub fn prepare_training_run(
    data_config: &DataConfig,
    train_config: &TrainConfig,
    validation_config: Option<&DataConfig>,
    resume_path: Option<&Path>,
) -> Result<PreparedTrainingRun> {
    let dataset = Dataset::from_path(data_config)?;
    let (model_config, tokenizer, starting_step, checkpoint_seed, chat_template, resume_manifest) =
        if let Some(resume_path) = resume_path {
            let checkpoint = read_checkpoint_metadata(resume_path)?;
            (
                checkpoint.model_config,
                checkpoint.tokenizer,
                checkpoint.trained_steps,
                checkpoint.seed,
                checkpoint.chat_template,
                checkpoint.run_manifest,
            )
        } else {
            let tokenizer = build_tokenizer(data_config, &dataset, train_config)?;
            let mut model_cfg = train_config.to_model_config(tokenizer.vocab_size());
            model_cfg.boundary_mode = tokenizer.boundary_mode();
            (
                model_cfg,
                tokenizer,
                0,
                train_config.seed,
                data_config.chat_template,
                None,
            )
        };
    let mut data_rng = crate::core::rng::Rng::from_seed(checkpoint_seed);
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
        model_config,
        tokenizer,
        dataset,
        training_data,
        validation_data,
        starting_step,
        checkpoint_seed,
        chat_template,
        resume_manifest,
    })
}

impl<AD: AutodiffBackend> TrainingArtifacts<AD> {
    pub fn new(
        model_config: &crate::core::config::ModelConfig,
        train_config: &TrainConfig,
        device: &AD::Device,
    ) -> Result<Self> {
        Ok(Self {
            model: init_model::<AD>(model_config, train_config.seed, device)?,
            optimizer: build_optimizer::<AD>(train_config),
        })
    }
}

pub fn run_training_steps<B, AD>(
    artifacts: TrainingArtifacts<AD>,
    request: TrainingRunRequest<'_, AD>,
) -> Result<(TrainingArtifacts<AD>, TrainingRunSummary)>
where
    B: Backend<Device = AD::Device>,
    AD: AutodiffBackend<InnerBackend = B>,
{
    let TrainingRunRequest {
        tokenizer,
        training_data,
        train_config,
        starting_step,
        device,
        profile,
        capture_logs,
        validation_data,
        mut on_validation,
    } = request;
    let started_at = Instant::now();
    let batch_size = usize::max(1, train_config.batch_size);
    let grad_accum_steps = usize::max(1, train_config.gradient_accumulation_steps);
    validate_train_loop_config(train_config)?;
    let mut summary = TrainingRunSummary::default();
    let mut best_validation_loss = None::<f32>;
    let mut model = artifacts.model;
    let mut optimizer = artifacts.optimizer;
    let total_steps = total_training_steps(starting_step, train_config.steps);

    for step in 0..train_config.steps {
        let step_started_at = Instant::now();
        let absolute_step = starting_step + step;
        let should_log = should_log_step(step, absolute_step, train_config);
        let example_start_idx = absolute_step
            .checked_mul(batch_size.saturating_mul(grad_accum_steps))
            .unwrap_or(absolute_step);
        let mut processed_tokens = 0;
        let mut mean_loss_sum = 0.0;
        let mut first_doc = None;
        let mut grads_accumulator = GradientsAccumulator::<LanguageModel<AD>>::new();

        for micro_step in 0..grad_accum_steps {
            let micro_start_idx = example_start_idx + micro_step * batch_size;
            let batch_examples =
                training_data.build_batch(micro_start_idx, batch_size, model.config().block_size);
            processed_tokens += batch_examples
                .iter()
                .map(SequenceExample::len)
                .sum::<usize>();
            if first_doc.is_none() {
                first_doc =
                    Some(tokenizer.decode(&batch_examples[0].tokens_with_boundaries(), true)?);
            }
            let batch = measure(profile, "train.batch.build", || {
                build_batch_tensors::<AD>(&batch_examples, tokenizer, device)
            })?;

            let logits = measure(profile, "train.forward", || {
                model.forward(batch.input_ids, Some(batch.pad_mask))
            });
            let (loss, batch_mean_loss) = measure(profile, "train.loss", || {
                compute_weighted_loss(
                    logits,
                    batch.target_ids,
                    batch.loss_weights,
                    batch.weight_sum,
                )
            })?;
            mean_loss_sum += batch_mean_loss;

            let scaled_loss = loss.div_scalar(grad_accum_steps as f32);
            let grads = measure(profile, "train.backward", || scaled_loss.backward());
            let grads = GradientsParams::from_grads(grads, &model);
            grads_accumulator.accumulate(&model, grads);
        }

        summary.total_tokens += processed_tokens;
        let batch_mean_loss = mean_loss_sum / grad_accum_steps as f32;
        summary.final_loss = batch_mean_loss;
        let first_doc = first_doc.unwrap_or_default();
        let grads = grads_accumulator.grads();
        let learning_rate = learning_rate_at_step(train_config, absolute_step, total_steps) as f64;
        model = measure(profile, "train.optimize", || {
            optimizer.step(learning_rate, model, grads)
        });

        let validation_loss = if should_log {
            if let Some(validation_data) = validation_data {
                let valid_model = model.valid();
                let metrics = measure(profile, "eval.validation", || {
                    evaluate_training_data::<B>(
                        &valid_model,
                        tokenizer,
                        validation_data,
                        model.config().block_size,
                        train_config.validation_max_examples,
                        device,
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
                        callback(absolute_step + 1, metrics.mean_loss, &model, &optimizer)?;
                    }
                }
                Some(metrics.mean_loss)
            } else {
                None
            }
        } else {
            None
        };
        let step_elapsed = step_started_at.elapsed();
        let tokens_per_second = processed_tokens as f32 / step_elapsed.as_secs_f32().max(1e-6);

        if should_log && capture_logs {
            summary.log_entries.push(TrainingLogEntry {
                run_step: step + 1,
                total_step: absolute_step + 1,
                loss: batch_mean_loss,
                batch_size: batch_size * grad_accum_steps,
                processed_tokens,
                learning_rate: learning_rate as f32,
                tokens_per_second,
                step_elapsed,
                validation_loss,
                best_validation_loss,
                elapsed: started_at.elapsed(),
                first_doc: truncate_string(&first_doc, 24),
            });
        }
    }

    summary.elapsed = started_at.elapsed();
    summary.average_tokens_per_second =
        summary.total_tokens as f32 / summary.elapsed.as_secs_f32().max(1e-6);

    Ok((TrainingArtifacts { model, optimizer }, summary))
}

fn validate_train_loop_config(train_config: &TrainConfig) -> Result<()> {
    if train_config.sample_every == 0 {
        return Err(RustGptError::Config(
            "sample_every must be at least 1".to_string(),
        ));
    }
    if train_config.gradient_accumulation_steps == 0 {
        return Err(RustGptError::Config(
            "gradient_accumulation_steps must be at least 1".to_string(),
        ));
    }
    if train_config.validation_max_examples == 0 {
        return Err(RustGptError::Config(
            "validation_max_examples must be at least 1".to_string(),
        ));
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalMetrics {
    pub mean_loss: f32,
    pub perplexity: f32,
    pub examples: usize,
}

pub fn evaluate_training_data<B: Backend>(
    model: &LanguageModel<B>,
    tokenizer: &Tokenizer,
    training_data: &TrainingData,
    block_size: usize,
    max_examples: usize,
    device: &B::Device,
) -> Result<EvalMetrics> {
    let example_count = usize::min(max_examples, training_data.example_count(block_size));
    if example_count == 0 {
        return Err(RustGptError::Data(
            "evaluation requires at least one example".to_string(),
        ));
    }

    let mut total_loss = 0.0;
    let mut total_weight = 0.0;
    for example_idx in 0..example_count {
        let example = training_data
            .build_batch(example_idx, 1, block_size)
            .pop()
            .ok_or_else(|| RustGptError::Data("evaluation batch was empty".to_string()))?;
        let batch = build_batch_tensors::<B>(&[example], tokenizer, device)?;
        let logits = model.forward(batch.input_ids, Some(batch.pad_mask));
        let (_, mean_loss) = compute_weighted_loss(
            logits,
            batch.target_ids,
            batch.loss_weights,
            batch.weight_sum,
        )?;
        total_loss += mean_loss * batch.weight_sum;
        total_weight += batch.weight_sum;
    }

    if total_weight <= 0.0 {
        return Err(RustGptError::Data(
            "evaluation examples did not contain any supervised tokens".to_string(),
        ));
    }

    let mean_loss = total_loss / total_weight;
    Ok(EvalMetrics {
        mean_loss,
        perplexity: mean_loss.exp(),
        examples: example_count,
    })
}

struct BatchTensors<B: Backend> {
    input_ids: Tensor<B, 2, Int>,
    target_ids: Tensor<B, 2, Int>,
    loss_weights: Tensor<B, 2>,
    pad_mask: Tensor<B, 2, Bool>,
    weight_sum: f32,
}

fn build_batch_tensors<B: Backend>(
    examples: &[SequenceExample],
    tokenizer: &Tokenizer,
    device: &B::Device,
) -> Result<BatchTensors<B>> {
    let batch_size = examples.len();
    let seq_length = examples.iter().map(SequenceExample::len).max().unwrap_or(0);
    if seq_length == 0 {
        return Err(RustGptError::Data(
            "training batch examples must not be empty".to_string(),
        ));
    }
    let pad_token = tokenizer.eos_id().unwrap_or(tokenizer.bos_id());
    build_batch_tensors_with_pad::<B>(examples, pad_token, batch_size, seq_length, device)
}

fn build_batch_tensors_with_pad<B: Backend>(
    examples: &[SequenceExample],
    pad_token: usize,
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Result<BatchTensors<B>> {
    let mut input_ids = vec![pad_token as i64; batch_size * seq_length];
    let mut target_ids = vec![pad_token as i64; batch_size * seq_length];
    let mut loss_weights = vec![0.0_f32; batch_size * seq_length];
    let mut pad_mask = vec![true; batch_size * seq_length];
    let mut weight_sum = 0.0_f32;

    for (row, example) in examples.iter().enumerate() {
        for col in 0..example.len() {
            let offset = row * seq_length + col;
            input_ids[offset] = example.input_ids[col] as i64;
            target_ids[offset] = example.target_ids[col] as i64;
            loss_weights[offset] = example.loss_mask[col];
            pad_mask[offset] = false;
            weight_sum += example.loss_mask[col];
        }
    }

    if weight_sum <= 0.0 {
        return Err(RustGptError::Data(
            "training batch did not contain any supervised tokens".to_string(),
        ));
    }
    Ok(BatchTensors {
        input_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_ids, [batch_size, seq_length]),
            device,
        ),
        target_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_ids, [batch_size, seq_length]),
            device,
        ),
        loss_weights: Tensor::<B, 2>::from_data(
            TensorData::new(loss_weights, [batch_size, seq_length]),
            device,
        ),
        pad_mask: Tensor::<B, 2, Bool>::from_data(
            TensorData::new(pad_mask, [batch_size, seq_length]),
            device,
        ),
        weight_sum,
    })
}

fn compute_weighted_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    loss_weights: Tensor<B, 2>,
    weight_sum: f32,
) -> Result<(Tensor<B, 1>, f32)> {
    let [batch_size, seq_length, vocab_size] = logits.dims();
    let flat_logits = logits.reshape([batch_size * seq_length, vocab_size]);
    let flat_targets = targets.reshape([batch_size * seq_length]);
    let flat_weights = loss_weights.reshape([batch_size * seq_length]);
    let gathered = log_softmax(flat_logits, 1).gather(
        1,
        flat_targets.clone().reshape([batch_size * seq_length, 1]),
    );
    let loss = gathered.reshape([batch_size * seq_length]) * flat_weights;
    let loss = loss.sum().neg() / weight_sum;
    let mean_loss = scalar_from_tensor(&loss)?;
    Ok((loss, mean_loss))
}

fn scalar_from_tensor<B: Backend>(tensor: &Tensor<B, 1>) -> Result<f32> {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| RustGptError::Tensor(format!("failed to read scalar from backend: {err:?}")))
        .and_then(|values| {
            values.into_iter().next().ok_or_else(|| {
                RustGptError::Tensor("backend tensor did not contain a scalar".to_string())
            })
        })
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

fn total_training_steps(starting_step: usize, run_steps: usize) -> usize {
    starting_step.saturating_add(run_steps)
}

fn should_log_step(run_step: usize, absolute_step: usize, train_config: &TrainConfig) -> bool {
    run_step == 0
        || (run_step + 1) == train_config.steps
        || (absolute_step + 1).is_multiple_of(train_config.sample_every)
}

pub fn learning_rate_at_step(train_config: &TrainConfig, step: usize, total_steps: usize) -> f32 {
    if total_steps <= 1 {
        return train_config.learning_rate;
    }

    if train_config.warmup_steps > 0 && step < train_config.warmup_steps {
        let warmup_progress = (step + 1) as f32 / train_config.warmup_steps as f32;
        return train_config.learning_rate * warmup_progress;
    }

    let decay_start = train_config.warmup_steps.min(total_steps.saturating_sub(1));
    let decay_steps = total_steps.saturating_sub(decay_start).max(1);
    let decay_progress =
        (step.saturating_sub(decay_start)) as f32 / decay_steps.saturating_sub(1).max(1) as f32;
    let schedule_scale = match train_config.lr_schedule {
        LrScheduleKind::Linear => 1.0 - decay_progress,
        LrScheduleKind::Cosine => 0.5 * (1.0 + (std::f32::consts::PI * decay_progress).cos()),
    };
    train_config.learning_rate * schedule_scale.max(0.0)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::backend::{Autodiff, NdArray};

    use crate::core::config::{
        BoundaryMode, ChatTemplateKind, DataFormat, LrScheduleKind, TrainConfig, TrainMode,
    };
    use crate::data::corpus::Dataset;
    use crate::data::tokenizer::Tokenizer;
    use crate::data::training_data::TrainingData;
    use crate::runtime::checkpoint::{
        SaveTrainingCheckpointRequest, load_training_checkpoint, save_training_checkpoint,
    };

    use super::{
        TrainingArtifacts, TrainingRunRequest, learning_rate_at_step, run_training_steps,
        should_log_step, total_training_steps,
    };

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<TestBackend>;
    type TestCheckpointAutodiffBackend = Autodiff<TestBackend, BalancedCheckpointing>;

    fn temp_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rustgpt-{unique}-{name}"))
    }

    #[test]
    fn resume_schedule_uses_absolute_step_against_total_target() {
        let train_config = TrainConfig {
            steps: 4,
            learning_rate: 1.0,
            warmup_steps: 2,
            lr_schedule: LrScheduleKind::Linear,
            ..TrainConfig::default()
        };
        let total_steps = total_training_steps(6, train_config.steps);

        let resumed_lr = learning_rate_at_step(&train_config, 6, total_steps);
        assert!(resumed_lr > 0.0);
        assert!(resumed_lr < 1.0);
    }

    #[test]
    fn logging_cadence_continues_from_absolute_step() {
        let train_config = TrainConfig {
            steps: 3,
            sample_every: 4,
            ..TrainConfig::default()
        };

        assert!(should_log_step(0, 7, &train_config));
        assert!(!should_log_step(1, 8, &train_config));
        assert!(should_log_step(2, 9, &train_config));
    }

    #[test]
    fn tiny_run_can_overfit_a_single_example() {
        let dataset = Dataset::from_text("hello\n", DataFormat::Lines, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training_data = TrainingData::from_dataset(
            &dataset,
            &tokenizer,
            TrainMode::Auto,
            ChatTemplateKind::SimpleTranscript,
        )
        .unwrap();
        let train_config = TrainConfig {
            steps: 12,
            sample_every: 1,
            batch_size: 1,
            gradient_accumulation_steps: 2,
            learning_rate: 0.05,
            warmup_steps: 0,
            lr_schedule: LrScheduleKind::Linear,
            block_size: 8,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 0,
            ..TrainConfig::default()
        };
        let model_config = train_config.to_model_config(tokenizer.vocab_size());
        let device = Default::default();
        let artifacts =
            TrainingArtifacts::<TestAutodiffBackend>::new(&model_config, &train_config, &device)
                .unwrap();

        let (_artifacts, summary) = run_training_steps::<TestBackend, TestAutodiffBackend>(
            artifacts,
            TrainingRunRequest {
                tokenizer: &tokenizer,
                training_data: &training_data,
                train_config: &train_config,
                starting_step: 0,
                device: &device,
                profile: None,
                capture_logs: true,
                validation_data: None,
                on_validation: None,
            },
        )
        .unwrap();

        let first_loss = summary.log_entries.first().unwrap().loss;
        let last_loss = summary.log_entries.last().unwrap().loss;
        assert!(last_loss < first_loss, "{first_loss} !> {last_loss}");
        assert!(summary.total_tokens > 0);
        assert!(summary.average_tokens_per_second > 0.0);
        assert_eq!(summary.log_entries.first().unwrap().batch_size, 2);
        assert!(
            summary
                .log_entries
                .iter()
                .all(|entry| entry.tokens_per_second > 0.0 && entry.processed_tokens > 0)
        );
    }

    #[test]
    fn checkpoint_resume_smoke_test_continues_total_step_count() {
        let dataset = Dataset::from_text("hello\nworld\n", DataFormat::Lines, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training_data = TrainingData::from_dataset(
            &dataset,
            &tokenizer,
            TrainMode::Auto,
            ChatTemplateKind::SimpleTranscript,
        )
        .unwrap();
        let train_config = TrainConfig {
            steps: 2,
            sample_every: 1,
            batch_size: 1,
            learning_rate: 0.05,
            warmup_steps: 0,
            lr_schedule: LrScheduleKind::Linear,
            block_size: 8,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 0,
            ..TrainConfig::default()
        };
        let model_config = train_config.to_model_config(tokenizer.vocab_size());
        let device = Default::default();
        let artifacts =
            TrainingArtifacts::<TestAutodiffBackend>::new(&model_config, &train_config, &device)
                .unwrap();

        let (artifacts, _) = run_training_steps::<TestBackend, TestAutodiffBackend>(
            artifacts,
            TrainingRunRequest {
                tokenizer: &tokenizer,
                training_data: &training_data,
                train_config: &train_config,
                starting_step: 0,
                device: &device,
                profile: None,
                capture_logs: true,
                validation_data: None,
                on_validation: None,
            },
        )
        .unwrap();

        let temp_dir = temp_dir("resume-smoke");
        fs::create_dir_all(&temp_dir).unwrap();
        let checkpoint = temp_dir.join("resume.ckpt");
        save_training_checkpoint(SaveTrainingCheckpointRequest {
            path: &checkpoint,
            model: &artifacts.model,
            optimizer: Some(&artifacts.optimizer),
            tokenizer: &tokenizer,
            chat_template: ChatTemplateKind::SimpleTranscript,
            trained_steps: 2,
            seed: train_config.seed,
            run_manifest: None,
        })
        .unwrap();

        let loaded =
            load_training_checkpoint::<TestAutodiffBackend>(&checkpoint, &device, &train_config)
                .unwrap();
        let resumed_artifacts = TrainingArtifacts {
            model: loaded.model,
            optimizer: loaded.optimizer,
        };
        let (_artifacts, resumed_summary) = run_training_steps::<TestBackend, TestAutodiffBackend>(
            resumed_artifacts,
            TrainingRunRequest {
                tokenizer: &tokenizer,
                training_data: &training_data,
                train_config: &train_config,
                starting_step: 2,
                device: &device,
                profile: None,
                capture_logs: true,
                validation_data: None,
                on_validation: None,
            },
        )
        .unwrap();

        assert_eq!(resumed_summary.log_entries.first().unwrap().total_step, 3);
        assert_eq!(resumed_summary.log_entries.last().unwrap().total_step, 4);
        assert!(resumed_summary.final_loss.is_finite());

        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn tiny_run_supports_activation_checkpointing_backend() {
        let dataset = Dataset::from_text("hello\n", DataFormat::Lines, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training_data = TrainingData::from_dataset(
            &dataset,
            &tokenizer,
            TrainMode::Auto,
            ChatTemplateKind::SimpleTranscript,
        )
        .unwrap();
        let train_config = TrainConfig {
            steps: 2,
            sample_every: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            activation_checkpointing: true,
            learning_rate: 0.05,
            warmup_steps: 0,
            lr_schedule: LrScheduleKind::Linear,
            block_size: 8,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 0,
            ..TrainConfig::default()
        };
        let model_config = train_config.to_model_config(tokenizer.vocab_size());
        let device = Default::default();
        let artifacts = TrainingArtifacts::<TestCheckpointAutodiffBackend>::new(
            &model_config,
            &train_config,
            &device,
        )
        .unwrap();

        let (_artifacts, summary) =
            run_training_steps::<TestBackend, TestCheckpointAutodiffBackend>(
                artifacts,
                TrainingRunRequest {
                    tokenizer: &tokenizer,
                    training_data: &training_data,
                    train_config: &train_config,
                    starting_step: 0,
                    device: &device,
                    profile: None,
                    capture_logs: true,
                    validation_data: None,
                    on_validation: None,
                },
            )
            .unwrap();

        assert_eq!(summary.log_entries.len(), 2);
        assert!(summary.final_loss.is_finite());
    }
}
