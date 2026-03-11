//! Shared evaluation helpers used by both the `eval` command and training-time validation.

use crate::core::error::{Result, RustGptError};
use crate::data::training_data::TrainingData;
use crate::model::Model;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::forward::forward_example_profiled_with_loss;
use crate::runtime::profile::RuntimeProfile;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalMetrics {
    pub mean_loss: f32,
    pub perplexity: f32,
    pub examples: usize,
}

pub fn evaluate_training_data(
    model: &Model,
    training_data: &TrainingData,
    backend: &ComputeBackend,
    block_size: usize,
    max_examples: usize,
    profile: Option<&RuntimeProfile>,
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
        let cache =
            forward_example_profiled_with_loss(model, &example, backend, profile, true, 1.0)?;
        total_loss += cache.mean_loss * cache.loss_weight_sum;
        total_weight += cache.loss_weight_sum;
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

#[cfg(test)]
mod tests {
    use crate::core::config::{ActivationKind, BoundaryMode, ModelConfig, PositionEncodingKind};
    use crate::core::rng::Rng;
    use crate::data::tokenizer::Tokenizer;
    use crate::data::training_data::{SequenceExample, TrainingData};
    use crate::model::Model;
    use crate::runtime::backend::ComputeBackend;
    use crate::runtime::forward::forward_example_profiled_with_loss;

    use super::evaluate_training_data;

    #[test]
    fn evaluation_weights_mean_loss_by_supervised_tokens() {
        let docs = vec!["abba".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: 8,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 4,
            tie_embeddings: false,
            activation: ActivationKind::Relu,
            position_encoding: PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
        };
        let mut init_rng = Rng::from_seed(42);
        let model = Model::new(cfg, &mut init_rng).unwrap();
        let backend = ComputeBackend::cpu();

        let encoded = tokenizer.encode_with_boundaries("abba").unwrap();
        let examples = vec![
            SequenceExample {
                input_ids: encoded[..2].to_vec(),
                target_ids: encoded[1..3].to_vec(),
                loss_mask: vec![1.0, 0.0],
            },
            SequenceExample {
                input_ids: encoded[..4].to_vec(),
                target_ids: encoded[1..5].to_vec(),
                loss_mask: vec![1.0, 1.0, 1.0, 1.0],
            },
        ];
        let training_data = TrainingData::Examples {
            examples: examples.clone(),
        };

        let metrics = evaluate_training_data(
            &model,
            &training_data,
            &backend,
            model.cfg.block_size,
            8,
            None,
        )
        .unwrap();
        let caches = examples
            .iter()
            .map(|example| {
                forward_example_profiled_with_loss(&model, example, &backend, None, true, 1.0)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let total_loss = caches
            .iter()
            .map(|cache| cache.mean_loss * cache.loss_weight_sum)
            .sum::<f32>();
        let total_weight = caches
            .iter()
            .map(|cache| cache.loss_weight_sum)
            .sum::<f32>();
        let expected = total_loss / total_weight;

        assert!((metrics.mean_loss - expected).abs() < 1e-6);
    }
}
