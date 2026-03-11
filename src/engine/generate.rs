use std::collections::HashMap;

use burn::tensor::backend::Backend;

use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::core::tensor::softmax;
use crate::data::tokenizer::Tokenizer;
use crate::engine::model::LanguageModel;
use crate::engine::profile::{RuntimeProfile, measure};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StopCondition {
    token_sequences: Vec<Vec<usize>>,
}

impl StopCondition {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn from_text_sequences(tokenizer: &Tokenizer, stop_sequences: &[&str]) -> Self {
        Self {
            token_sequences: stop_sequences
                .iter()
                .filter(|sequence| !sequence.is_empty())
                .map(|sequence| tokenizer.encode_text(sequence))
                .filter(|sequence| !sequence.is_empty())
                .collect(),
        }
    }

    pub fn from_strings(tokenizer: &Tokenizer, stop_sequences: &[String]) -> Self {
        Self {
            token_sequences: stop_sequences
                .iter()
                .filter(|sequence| !sequence.is_empty())
                .map(|sequence| tokenizer.encode_text(sequence))
                .filter(|sequence| !sequence.is_empty())
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.token_sequences.is_empty()
    }

    pub fn matched_suffix_len(&self, generated_tokens: &[usize]) -> Option<usize> {
        self.token_sequences.iter().find_map(|sequence| {
            generated_tokens
                .ends_with(sequence)
                .then_some(sequence.len())
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingStrategy {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl SamplingStrategy {
    pub fn temperature_only(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(RustGptError::Config("temperature must be > 0".to_string()));
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(RustGptError::Config("top_p must be in (0, 1]".to_string()));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(RustGptError::Config(
                "repetition_penalty must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn generate_sample<B: Backend>(
    model: &LanguageModel<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
) -> Result<String> {
    let completion = generate_completion(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        strategy,
        &StopCondition::none(),
        profile,
        rng,
    )?;
    Ok(format!("{prompt}{completion}"))
}

pub fn generate_completion<B: Backend>(
    model: &LanguageModel<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
) -> Result<String> {
    generate_completion_inner(
        model,
        tokenizer,
        tokenizer.encode_text(prompt),
        max_new_tokens,
        strategy,
        stop_condition,
        profile,
        rng,
        None,
    )
}

pub fn generate_completion_from_tokens<B: Backend>(
    model: &LanguageModel<B>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
) -> Result<String> {
    generate_completion_inner(
        model,
        tokenizer,
        prompt_tokens.to_vec(),
        max_new_tokens,
        strategy,
        stop_condition,
        profile,
        rng,
        None,
    )
}

pub fn generate_completion_streaming<B: Backend>(
    model: &LanguageModel<B>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
    on_text: &mut dyn FnMut(&str),
) -> Result<String> {
    generate_completion_inner(
        model,
        tokenizer,
        prompt_tokens.to_vec(),
        max_new_tokens,
        strategy,
        stop_condition,
        profile,
        rng,
        Some(on_text),
    )
}

fn generate_completion_inner<B: Backend>(
    model: &LanguageModel<B>,
    tokenizer: &Tokenizer,
    prompt_tokens: Vec<usize>,
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
    mut on_text: Option<&mut dyn FnMut(&str)>,
) -> Result<String> {
    strategy.validate()?;

    let conditioning =
        conditioning_from_prompt_tokens(model, tokenizer, &prompt_tokens, max_new_tokens);
    let mut seen_token_counts = HashMap::new();
    for token_id in &conditioning {
        *seen_token_counts.entry(*token_id).or_insert(0_usize) += 1;
    }

    let mut cache = model.new_decode_cache();
    let mut logits = measure(profile, "sample.prefill", || {
        model.prefill(&conditioning, &mut cache)
    });
    let max_steps =
        max_new_tokens.min(model.config().block_size.saturating_sub(conditioning.len()));
    let mut generated_tokens = Vec::with_capacity(max_steps);
    let mut emitted_len = 0;

    for step in 0..max_steps {
        let next_token = measure(profile, "sample.select_token", || {
            select_next_token_from_tensor(&logits, strategy, &seen_token_counts, rng)
        })?;
        if tokenizer.is_end_token(next_token) {
            break;
        }

        generated_tokens.push(next_token);
        *seen_token_counts.entry(next_token).or_insert(0_usize) += 1;

        let visible_token_count =
            if let Some(stop_len) = stop_condition.matched_suffix_len(&generated_tokens) {
                generated_tokens.len().saturating_sub(stop_len)
            } else {
                generated_tokens.len()
            };
        let visible_tokens = &generated_tokens[..visible_token_count];
        let generated = if on_text.is_some() {
            tokenizer.decode_streaming(visible_tokens)?
        } else {
            tokenizer.decode(visible_tokens, true)?
        };
        if let Some(callback) = &mut on_text {
            if generated.len() > emitted_len {
                callback(&generated[emitted_len..]);
                emitted_len = generated.len();
            }
        }
        if visible_token_count < generated_tokens.len() {
            return Ok(generated);
        }

        let position = conditioning.len() + step;
        if position >= model.config().block_size.saturating_sub(1) {
            break;
        }
        logits = measure(profile, "sample.decode_step", || {
            model.forward_step(next_token, position, &mut cache)
        });
    }

    let generated = tokenizer.decode(&generated_tokens, true)?;
    if let Some(callback) = &mut on_text {
        if generated.len() > emitted_len {
            callback(&generated[emitted_len..]);
        }
    }
    Ok(generated)
}

fn conditioning_from_prompt_tokens<B: Backend>(
    model: &LanguageModel<B>,
    tokenizer: &Tokenizer,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
) -> Vec<usize> {
    let reserve = max_new_tokens.min(model.config().block_size.saturating_sub(1));
    let max_prompt_tokens = usize::max(1, model.config().block_size.saturating_sub(reserve));
    let prompt_tokens = if prompt_tokens.len() + 1 > max_prompt_tokens {
        &prompt_tokens[prompt_tokens
            .len()
            .saturating_sub(max_prompt_tokens.saturating_sub(1))..]
    } else {
        prompt_tokens
    };

    let mut conditioning = Vec::with_capacity(prompt_tokens.len() + 1);
    conditioning.push(tokenizer.bos_id());
    conditioning.extend_from_slice(prompt_tokens);
    conditioning
}

pub fn select_next_token(
    logits: &[f32],
    strategy: &SamplingStrategy,
    seen_token_counts: &HashMap<usize, usize>,
    rng: &mut Rng,
) -> Result<usize> {
    let adjusted = adjusted_logits(logits, strategy, seen_token_counts);
    let probs = softmax(&adjusted);
    rng.sample_weighted(&probs).ok_or_else(|| {
        RustGptError::Tensor("sampling failed because probabilities were invalid".to_string())
    })
}

fn select_next_token_from_tensor<B: Backend>(
    logits: &burn::tensor::Tensor<B, 1>,
    strategy: &SamplingStrategy,
    seen_token_counts: &HashMap<usize, usize>,
    rng: &mut Rng,
) -> Result<usize> {
    if can_use_device_topk_path(strategy, seen_token_counts) {
        let vocab_size = logits.dims()[0];
        let top_k = strategy.top_k.min(vocab_size);
        if top_k == 1 {
            return argmax_token(logits);
        }
        if top_k > 1 && top_k < vocab_size {
            return select_next_token_from_topk_tensor(logits, top_k, strategy, rng);
        }
    }

    select_next_token(&tensor_to_vec(logits)?, strategy, seen_token_counts, rng)
}

fn adjusted_logits(
    logits: &[f32],
    strategy: &SamplingStrategy,
    seen_token_counts: &HashMap<usize, usize>,
) -> Vec<f32> {
    let mut adjusted = logits.to_vec();
    apply_penalties(&mut adjusted, strategy, seen_token_counts);
    for value in &mut adjusted {
        *value /= strategy.temperature;
    }
    apply_top_k(&mut adjusted, strategy.top_k);
    apply_top_p(&mut adjusted, strategy.top_p);
    adjusted
}

fn adjusted_topk_logits(logits: &[f32], strategy: &SamplingStrategy) -> Vec<f32> {
    let mut adjusted = logits.to_vec();
    for value in &mut adjusted {
        *value /= strategy.temperature;
    }
    apply_top_p(&mut adjusted, strategy.top_p);
    adjusted
}

fn can_use_device_topk_path(
    strategy: &SamplingStrategy,
    _seen_token_counts: &HashMap<usize, usize>,
) -> bool {
    strategy.top_k > 0
        && strategy.repetition_penalty == 1.0
        && strategy.presence_penalty == 0.0
        && strategy.frequency_penalty == 0.0
}

fn apply_penalties(
    logits: &mut [f32],
    strategy: &SamplingStrategy,
    seen_token_counts: &HashMap<usize, usize>,
) {
    if strategy.repetition_penalty == 1.0
        && strategy.presence_penalty == 0.0
        && strategy.frequency_penalty == 0.0
    {
        return;
    }

    for (token_id, count) in seen_token_counts {
        let logit = &mut logits[*token_id];
        if strategy.repetition_penalty != 1.0 {
            if *logit >= 0.0 {
                *logit /= strategy.repetition_penalty;
            } else {
                *logit *= strategy.repetition_penalty;
            }
        }
        if strategy.presence_penalty != 0.0 {
            *logit -= strategy.presence_penalty;
        }
        if strategy.frequency_penalty != 0.0 {
            *logit -= strategy.frequency_penalty * *count as f32;
        }
    }
}

fn argmax_token<B: Backend>(logits: &burn::tensor::Tensor<B, 1>) -> Result<usize> {
    let token_ids = int_tensor_to_vec(&logits.clone().argmax(0))?;
    let token_id = *token_ids.first().ok_or_else(|| {
        RustGptError::Tensor("sampling argmax did not return a token id".to_string())
    })?;
    usize::try_from(token_id).map_err(|_| {
        RustGptError::Tensor(format!("sampling returned an invalid token id: {token_id}"))
    })
}

fn select_next_token_from_topk_tensor<B: Backend>(
    logits: &burn::tensor::Tensor<B, 1>,
    top_k: usize,
    strategy: &SamplingStrategy,
    rng: &mut Rng,
) -> Result<usize> {
    let (top_logits, top_indices) = logits.clone().topk_with_indices(top_k, 0);
    let adjusted = adjusted_topk_logits(&tensor_to_vec(&top_logits)?, strategy);
    let probs = softmax(&adjusted);
    let sampled_idx = rng.sample_weighted(&probs).ok_or_else(|| {
        RustGptError::Tensor("sampling failed because probabilities were invalid".to_string())
    })?;
    let top_indices = int_tensor_to_vec(&top_indices)?;
    let token_id = *top_indices.get(sampled_idx).ok_or_else(|| {
        RustGptError::Tensor("sampling picked an out-of-range top-k candidate".to_string())
    })?;
    usize::try_from(token_id).map_err(|_| {
        RustGptError::Tensor(format!("sampling returned an invalid token id: {token_id}"))
    })
}

fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut ranked = logits
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let threshold = ranked[top_k - 1].1;
    for value in logits.iter_mut() {
        if *value < threshold {
            *value = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let probs = softmax(logits);
    let mut ranked = probs
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));

    let mut keep = vec![false; logits.len()];
    let mut cumulative = 0.0;
    for (rank_idx, (token_id, prob)) in ranked.iter().enumerate() {
        keep[*token_id] = true;
        cumulative += *prob;
        if cumulative >= top_p && rank_idx > 0 {
            break;
        }
    }
    for (token_id, value) in logits.iter_mut().enumerate() {
        if !keep[token_id] {
            *value = f32::NEG_INFINITY;
        }
    }
}

fn tensor_to_vec<B: Backend>(tensor: &burn::tensor::Tensor<B, 1>) -> Result<Vec<f32>> {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| RustGptError::Tensor(format!("failed to read logits from backend: {err:?}")))
}

fn int_tensor_to_vec<B: Backend>(
    tensor: &burn::tensor::Tensor<B, 1, burn::tensor::Int>,
) -> Result<Vec<i64>> {
    let data = tensor.clone().into_data();
    Ok(data.iter::<i64>().collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use burn::tensor::{Tensor, TensorData};

    use crate::core::rng::Rng;

    use super::{SamplingStrategy, argmax_token, select_next_token_from_tensor};

    type TestBackend = NdArray<f32>;

    #[test]
    fn argmax_path_reads_only_the_winning_token() {
        let device = NdArrayDevice::Cpu;
        let logits = Tensor::<TestBackend, 1>::from_data([0.2, 1.8, 0.5], &device);

        assert_eq!(argmax_token(&logits).unwrap(), 1);
    }

    #[test]
    fn top_k_device_path_preserves_sampling_domain() {
        let device = NdArrayDevice::Cpu;
        let logits = Tensor::<TestBackend, 1>::from_data([0.1, 2.0, 1.5, -4.0], &device);
        let strategy = SamplingStrategy {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        };

        let token = select_next_token_from_tensor(
            &logits,
            &strategy,
            &HashMap::new(),
            &mut Rng::from_seed(7),
        )
        .unwrap();

        assert!(token == 1 || token == 2);
    }

    #[test]
    fn int_data_iteration_handles_i32_storage() {
        let data = TensorData::new(vec![1_i32, 7_i32, 9_i32], [3]);
        let values = data.iter::<i64>().collect::<Vec<_>>();

        assert_eq!(values, vec![1, 7, 9]);
    }
}
