use std::collections::HashMap;

use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::core::tensor::softmax;
use crate::data::tokenizer::Tokenizer;
use crate::model::Model;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::forward::forward_token_profiled;
use crate::runtime::profile::{RuntimeProfile, measure};
use crate::runtime::workspace::new_kv_cache;

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

    pub(crate) fn matched_suffix_len(&self, generated_tokens: &[usize]) -> Option<usize> {
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

pub fn generate_sample(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    rng: &mut Rng,
) -> Result<String> {
    let backend = ComputeBackend::cpu();
    generate_sample_with_backend(
        model,
        &backend,
        tokenizer,
        prompt,
        max_new_tokens,
        &SamplingStrategy::temperature_only(temperature),
        None,
        rng,
    )
}

pub fn generate_completion(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    stop_sequences: &[&str],
    rng: &mut Rng,
) -> Result<String> {
    let backend = ComputeBackend::cpu();
    generate_completion_with_backend(
        model,
        &backend,
        tokenizer,
        prompt,
        max_new_tokens,
        &SamplingStrategy::temperature_only(temperature),
        &StopCondition::from_text_sequences(tokenizer, stop_sequences),
        None,
        rng,
    )
}

pub fn generate_sample_with_backend(
    model: &Model,
    backend: &ComputeBackend,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
) -> Result<String> {
    let completion = generate_completion_with_backend(
        model,
        backend,
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

pub fn generate_completion_with_backend(
    model: &Model,
    backend: &ComputeBackend,
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
        backend,
        tokenizer,
        Some(prompt),
        None,
        max_new_tokens,
        strategy,
        stop_condition,
        profile,
        rng,
        None,
    )
}

pub fn generate_completion_from_tokens_with_backend(
    model: &Model,
    backend: &ComputeBackend,
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
        backend,
        tokenizer,
        None,
        Some(prompt_tokens),
        max_new_tokens,
        strategy,
        stop_condition,
        profile,
        rng,
        None,
    )
}

pub fn generate_completion_streaming_with_backend(
    model: &Model,
    backend: &ComputeBackend,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
    on_text: &mut dyn FnMut(&str),
) -> Result<String> {
    generate_completion_inner(
        model,
        backend,
        tokenizer,
        Some(prompt),
        None,
        max_new_tokens,
        strategy,
        stop_condition,
        profile,
        rng,
        Some(on_text),
    )
}

pub fn generate_completion_from_tokens_streaming_with_backend(
    model: &Model,
    backend: &ComputeBackend,
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
        backend,
        tokenizer,
        None,
        Some(prompt_tokens),
        max_new_tokens,
        strategy,
        stop_condition,
        profile,
        rng,
        Some(on_text),
    )
}

fn generate_completion_inner(
    model: &Model,
    backend: &ComputeBackend,
    tokenizer: &Tokenizer,
    prompt: Option<&str>,
    prompt_tokens: Option<&[usize]>,
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
    mut on_text: Option<&mut dyn FnMut(&str)>,
) -> Result<String> {
    strategy.validate()?;

    if on_text.is_none() {
        if let Some(prompt) = prompt {
            if let Some(generated) = backend.generate_completion_on_device(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                strategy,
                stop_condition,
                rng,
                profile,
            )? {
                return Ok(generated);
            }
        } else if let Some(prompt_tokens) = prompt_tokens {
            if let Some(generated) = backend.generate_completion_from_tokens_on_device(
                model,
                tokenizer,
                prompt_tokens,
                max_new_tokens,
                strategy,
                stop_condition,
                rng,
                profile,
            )? {
                return Ok(generated);
            }
        }
    }

    let prompt_tokens = prompt_tokens
        .map(|tokens| tokens.to_vec())
        .unwrap_or_else(|| tokenizer.encode_text(prompt.expect("string prompt must exist")));
    let conditioning =
        conditioning_from_prompt_tokens(model, tokenizer, &prompt_tokens, max_new_tokens);

    let mut seen_token_counts = HashMap::new();
    for token_id in &conditioning {
        *seen_token_counts.entry(*token_id).or_insert(0_usize) += 1;
    }

    let mut kv_cache = new_kv_cache(model.cfg.n_layer, model.cfg.n_embd);
    let mut last_logits = None;
    for (pos_id, token_id) in conditioning.iter().copied().enumerate() {
        last_logits = Some(
            measure(profile, "sample.prefill", || {
                forward_token_profiled(
                    model,
                    token_id,
                    token_id,
                    pos_id,
                    1.0,
                    &mut kv_cache,
                    backend,
                    profile,
                )
            })?
            .logits,
        );
    }

    let mut pos_id = conditioning.len().saturating_sub(1);
    let mut logits = last_logits.ok_or_else(|| {
        RustGptError::Tensor("sampling failed to build an initial logits state".to_string())
    })?;
    let max_steps = max_new_tokens.min(model.cfg.block_size.saturating_sub(conditioning.len()));
    let mut generated_tokens = Vec::with_capacity(max_steps);
    let mut emitted_len = 0;

    for _ in 0..max_steps {
        let next_token = measure(profile, "sample.filter_logits", || {
            select_next_token(&logits, strategy, &seen_token_counts, rng)
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
        let visible = &generated_tokens[..visible_token_count];
        let generated = if on_text.is_some() {
            tokenizer.decode_streaming(visible)?
        } else {
            tokenizer.decode(visible, true)?
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

        pos_id += 1;
        if pos_id >= model.cfg.block_size {
            break;
        }
        logits = measure(profile, "sample.decode_step", || {
            forward_token_profiled(
                model,
                next_token,
                next_token,
                pos_id,
                1.0,
                &mut kv_cache,
                backend,
                profile,
            )
        })?
        .logits;
    }

    let generated = tokenizer.decode(&generated_tokens, true)?;
    if let Some(callback) = &mut on_text {
        if generated.len() > emitted_len {
            callback(&generated[emitted_len..]);
        }
    }
    Ok(generated)
}

fn conditioning_from_prompt_tokens(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
) -> Vec<usize> {
    let reserve = max_new_tokens.min(model.cfg.block_size.saturating_sub(1));
    let max_prompt_tokens = usize::max(1, model.cfg.block_size.saturating_sub(reserve));
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::core::config::{ActivationKind, BoundaryMode, ModelConfig, PositionEncodingKind};
    use crate::core::rng::Rng;
    use crate::data::tokenizer::Tokenizer;
    use crate::model::Model;

    use super::{
        SamplingStrategy, StopCondition, generate_completion,
        generate_completion_from_tokens_with_backend, generate_sample, select_next_token,
    };

    #[test]
    fn sample_generation_returns_prompt_prefixed_text() {
        let docs = vec!["emma".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: 16,
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
        let mut sample_rng = Rng::from_seed(7);
        let sample = generate_sample(&model, &tokenizer, "em", 8, 0.8, &mut sample_rng).unwrap();
        assert!(sample.starts_with("em"));
    }

    #[test]
    fn completion_respects_stop_sequence() {
        let docs = vec!["aaaa".to_string()];
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
        let mut sample_rng = Rng::from_seed(7);
        let completion =
            generate_completion(&model, &tokenizer, "a", 6, 0.8, &["aa"], &mut sample_rng).unwrap();
        assert!(!completion.ends_with("aa"));
    }

    #[test]
    fn repetition_penalty_can_change_choice() {
        let logits = vec![3.0, 2.0, 1.0];
        let strategy = SamplingStrategy {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 2.5,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        };
        let mut rng = Rng::from_seed(7);
        let mut seen = HashMap::new();
        seen.insert(0, 1);
        let next = select_next_token(&logits, &strategy, &seen, &mut rng).unwrap();
        assert_ne!(next, 0);
    }

    #[test]
    fn token_prompt_path_matches_string_prompt_path() {
        let docs = vec!["emma".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: 16,
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
        let backend = crate::runtime::backend::ComputeBackend::cpu();
        let strategy = SamplingStrategy::temperature_only(0.8);
        let prompt = "em";
        let prompt_tokens = tokenizer.encode_text(prompt);

        let mut string_rng = Rng::from_seed(7);
        let string_completion =
            generate_completion(&model, &tokenizer, prompt, 8, 0.8, &[], &mut string_rng).unwrap();

        let mut token_rng = Rng::from_seed(7);
        let token_completion = generate_completion_from_tokens_with_backend(
            &model,
            &backend,
            &tokenizer,
            &prompt_tokens,
            8,
            &strategy,
            &StopCondition::none(),
            None,
            &mut token_rng,
        )
        .unwrap();

        assert_eq!(token_completion, string_completion);
    }
}
