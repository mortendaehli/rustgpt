use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::core::tensor::softmax;
use crate::data::tokenizer::{TokenSymbol, Tokenizer};
use crate::model::Model;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::forward::forward_token_profiled;
use crate::runtime::profile::{RuntimeProfile, measure};
use crate::runtime::workspace::new_kv_cache;

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
        temperature,
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
        temperature,
        stop_sequences,
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
    temperature: f32,
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
) -> Result<String> {
    let completion = generate_completion_with_backend(
        model,
        backend,
        tokenizer,
        prompt,
        max_new_tokens,
        temperature,
        &[],
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
    temperature: f32,
    stop_sequences: &[&str],
    profile: Option<&RuntimeProfile>,
    rng: &mut Rng,
) -> Result<String> {
    if temperature <= 0.0 {
        return Err(RustGptError::Config("temperature must be > 0".to_string()));
    }

    if let Some(generated) = backend.generate_completion_on_device(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        temperature,
        stop_sequences,
        rng,
        profile,
    )? {
        return Ok(generated);
    }

    let mut conditioning = Vec::with_capacity(prompt.len() + 1);
    conditioning.push(tokenizer.bos_id());
    conditioning.extend(tokenizer.encode_text(prompt));

    let reserve = max_new_tokens.min(model.cfg.block_size.saturating_sub(1));
    let max_prompt_tokens = usize::max(1, model.cfg.block_size.saturating_sub(reserve));
    if conditioning.len() > max_prompt_tokens {
        conditioning = conditioning[conditioning.len() - max_prompt_tokens..].to_vec();
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

    for _ in 0..max_steps {
        let scaled = measure(profile, "sample.scale_logits", || {
            logits
                .iter()
                .map(|value| value / temperature)
                .collect::<Vec<_>>()
        });
        let probs = measure(profile, "sample.softmax", || softmax(&scaled));
        let next_token = rng.sample_weighted(&probs).ok_or_else(|| {
            RustGptError::Tensor("sampling failed because probabilities were invalid".to_string())
        })?;
        match tokenizer.symbol(next_token)? {
            TokenSymbol::Byte(_) => generated_tokens.push(next_token),
            TokenSymbol::Bos | TokenSymbol::Eos => break,
        }

        let generated = tokenizer.decode(&generated_tokens, true)?;
        if let Some(stop) = stop_sequences
            .iter()
            .find(|stop| generated.ends_with(**stop))
            .copied()
        {
            let new_len = generated.len().saturating_sub(stop.len());
            return Ok(generated[..new_len].to_string());
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
                &mut kv_cache,
                backend,
                profile,
            )
        })?
        .logits;
    }

    tokenizer.decode(&generated_tokens, true)
}

#[cfg(test)]
mod tests {
    use crate::core::config::{BoundaryMode, ModelConfig};
    use crate::core::rng::Rng;
    use crate::data::tokenizer::Tokenizer;
    use crate::model::Model;

    use super::{generate_completion, generate_sample};

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
            boundary_mode: BoundaryMode::SharedBos,
        };
        let mut init_rng = Rng::from_seed(42);
        let model = Model::new(cfg, &mut init_rng).unwrap();
        let mut sample_rng = Rng::from_seed(7);
        let completion =
            generate_completion(&model, &tokenizer, "a", 6, 0.8, &["aa"], &mut sample_rng).unwrap();
        assert!(!completion.ends_with("aa"));
    }
}
