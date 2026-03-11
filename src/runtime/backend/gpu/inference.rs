use std::collections::HashMap;

use crate::core::config::{ActivationKind, PositionEncodingKind};
use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::data::tokenizer::Tokenizer;
use crate::model::{Model, ParameterId};
use crate::runtime::profile::{RuntimeProfile, measure};
use crate::runtime::sampling::{SamplingStrategy, StopCondition, select_next_token};

use super::{GpuLayerCache, GpuMatrix, GpuRuntime, lookup_parameter};

pub(super) fn run_gpu_forward_token(
    runtime: &GpuRuntime,
    matrices: &HashMap<ParameterId, GpuMatrix>,
    model: &Model,
    layer_caches: &mut [GpuLayerCache],
    token_id: usize,
    pos_id: usize,
    profile: Option<&RuntimeProfile>,
) -> Result<Vec<f32>> {
    if token_id >= model.cfg.vocab_size {
        return Err(RustGptError::Gpu(format!(
            "token_id {token_id} out of range for vocab size {}",
            model.cfg.vocab_size
        )));
    }
    if pos_id >= model.cfg.block_size {
        return Err(RustGptError::Gpu(format!(
            "pos_id {pos_id} out of range for block size {}",
            model.cfg.block_size
        )));
    }
    if layer_caches.len() != model.cfg.n_layer {
        return Err(RustGptError::Gpu(format!(
            "layer cache mismatch: got {}, expected {}",
            layer_caches.len(),
            model.cfg.n_layer
        )));
    }

    let wte = lookup_parameter(matrices, ParameterId::Wte)?;
    let token_embed = measure(profile, "forward.gather", || {
        runtime.gather_row(wte, token_id)
    })?;
    let embed_sum = match model.cfg.position_encoding {
        PositionEncodingKind::LearnedAbsolute => {
            let wpe = lookup_parameter(matrices, ParameterId::Wpe)?;
            let pos_embed = measure(profile, "forward.gather", || {
                runtime.gather_row(wpe, pos_id)
            })?;
            measure(profile, "forward.add", || {
                runtime.add(&token_embed, &pos_embed)
            })?
        }
        PositionEncodingKind::Rope => token_embed,
    };
    let mut x = measure(profile, "forward.rmsnorm", || runtime.rmsnorm(&embed_sum))?;

    for layer_idx in 0..model.cfg.n_layer {
        let attn_wq = lookup_parameter(matrices, ParameterId::AttnWq(layer_idx))?;
        let attn_wk = lookup_parameter(matrices, ParameterId::AttnWk(layer_idx))?;
        let attn_wv = lookup_parameter(matrices, ParameterId::AttnWv(layer_idx))?;
        let attn_wo = lookup_parameter(matrices, ParameterId::AttnWo(layer_idx))?;
        let mlp_fc1 = lookup_parameter(matrices, ParameterId::MlpFc1(layer_idx))?;
        let mlp_fc2 = lookup_parameter(matrices, ParameterId::MlpFc2(layer_idx))?;

        let x_residual_attn = measure(profile, "forward.copy", || {
            runtime.copy_vector(&x, "rustgpt-attn-residual")
        })?;
        let x_norm_attn = measure(profile, "forward.rmsnorm", || runtime.rmsnorm(&x))?;
        let q = measure(profile, "forward.matvec", || {
            runtime.matvec_vector(&x_norm_attn, attn_wq)
        })?;
        let k_compact = measure(profile, "forward.matvec", || {
            runtime.matvec_vector(&x_norm_attn, attn_wk)
        })?;
        let q = match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => q,
            PositionEncodingKind::Rope => measure(profile, "forward.position", || {
                runtime.rope(&q, pos_id, model.head_dim())
            })?,
        };
        let k_compact = match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => k_compact,
            PositionEncodingKind::Rope => measure(profile, "forward.position", || {
                runtime.rope(&k_compact, pos_id, model.head_dim())
            })?,
        };
        let v_compact = measure(profile, "forward.matvec", || {
            runtime.matvec_vector(&x_norm_attn, attn_wv)
        })?;
        let k = measure(profile, "forward.attention_expand", || {
            runtime.expand_grouped_heads_rows(
                &k_compact,
                1,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let v = measure(profile, "forward.attention_expand", || {
            runtime.expand_grouped_heads_rows(
                &v_compact,
                1,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;

        let cache = &mut layer_caches[layer_idx];
        measure(profile, "forward.kv_append", || {
            runtime.append_to_cache(cache, &k, &v)
        })?;

        let scores = measure(profile, "forward.attention_scores", || {
            runtime.attn_scores(
                &q,
                &cache.keys,
                cache.len,
                model.cfg.n_head,
                model.head_dim(),
            )
        })?;
        let weights = measure(profile, "forward.softmax", || {
            runtime.softmax_rows(&scores, model.cfg.n_head, cache.len)
        })?;
        let x_attn = measure(profile, "forward.attention_values", || {
            runtime.attn_values(
                &weights,
                &cache.values,
                cache.len,
                model.cfg.n_head,
                model.head_dim(),
            )
        })?;
        let attn_out = measure(profile, "forward.matvec", || {
            runtime.matvec_vector(&x_attn, attn_wo)
        })?;
        x = measure(profile, "forward.add", || {
            runtime.add(&attn_out, &x_residual_attn)
        })?;

        let x_residual_mlp = measure(profile, "forward.copy", || {
            runtime.copy_vector(&x, "rustgpt-mlp-residual")
        })?;
        let x_norm_mlp = measure(profile, "forward.rmsnorm", || runtime.rmsnorm(&x))?;
        let hidden_pre = measure(profile, "forward.matvec", || {
            runtime.matvec_vector(&x_norm_mlp, mlp_fc1)
        })?;
        let gate_pre = if model.cfg.activation == ActivationKind::SwiGlu {
            let mlp_fc_gate = lookup_parameter(matrices, ParameterId::MlpFcGate(layer_idx))?;
            Some(measure(profile, "forward.matvec", || {
                runtime.matvec_vector(&x_norm_mlp, mlp_fc_gate)
            })?)
        } else {
            None
        };
        let hidden_act = measure(profile, "forward.activation", || {
            match model.cfg.activation {
                ActivationKind::Relu => runtime.relu(&hidden_pre),
                ActivationKind::Gelu => runtime.gelu(&hidden_pre),
                ActivationKind::SwiGlu => {
                    let gate_act = runtime.silu(
                        gate_pre
                            .as_ref()
                            .expect("swiglu activation requires gate pre-activations"),
                    )?;
                    runtime.mul(&hidden_pre, &gate_act)
                }
            }
        })?;
        let mlp_out = measure(profile, "forward.matvec", || {
            runtime.matvec_vector(&hidden_act, mlp_fc2)
        })?;
        x = measure(profile, "forward.add", || {
            runtime.add(&mlp_out, &x_residual_mlp)
        })?;
    }

    let x = measure(profile, "forward.rmsnorm", || runtime.rmsnorm(&x))?;
    let lm_head = lookup_parameter(matrices, model.output_parameter_id())?;
    let logits = measure(profile, "forward.matvec", || {
        runtime.matvec_vector(&x, lm_head)
    })?;
    measure(profile, "sample.readback_logits", || {
        runtime.readback_vector(&logits)
    })
}

pub(super) fn run_gpu_completion(
    runtime: &GpuRuntime,
    matrices: &HashMap<ParameterId, GpuMatrix>,
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    rng: &mut Rng,
    profile: Option<&RuntimeProfile>,
) -> Result<String> {
    strategy.validate()?;

    let prompt_tokens = tokenizer.encode_text(prompt);
    run_gpu_completion_from_prompt_tokens(
        runtime,
        matrices,
        model,
        tokenizer,
        &prompt_tokens,
        max_new_tokens,
        strategy,
        stop_condition,
        rng,
        profile,
    )
}

pub(super) fn run_gpu_completion_from_tokens(
    runtime: &GpuRuntime,
    matrices: &HashMap<ParameterId, GpuMatrix>,
    model: &Model,
    tokenizer: &Tokenizer,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    rng: &mut Rng,
    profile: Option<&RuntimeProfile>,
) -> Result<String> {
    strategy.validate()?;
    run_gpu_completion_from_prompt_tokens(
        runtime,
        matrices,
        model,
        tokenizer,
        prompt_tokens,
        max_new_tokens,
        strategy,
        stop_condition,
        rng,
        profile,
    )
}

fn run_gpu_completion_from_prompt_tokens(
    runtime: &GpuRuntime,
    matrices: &HashMap<ParameterId, GpuMatrix>,
    model: &Model,
    tokenizer: &Tokenizer,
    prompt_tokens: &[usize],
    max_new_tokens: usize,
    strategy: &SamplingStrategy,
    stop_condition: &StopCondition,
    rng: &mut Rng,
    profile: Option<&RuntimeProfile>,
) -> Result<String> {
    let mut conditioning = Vec::with_capacity(prompt_tokens.len() + 1);
    conditioning.push(tokenizer.bos_id());
    conditioning.extend_from_slice(prompt_tokens);

    let reserve = max_new_tokens.min(model.cfg.block_size.saturating_sub(1));
    let max_prompt_tokens = usize::max(1, model.cfg.block_size.saturating_sub(reserve));
    if conditioning.len() > max_prompt_tokens {
        conditioning = conditioning[conditioning.len() - max_prompt_tokens..].to_vec();
    }
    let mut seen_token_counts = HashMap::new();
    for token_id in &conditioning {
        *seen_token_counts.entry(*token_id).or_insert(0_usize) += 1;
    }

    let mut layer_caches = (0..model.cfg.n_layer)
        .map(|layer_idx| runtime.create_kv_cache(model.cfg.block_size, model.cfg.n_embd, layer_idx))
        .collect::<Result<Vec<_>>>()?;

    let mut last_logits = None;
    for (pos_id, token_id) in conditioning.iter().copied().enumerate() {
        last_logits = Some(measure(profile, "sample.prefill", || {
            run_gpu_forward_token(
                runtime,
                matrices,
                model,
                &mut layer_caches,
                token_id,
                pos_id,
                profile,
            )
        })?);
    }

    let mut pos_id = conditioning.len().saturating_sub(1);
    let mut logits = last_logits.ok_or_else(|| {
        RustGptError::Tensor("sampling failed to build an initial logits state".to_string())
    })?;
    let max_steps = max_new_tokens.min(model.cfg.block_size.saturating_sub(conditioning.len()));
    let mut generated_tokens = Vec::with_capacity(max_steps);

    for _ in 0..max_steps {
        let next_token = measure(profile, "sample.filter_logits", || {
            select_next_token(&logits, strategy, &seen_token_counts, rng)
        })?;

        if tokenizer.is_end_token(next_token) {
            break;
        }
        generated_tokens.push(next_token);
        *seen_token_counts.entry(next_token).or_insert(0_usize) += 1;

        if let Some(stop_len) = stop_condition.matched_suffix_len(&generated_tokens) {
            return tokenizer.decode(
                &generated_tokens[..generated_tokens.len().saturating_sub(stop_len)],
                true,
            );
        }

        pos_id += 1;
        if pos_id >= model.cfg.block_size {
            break;
        }
        logits = measure(profile, "sample.decode_step", || {
            run_gpu_forward_token(
                runtime,
                matrices,
                model,
                &mut layer_caches,
                next_token,
                pos_id,
                profile,
            )
        })?;
    }

    tokenizer.decode(&generated_tokens, true)
}
