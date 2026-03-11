use crate::core::config::{ActivationKind, PositionEncodingKind};
use crate::core::error::Result;
use crate::core::tensor::swiglu;
use crate::data::training_data::SequenceExample;
use crate::model::Model;
use crate::model::attention::expand_grouped_heads;
use crate::model::position::apply_qk_position_encoding_in_place;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::profile::RuntimeProfile;
use crate::runtime::train_cache::{LayerForwardCache, SequenceForwardCache, TokenForwardCache};
use crate::runtime::workspace::{KvCache, new_kv_cache};

pub fn forward_sequence(model: &Model, tokens: &[usize]) -> Result<SequenceForwardCache> {
    let backend = ComputeBackend::cpu();
    forward_sequence_with_backend(model, tokens, &backend)
}

pub fn forward_sequence_with_backend(
    model: &Model,
    tokens: &[usize],
    backend: &ComputeBackend,
) -> Result<SequenceForwardCache> {
    forward_sequence_profiled(model, tokens, backend, None)
}

pub fn forward_sequence_profiled(
    model: &Model,
    tokens: &[usize],
    backend: &ComputeBackend,
    profile: Option<&RuntimeProfile>,
) -> Result<SequenceForwardCache> {
    let example = full_loss_example(tokens)?;
    forward_example_profiled_with_loss(model, &example, backend, profile, true, 1.0)
}

pub fn forward_example_profiled_with_loss(
    model: &Model,
    example: &SequenceExample,
    backend: &ComputeBackend,
    profile: Option<&RuntimeProfile>,
    capture_loss: bool,
    grad_scale: f32,
) -> Result<SequenceForwardCache> {
    if let Some(cache) =
        backend.forward_training_sequence(model, example, capture_loss, grad_scale, profile)?
    {
        return Ok(cache);
    }

    if example.input_ids.is_empty() {
        return Err(crate::core::error::RustGptError::Data(
            "need at least two tokens to compute a next-token loss".to_string(),
        ));
    }

    let n_pred = usize::min(model.cfg.block_size, example.input_ids.len());
    let mut token_caches = Vec::with_capacity(n_pred);
    let mut total_loss = 0.0;
    let mut loss_weight_sum = 0.0;
    let mut kv_cache = new_kv_cache(model.cfg.n_layer, model.cfg.n_embd);

    for pos_id in 0..n_pred {
        let token_id = example.input_ids[pos_id];
        let target_id = example.target_ids[pos_id];
        let loss_weight = example.loss_mask[pos_id];
        let cache = forward_token_with_existing_cache(
            model,
            token_id,
            target_id,
            pos_id,
            loss_weight,
            &mut kv_cache,
            backend,
            profile,
        )?;
        if capture_loss && loss_weight > 0.0 {
            total_loss += -cache.probs[target_id].ln() * loss_weight;
            loss_weight_sum += loss_weight;
        }
        token_caches.push(cache);
    }

    Ok(SequenceForwardCache {
        tokens: token_caches,
        mean_loss: if capture_loss && loss_weight_sum > 0.0 {
            total_loss / loss_weight_sum
        } else {
            0.0
        },
        loss_weight_sum,
        grad_scale,
        device_sequence: None,
    })
}

pub fn average_sequence_loss(model: &Model, tokens: &[usize]) -> Result<f32> {
    Ok(forward_sequence(model, tokens)?.mean_loss)
}

pub fn forward_token(
    model: &Model,
    token_id: usize,
    target_id: usize,
    pos_id: usize,
    kv_cache: &mut KvCache,
) -> Result<TokenForwardCache> {
    let backend = ComputeBackend::cpu();
    forward_token_with_backend(model, token_id, target_id, pos_id, 1.0, kv_cache, &backend)
}

pub fn forward_token_with_backend(
    model: &Model,
    token_id: usize,
    target_id: usize,
    pos_id: usize,
    loss_weight: f32,
    kv_cache: &mut KvCache,
    backend: &ComputeBackend,
) -> Result<TokenForwardCache> {
    forward_token_profiled(
        model,
        token_id,
        target_id,
        pos_id,
        loss_weight,
        kv_cache,
        backend,
        None,
    )
}

pub fn forward_token_profiled(
    model: &Model,
    token_id: usize,
    target_id: usize,
    pos_id: usize,
    loss_weight: f32,
    kv_cache: &mut KvCache,
    backend: &ComputeBackend,
    profile: Option<&RuntimeProfile>,
) -> Result<TokenForwardCache> {
    forward_token_with_existing_cache(
        model,
        token_id,
        target_id,
        pos_id,
        loss_weight,
        kv_cache,
        backend,
        profile,
    )
}

fn forward_token_with_existing_cache(
    model: &Model,
    token_id: usize,
    target_id: usize,
    pos_id: usize,
    loss_weight: f32,
    kv_cache: &mut KvCache,
    backend: &ComputeBackend,
    profile: Option<&RuntimeProfile>,
) -> Result<TokenForwardCache> {
    use crate::core::error::RustGptError;
    use crate::core::tensor::{add_in_place, gelu, relu, rmsnorm, softmax};
    use crate::runtime::profile::measure;
    use crate::runtime::workspace::AttentionWeights;

    if token_id >= model.cfg.vocab_size {
        return Err(RustGptError::Tensor(format!(
            "token_id {token_id} out of range for vocab size {}",
            model.cfg.vocab_size
        )));
    }
    if target_id >= model.cfg.vocab_size {
        return Err(RustGptError::Tensor(format!(
            "target_id {target_id} out of range for vocab size {}",
            model.cfg.vocab_size
        )));
    }
    if pos_id >= model.cfg.block_size {
        return Err(RustGptError::Tensor(format!(
            "pos_id {pos_id} out of range for block size {}",
            model.cfg.block_size
        )));
    }

    let embed_sum = measure(profile, "forward.embed_add", || {
        match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => {
                let pos_row = model
                    .position_embedding()
                    .expect("learned position encoding requires wpe")
                    .row(pos_id);
                model
                    .wte
                    .row(token_id)
                    .iter()
                    .zip(pos_row)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>()
            }
            PositionEncodingKind::Rope => model.wte.row(token_id).to_vec(),
        }
    });
    let (mut x, embed_rms_inv) = measure(profile, "forward.rmsnorm", || rmsnorm(&embed_sum));
    let mut layer_caches = Vec::with_capacity(model.cfg.n_layer);

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let x_residual_attn = x.clone();
        let (x_norm_attn, rms_inv_attn) = measure(profile, "forward.rmsnorm", || rmsnorm(&x));

        let mut q = measure(profile, "forward.matvec", || {
            backend.matvec(&x_norm_attn, &layer.attn_wq)
        })?;
        let mut k_compact = measure(profile, "forward.matvec", || {
            backend.matvec(&x_norm_attn, &layer.attn_wk)
        })?;
        let v_compact = measure(profile, "forward.matvec", || {
            backend.matvec(&x_norm_attn, &layer.attn_wv)
        })?;
        measure(profile, "forward.position", || {
            apply_qk_position_encoding_in_place(
                &mut q,
                model.cfg.position_encoding,
                pos_id,
                model.cfg.n_head,
                model.head_dim(),
            );
            apply_qk_position_encoding_in_place(
                &mut k_compact,
                model.cfg.position_encoding,
                pos_id,
                model.cfg.n_kv_head,
                model.head_dim(),
            );
        });
        let k = measure(profile, "forward.attention_expand", || {
            expand_grouped_heads(
                &k_compact,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let v = measure(profile, "forward.attention_expand", || {
            expand_grouped_heads(
                &v_compact,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        kv_cache[layer_idx].push(&k, &v)?;

        let (x_attn, attn_weights) = measure(profile, "forward.attention", || {
            let mut x_attn = vec![0.0; model.cfg.n_embd];
            let mut attn_weights =
                AttentionWeights::zeros(model.cfg.n_head, kv_cache[layer_idx].len());
            let scale = (model.head_dim() as f32).sqrt();
            for head_idx in 0..model.cfg.n_head {
                let start = head_idx * model.head_dim();
                let end = start + model.head_dim();
                let q_slice = &q[start..end];

                let mut attn_logits = Vec::with_capacity(kv_cache[layer_idx].len());
                for time_idx in 0..kv_cache[layer_idx].len() {
                    let past_k = kv_cache[layer_idx].key(time_idx);
                    attn_logits.push(
                        q_slice
                            .iter()
                            .zip(&past_k[start..end])
                            .map(|(left, right)| left * right)
                            .sum::<f32>()
                            / scale,
                    );
                }
                let head_weights = softmax(&attn_logits);
                attn_weights
                    .head_mut(head_idx)
                    .copy_from_slice(&head_weights);

                for feature_idx in 0..model.head_dim() {
                    x_attn[start + feature_idx] = (0..kv_cache[layer_idx].len())
                        .map(|time_idx| {
                            head_weights[time_idx]
                                * kv_cache[layer_idx].value(time_idx)[start + feature_idx]
                        })
                        .sum::<f32>();
                }
            }
            (x_attn, attn_weights)
        });

        x = measure(profile, "forward.matvec", || {
            backend.matvec(&x_attn, &layer.attn_wo)
        })?;
        add_in_place(&mut x, &x_residual_attn);

        let x_residual_mlp = x.clone();
        let (x_norm_mlp, rms_inv_mlp) = measure(profile, "forward.rmsnorm", || rmsnorm(&x));
        let mlp_hidden_pre = measure(profile, "forward.matvec", || {
            backend.matvec(&x_norm_mlp, &layer.mlp_fc1)
        })?;
        let mlp_gate_pre = if model.cfg.activation == ActivationKind::SwiGlu {
            let gate = layer
                .mlp_fc_gate
                .as_ref()
                .expect("swiglu activation requires mlp_fc_gate");
            Some(measure(profile, "forward.matvec", || {
                backend.matvec(&x_norm_mlp, gate)
            })?)
        } else {
            None
        };
        let mlp_hidden_act = measure(profile, "forward.activation", || {
            match model.cfg.activation {
                ActivationKind::Relu => relu(&mlp_hidden_pre),
                ActivationKind::Gelu => gelu(&mlp_hidden_pre),
                ActivationKind::SwiGlu => swiglu(
                    &mlp_hidden_pre,
                    mlp_gate_pre
                        .as_ref()
                        .expect("swiglu activation requires gate pre-activations"),
                )
                .expect("swiglu forward shapes must match"),
            }
        });
        x = measure(profile, "forward.matvec", || {
            backend.matvec(&mlp_hidden_act, &layer.mlp_fc2)
        })?;
        add_in_place(&mut x, &x_residual_mlp);

        layer_caches.push(LayerForwardCache {
            x_residual_attn,
            x_norm_attn,
            rms_inv_attn,
            q,
            k,
            v,
            attn_weights,
            x_attn,
            x_residual_mlp,
            x_norm_mlp,
            rms_inv_mlp,
            mlp_hidden_pre,
            mlp_gate_pre,
            mlp_hidden_act,
        });
    }

    let final_x = x.clone();
    let (final_norm_x, final_rms_inv) = measure(profile, "forward.rmsnorm", || rmsnorm(&x));
    let logits = measure(profile, "forward.matvec", || {
        backend.matvec(&final_norm_x, model.output_projection())
    })?;
    let probs = measure(profile, "forward.softmax", || softmax(&logits));

    Ok(TokenForwardCache {
        token_id,
        target_id,
        pos_id,
        loss_weight,
        embed_sum,
        embed_rms_inv,
        layers: layer_caches,
        final_x,
        final_norm_x,
        final_rms_inv,
        logits,
        probs,
    })
}

fn full_loss_example(tokens: &[usize]) -> Result<SequenceExample> {
    if tokens.len() < 2 {
        return Err(crate::core::error::RustGptError::Data(
            "need at least two tokens to compute a next-token loss".to_string(),
        ));
    }
    Ok(SequenceExample {
        input_ids: tokens[..tokens.len() - 1].to_vec(),
        target_ids: tokens[1..].to_vec(),
        loss_mask: vec![1.0; tokens.len() - 1],
    })
}

#[cfg(test)]
mod tests {
    use crate::core::config::{ActivationKind, BoundaryMode, ModelConfig, PositionEncodingKind};
    use crate::core::rng::Rng;
    use crate::data::tokenizer::Tokenizer;
    use crate::model::Model;

    use super::{
        average_sequence_loss, forward_sequence, forward_sequence_with_backend, forward_token,
    };

    #[test]
    fn forward_token_produces_vocab_sized_logits() {
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
        let mut rng = Rng::from_seed(42);
        let model = Model::new(cfg, &mut rng).unwrap();
        let mut kv_cache =
            crate::runtime::workspace::new_kv_cache(model.cfg.n_layer, model.cfg.n_embd);
        let token = forward_token(
            &model,
            tokenizer.bos_id(),
            tokenizer.bos_id(),
            0,
            &mut kv_cache,
        )
        .unwrap();
        assert_eq!(token.logits.len(), tokenizer.vocab_size());
        assert_eq!(token.probs.len(), tokenizer.vocab_size());
    }

    #[test]
    fn forward_sequence_returns_one_cache_per_prediction() {
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
        let mut rng = Rng::from_seed(42);
        let model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("emma").unwrap();
        let sequence = forward_sequence(&model, &tokens).unwrap();
        assert_eq!(sequence.tokens.len(), tokens.len() - 1);
    }

    #[test]
    fn average_sequence_loss_is_finite() {
        let docs = vec!["olivia".to_string()];
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
        let mut rng = Rng::from_seed(7);
        let model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("olivia").unwrap();
        let loss = average_sequence_loss(&model, &tokens).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn backend_forward_matches_cpu_forward() {
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
        let mut rng = Rng::from_seed(42);
        let model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("emma").unwrap();
        let cpu = forward_sequence(&model, &tokens).unwrap();
        let backend = crate::runtime::backend::ComputeBackend::cpu();
        let backend_forward = forward_sequence_with_backend(&model, &tokens, &backend).unwrap();
        assert!((cpu.mean_loss - backend_forward.mean_loss).abs() < 1e-6);
    }

    #[test]
    fn forward_sequence_supports_gelu_activation() {
        let docs = vec!["naomi".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 4,
            tie_embeddings: false,
            activation: ActivationKind::Gelu,
            position_encoding: PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
        };
        let mut rng = Rng::from_seed(21);
        let model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("naomi").unwrap();
        let sequence = forward_sequence(&model, &tokens).unwrap();
        assert!(sequence.mean_loss.is_finite());
        assert_eq!(sequence.tokens.len(), tokens.len() - 1);
    }

    #[test]
    fn forward_sequence_supports_swiglu_activation() {
        let docs = vec!["naomi".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 4,
            tie_embeddings: false,
            activation: ActivationKind::SwiGlu,
            position_encoding: PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
        };
        let mut rng = Rng::from_seed(23);
        let model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("naomi").unwrap();
        let sequence = forward_sequence(&model, &tokens).unwrap();
        assert!(sequence.mean_loss.is_finite());
    }

    #[test]
    fn forward_sequence_supports_grouped_query_attention() {
        let docs = vec!["naomi".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 2,
            tie_embeddings: false,
            activation: ActivationKind::Relu,
            position_encoding: PositionEncodingKind::Rope,
            boundary_mode: BoundaryMode::SharedBos,
        };
        let mut rng = Rng::from_seed(24);
        let model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("naomi").unwrap();
        let sequence = forward_sequence(&model, &tokens).unwrap();
        assert!(sequence.mean_loss.is_finite());
    }

    #[test]
    fn forward_sequence_supports_rope_positions() {
        let docs = vec!["naomi".to_string()];
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
            position_encoding: PositionEncodingKind::Rope,
            boundary_mode: BoundaryMode::SharedBos,
        };
        let mut rng = Rng::from_seed(22);
        let model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("naomi").unwrap();
        let sequence = forward_sequence(&model, &tokens).unwrap();
        assert!(sequence.mean_loss.is_finite());
        assert_eq!(sequence.tokens.len(), tokens.len() - 1);
    }
}
