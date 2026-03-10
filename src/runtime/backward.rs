use crate::core::error::Result;
use crate::core::tensor::{add_in_place, rmsnorm_backward, softmax_backward};
use crate::model::Model;
use crate::runtime::backend::ComputeBackend;
use crate::runtime::profile::{RuntimeProfile, measure};
use crate::runtime::train_cache::SequenceForwardCache;
use crate::runtime::workspace::KvGrad;

pub fn train_step(
    model: &mut Model,
    sequence: &SequenceForwardCache,
    lr_t: f32,
    beta1: f32,
    beta2: f32,
    eps_adam: f32,
    optimizer_step_num: usize,
) -> Result<f32> {
    let mut backend = ComputeBackend::cpu();
    train_step_with_backend(
        model,
        sequence,
        &mut backend,
        lr_t,
        beta1,
        beta2,
        eps_adam,
        optimizer_step_num,
    )
}

pub fn train_step_with_backend(
    model: &mut Model,
    sequence: &SequenceForwardCache,
    backend: &mut ComputeBackend,
    lr_t: f32,
    beta1: f32,
    beta2: f32,
    eps_adam: f32,
    optimizer_step_num: usize,
) -> Result<f32> {
    train_step_profiled(
        model,
        sequence,
        backend,
        lr_t,
        beta1,
        beta2,
        eps_adam,
        optimizer_step_num,
        None,
    )
}

pub fn train_step_profiled(
    model: &mut Model,
    sequence: &SequenceForwardCache,
    backend: &mut ComputeBackend,
    lr_t: f32,
    beta1: f32,
    beta2: f32,
    eps_adam: f32,
    optimizer_step_num: usize,
    profile: Option<&RuntimeProfile>,
) -> Result<f32> {
    begin_gradient_accumulation(model, backend);
    accumulate_sequence_gradients_profiled(model, sequence, backend, profile)?;
    apply_optimizer_profiled(
        model,
        backend,
        lr_t,
        beta1,
        beta2,
        eps_adam,
        optimizer_step_num,
        profile,
    )?;
    Ok(sequence.mean_loss)
}

pub fn begin_gradient_accumulation(model: &mut Model, backend: &ComputeBackend) {
    if !backend.uses_device_optimizer() {
        model.zero_grads();
    }
}

pub fn accumulate_sequence_gradients_profiled(
    model: &mut Model,
    sequence: &SequenceForwardCache,
    backend: &ComputeBackend,
    profile: Option<&RuntimeProfile>,
) -> Result<()> {
    backward_sequence_profiled(model, sequence, backend, profile)
}

pub fn apply_optimizer_profiled(
    model: &mut Model,
    backend: &mut ComputeBackend,
    lr_t: f32,
    beta1: f32,
    beta2: f32,
    eps_adam: f32,
    optimizer_step_num: usize,
    profile: Option<&RuntimeProfile>,
) -> Result<()> {
    measure(profile, "optimizer.adam", || {
        backend.adam_step(model, lr_t, beta1, beta2, eps_adam, optimizer_step_num)
    })
}

pub fn backward_sequence(model: &mut Model, sequence: &SequenceForwardCache) -> Result<()> {
    let backend = ComputeBackend::cpu();
    backward_sequence_with_backend(model, sequence, &backend)
}

pub fn backward_sequence_with_backend(
    model: &mut Model,
    sequence: &SequenceForwardCache,
    backend: &ComputeBackend,
) -> Result<()> {
    backward_sequence_profiled(model, sequence, backend, None)
}

pub fn backward_sequence_profiled(
    model: &mut Model,
    sequence: &SequenceForwardCache,
    backend: &ComputeBackend,
    profile: Option<&RuntimeProfile>,
) -> Result<()> {
    if backend.backward_training_sequence(model, sequence, profile)? {
        return Ok(());
    }

    let seq_len = sequence.tokens.len();
    let norm = sequence.grad_scale / seq_len as f32;
    let mut kv_grad = KvGrad::new(model.cfg.n_layer, seq_len, model.cfg.n_embd);

    for pos in (0..seq_len).rev() {
        backward_token(model, sequence, pos, norm, backend, &mut kv_grad, profile)?;
    }

    Ok(())
}

fn backward_token(
    model: &mut Model,
    sequence: &SequenceForwardCache,
    pos: usize,
    norm: f32,
    backend: &ComputeBackend,
    kv_grad: &mut KvGrad,
    profile: Option<&RuntimeProfile>,
) -> Result<()> {
    let token_cache = &sequence.tokens[pos];
    let head_dim = model.head_dim();
    let n_head = model.cfg.n_head;

    let mut d_logits = token_cache.probs.clone();
    d_logits[token_cache.target_id] -= 1.0;
    for value in &mut d_logits {
        *value *= norm;
    }

    let mut dx = measure(profile, "backward.matvec_t", || {
        backend.matvec_transposed(&d_logits, &model.lm_head)
    })?;
    measure(profile, "backward.grad_accum", || {
        backend.accumulate_linear_grad(&token_cache.final_x, &d_logits, &mut model.lm_head)
    })?;

    for layer_idx in (0..model.cfg.n_layer).rev() {
        let layer_cache = &token_cache.layers[layer_idx];

        let d_mlp_out = dx.clone();
        let mut dx_after_attn = dx.clone();

        let d_x_from_norm_mlp = {
            let layer = &mut model.layers[layer_idx];
            measure(profile, "backward.mlp_block", || {
                backend.backward_mlp_residual(
                    &layer_cache.x_residual_mlp,
                    &layer_cache.x_norm_mlp,
                    layer_cache.rms_inv_mlp,
                    &layer_cache.mlp_hidden_pre,
                    &layer_cache.mlp_hidden_act,
                    &d_mlp_out,
                    &mut layer.mlp_fc1,
                    &mut layer.mlp_fc2,
                )
            })?
        };
        add_in_place(&mut dx_after_attn, &d_x_from_norm_mlp);
        dx = dx_after_attn;

        let d_attn_out = dx.clone();
        let mut dx_before_attn = dx.clone();
        let d_x_attn = measure(profile, "backward.matvec_t", || {
            backend.matvec_transposed(&d_attn_out, &model.layers[layer_idx].attn_wo)
        })?;
        measure(profile, "backward.grad_accum", || {
            backend.accumulate_linear_grad(
                &layer_cache.x_attn,
                &d_attn_out,
                &mut model.layers[layer_idx].attn_wo,
            )
        })?;

        let (d_q, d_k_current, d_v_current) = measure(profile, "backward.attention", || {
            let mut d_q = vec![0.0; model.cfg.n_embd];
            let mut d_k_current = kv_grad.layers[layer_idx].key(pos).to_vec();
            let mut d_v_current = kv_grad.layers[layer_idx].value(pos).to_vec();
            let scale = (head_dim as f32).sqrt();

            for head_idx in 0..n_head {
                let start = head_idx * head_dim;
                let end = start + head_dim;
                let d_head = &d_x_attn[start..end];
                let q_slice = &layer_cache.q[start..end];
                let head_weights = layer_cache.attn_weights.head(head_idx);

                let mut d_weights = vec![0.0; pos + 1];
                for t in 0..=pos {
                    let v_slice = &sequence.tokens[t].layers[layer_idx].v[start..end];
                    d_weights[t] = d_head.iter().zip(v_slice).map(|(a, b)| a * b).sum();

                    let target_dv = if t == pos {
                        &mut d_v_current[start..end]
                    } else {
                        &mut kv_grad.layers[layer_idx].value_mut(t)[start..end]
                    };
                    for feature_idx in 0..head_dim {
                        target_dv[feature_idx] += d_head[feature_idx] * head_weights[t];
                    }
                }

                let d_logits_attn = softmax_backward(head_weights, &d_weights);
                for t in 0..=pos {
                    let k_slice = &sequence.tokens[t].layers[layer_idx].k[start..end];
                    let d_logit = d_logits_attn[t];
                    for feature_idx in 0..head_dim {
                        d_q[start + feature_idx] += d_logit * k_slice[feature_idx] / scale;
                    }

                    let target_dk = if t == pos {
                        &mut d_k_current[start..end]
                    } else {
                        &mut kv_grad.layers[layer_idx].key_mut(t)[start..end]
                    };
                    for feature_idx in 0..head_dim {
                        target_dk[feature_idx] += d_logit * q_slice[feature_idx] / scale;
                    }
                }
            }

            (d_q, d_k_current, d_v_current)
        });

        let d_x_from_norm_attn = {
            let layer = &mut model.layers[layer_idx];
            measure(profile, "backward.attn_proj_block", || {
                backend.backward_attention_projections(
                    &layer_cache.x_residual_attn,
                    &layer_cache.x_norm_attn,
                    layer_cache.rms_inv_attn,
                    &d_q,
                    &d_k_current,
                    &d_v_current,
                    &mut layer.attn_wq,
                    &mut layer.attn_wk,
                    &mut layer.attn_wv,
                )
            })?
        };
        add_in_place(&mut dx_before_attn, &d_x_from_norm_attn);
        dx = dx_before_attn;
    }

    let d_embed_sum = measure(profile, "backward.rmsnorm", || {
        rmsnorm_backward(&token_cache.embed_sum, token_cache.embed_rms_inv, &dx)
    });
    measure(profile, "backward.row_grad", || {
        backend.add_row_grad(token_cache.token_id, &d_embed_sum, &mut model.wte)
    })?;
    measure(profile, "backward.row_grad", || {
        backend.add_row_grad(token_cache.pos_id, &d_embed_sum, &mut model.wpe)
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::core::config::{BoundaryMode, DeviceKind, ModelConfig};
    use crate::core::rng::Rng;
    use crate::data::tokenizer::Tokenizer;
    use crate::runtime::backend::ComputeBackend;
    use crate::runtime::forward::{forward_sequence, forward_sequence_with_backend};

    use super::{train_step, train_step_with_backend};
    use crate::model::Model;

    #[test]
    fn repeated_training_reduces_loss_on_single_document() {
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
        let mut rng = Rng::from_seed(42);
        let mut model = Model::new(cfg, &mut rng).unwrap();
        let tokens = tokenizer.encode_with_boundaries("emma").unwrap();

        let initial = forward_sequence(&model, &tokens).unwrap().mean_loss;
        let mut final_loss = initial;
        for step in 0..32 {
            let sequence = forward_sequence(&model, &tokens).unwrap();
            let lr_t = 0.01 * (1.0 - step as f32 / 32.0);
            final_loss =
                train_step(&mut model, &sequence, lr_t, 0.85, 0.99, 1e-8, step + 1).unwrap();
        }

        assert!(
            final_loss < initial,
            "final_loss={final_loss} initial={initial}"
        );
    }

    #[test]
    fn auto_backend_training_reduces_loss_on_single_document() {
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
        let mut rng = Rng::from_seed(42);
        let mut model = Model::new(cfg, &mut rng).unwrap();
        let mut backend = ComputeBackend::from_model(&model, DeviceKind::Auto).unwrap();
        let tokens = tokenizer.encode_with_boundaries("emma").unwrap();

        backend.sync_model(&model).unwrap();
        let initial = forward_sequence_with_backend(&model, &tokens, &backend)
            .unwrap()
            .mean_loss;
        let mut final_loss = initial;
        for step in 0..8 {
            backend.sync_model(&model).unwrap();
            let sequence = forward_sequence_with_backend(&model, &tokens, &backend).unwrap();
            let lr_t = 0.01 * (1.0 - step as f32 / 8.0);
            final_loss = train_step_with_backend(
                &mut model,
                &sequence,
                &mut backend,
                lr_t,
                0.85,
                0.99,
                1e-8,
                step + 1,
            )
            .unwrap();
        }

        assert!(
            final_loss < initial,
            "final_loss={final_loss} initial={initial}"
        );
    }
}
