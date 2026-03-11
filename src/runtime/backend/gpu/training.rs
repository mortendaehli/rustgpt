use std::collections::HashMap;

use crate::core::config::{ActivationKind, PositionEncodingKind};
use crate::core::error::{Result, RustGptError};
use crate::data::training_data::SequenceExample;
use crate::model::{Model, ParameterId};
use crate::runtime::profile::{RuntimeProfile, measure};
use crate::runtime::train_cache::{DeviceSequenceHandle, SequenceForwardCache, TokenForwardCache};

use super::{
    GpuMatrix, GpuRuntime, GpuTrainingLayerCache, GpuTrainingSequenceCache, lookup_parameter,
};

pub(super) fn run_gpu_training_sequence(
    runtime: &GpuRuntime,
    matrices: &HashMap<ParameterId, GpuMatrix>,
    model: &Model,
    example: &SequenceExample,
    capture_loss: bool,
    grad_scale: f32,
    profile: Option<&RuntimeProfile>,
) -> Result<SequenceForwardCache> {
    if example.input_ids.is_empty() {
        return Err(RustGptError::Data(
            "need at least two tokens to compute a next-token loss".to_string(),
        ));
    }

    let n_pred = usize::min(model.cfg.block_size, example.input_ids.len());
    let mut embed_sum_host = vec![0.0; n_pred * model.cfg.n_embd];
    for pos_id in 0..n_pred {
        let token_id = example.input_ids[pos_id];
        let target_id = example.target_ids[pos_id];
        if token_id >= model.cfg.vocab_size || target_id >= model.cfg.vocab_size {
            return Err(RustGptError::Gpu(
                "sequence token id out of range".to_string(),
            ));
        }
        let out_row =
            &mut embed_sum_host[pos_id * model.cfg.n_embd..(pos_id + 1) * model.cfg.n_embd];
        let token_row = model.wte.row(token_id);
        match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => {
                let pos_row = model
                    .position_embedding()
                    .expect("learned position encoding requires wpe")
                    .row(pos_id);
                for feature_idx in 0..model.cfg.n_embd {
                    out_row[feature_idx] = token_row[feature_idx] + pos_row[feature_idx];
                }
            }
            PositionEncodingKind::Rope => out_row.copy_from_slice(token_row),
        }
    }

    let embed_sum = measure(profile, "forward.upload_input", || {
        runtime.upload_vector(&embed_sum_host, "rustgpt-train-embed-sum")
    })?;
    let token_ids = measure(profile, "forward.upload_input", || {
        runtime.upload_u32_vector(
            &example.input_ids[..n_pred]
                .iter()
                .map(|token| *token as u32)
                .collect::<Vec<_>>(),
            "rustgpt-train-token-ids",
        )
    })?;
    let target_ids = measure(profile, "forward.upload_input", || {
        runtime.upload_u32_vector(
            &example.target_ids[..n_pred]
                .iter()
                .map(|token| *token as u32)
                .collect::<Vec<_>>(),
            "rustgpt-train-target-ids",
        )
    })?;
    let loss_weights = measure(profile, "forward.upload_input", || {
        runtime.upload_vector(&example.loss_mask[..n_pred], "rustgpt-train-loss-weights")
    })?;
    let loss_weight_sum = example.loss_mask[..n_pred].iter().sum::<f32>();
    let mut x = measure(profile, "forward.rmsnorm", || {
        runtime.rmsnorm_rows(&embed_sum, n_pred, model.cfg.n_embd)
    })?;
    let mut gpu_layers = Vec::with_capacity(model.cfg.n_layer);

    for layer_idx in 0..model.cfg.n_layer {
        let attn_wq = lookup_parameter(matrices, ParameterId::AttnWq(layer_idx))?;
        let attn_wk = lookup_parameter(matrices, ParameterId::AttnWk(layer_idx))?;
        let attn_wv = lookup_parameter(matrices, ParameterId::AttnWv(layer_idx))?;
        let attn_wo = lookup_parameter(matrices, ParameterId::AttnWo(layer_idx))?;
        let mlp_fc1 = lookup_parameter(matrices, ParameterId::MlpFc1(layer_idx))?;
        let mlp_fc_gate = if model.cfg.activation == ActivationKind::SwiGlu {
            Some(lookup_parameter(
                matrices,
                ParameterId::MlpFcGate(layer_idx),
            )?)
        } else {
            None
        };
        let mlp_fc2 = lookup_parameter(matrices, ParameterId::MlpFc2(layer_idx))?;

        let x_residual_attn = measure(profile, "forward.copy", || {
            runtime.copy_vector(&x, "rustgpt-train-seq-attn-residual")
        })?;
        let x_norm_attn = measure(profile, "forward.rmsnorm", || {
            runtime.rmsnorm_rows(&x, n_pred, model.cfg.n_embd)
        })?;
        let q = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_attn, n_pred, attn_wq)
        })?;
        let k_compact = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_attn, n_pred, attn_wk)
        })?;
        let q = match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => q,
            PositionEncodingKind::Rope => measure(profile, "forward.position", || {
                runtime.rope_rows(&q, n_pred, model.cfg.n_embd, n_pred, model.head_dim())
            })?,
        };
        let k_compact = match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => k_compact,
            PositionEncodingKind::Rope => measure(profile, "forward.position", || {
                runtime.rope_rows(
                    &k_compact,
                    n_pred,
                    model.cfg.kv_dim(),
                    n_pred,
                    model.head_dim(),
                )
            })?,
        };
        let v_compact = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_attn, n_pred, attn_wv)
        })?;
        let k = measure(profile, "forward.attention_expand", || {
            runtime.expand_grouped_heads_rows(
                &k_compact,
                n_pred,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let v = measure(profile, "forward.attention_expand", || {
            runtime.expand_grouped_heads_rows(
                &v_compact,
                n_pred,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let scores = measure(profile, "forward.attention_scores", || {
            runtime.attn_scores_seq(
                &q,
                &k,
                n_pred,
                model.cfg.n_head,
                model.head_dim(),
                model.cfg.n_embd,
            )
        })?;
        let weights = measure(profile, "forward.softmax", || {
            runtime.softmax_rows(&scores, n_pred * model.cfg.n_head, n_pred)
        })?;
        let x_attn = measure(profile, "forward.attention_values", || {
            runtime.attn_values_seq(
                &weights,
                &v,
                n_pred,
                model.cfg.n_head,
                model.head_dim(),
                model.cfg.n_embd,
            )
        })?;
        let attn_out = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_attn, n_pred, attn_wo)
        })?;
        x = measure(profile, "forward.add", || {
            runtime.add(&attn_out, &x_residual_attn)
        })?;

        let x_residual_mlp = measure(profile, "forward.copy", || {
            runtime.copy_vector(&x, "rustgpt-train-seq-mlp-residual")
        })?;
        let x_norm_mlp = measure(profile, "forward.rmsnorm", || {
            runtime.rmsnorm_rows(&x, n_pred, model.cfg.n_embd)
        })?;
        let mlp_hidden_pre = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_mlp, n_pred, mlp_fc1)
        })?;
        let mlp_gate_pre = if let Some(mlp_fc_gate) = mlp_fc_gate {
            Some(measure(profile, "forward.matvec", || {
                runtime.matmul_rows(&x_norm_mlp, n_pred, mlp_fc_gate)
            })?)
        } else {
            None
        };
        let mlp_hidden_act = measure(profile, "forward.activation", || {
            match model.cfg.activation {
                ActivationKind::Relu => runtime.relu(&mlp_hidden_pre),
                ActivationKind::Gelu => runtime.gelu(&mlp_hidden_pre),
                ActivationKind::SwiGlu => {
                    let gate_act = runtime.silu(
                        mlp_gate_pre
                            .as_ref()
                            .expect("swiglu activation requires gate pre-activations"),
                    )?;
                    runtime.mul(&mlp_hidden_pre, &gate_act)
                }
            }
        })?;
        let mlp_out = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&mlp_hidden_act, n_pred, mlp_fc2)
        })?;
        x = measure(profile, "forward.add", || {
            runtime.add(&mlp_out, &x_residual_mlp)
        })?;

        gpu_layers.push(GpuTrainingLayerCache {
            x_residual_attn,
            x_norm_attn,
            q,
            k,
            v,
            attn_weights: weights,
            x_attn,
            x_residual_mlp,
            x_norm_mlp,
            mlp_hidden_pre,
            mlp_gate_pre,
            mlp_hidden_act,
        });
    }

    let final_x = measure(profile, "forward.copy", || {
        runtime.copy_vector(&x, "rustgpt-train-final-x")
    })?;
    let final_norm_x = measure(profile, "forward.rmsnorm", || {
        runtime.rmsnorm_rows(&final_x, n_pred, model.cfg.n_embd)
    })?;
    let lm_head = lookup_parameter(matrices, model.output_parameter_id())?;
    let logits = measure(profile, "forward.matvec", || {
        runtime.matmul_rows(&final_norm_x, n_pred, lm_head)
    })?;
    let probs_gpu = measure(profile, "forward.softmax", || {
        runtime.softmax_rows(&logits, n_pred, model.cfg.vocab_size)
    })?;
    let (d_logits, losses_gpu) = measure(profile, "forward.loss", || {
        runtime.cross_entropy_rows(
            &probs_gpu,
            &target_ids,
            &loss_weights,
            n_pred,
            model.cfg.vocab_size,
            if loss_weight_sum > 0.0 {
                grad_scale / loss_weight_sum
            } else {
                0.0
            },
        )
    })?;
    let mean_loss = if capture_loss {
        let losses = measure(profile, "forward.readback", || {
            runtime.readback_vector(&losses_gpu)
        })?;
        if loss_weight_sum > 0.0 {
            losses.iter().sum::<f32>() / loss_weight_sum
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut token_caches = Vec::with_capacity(n_pred);
    for pos_id in 0..n_pred {
        let token_id = example.input_ids[pos_id];
        let target_id = example.target_ids[pos_id];
        token_caches.push(TokenForwardCache {
            token_id,
            target_id,
            pos_id,
            loss_weight: example.loss_mask[pos_id],
            embed_sum: Vec::new(),
            embed_rms_inv: 0.0,
            layers: Vec::new(),
            final_x: Vec::new(),
            final_norm_x: Vec::new(),
            final_rms_inv: 0.0,
            logits: Vec::new(),
            probs: Vec::new(),
        });
    }

    let handle = runtime.store_training_sequence(GpuTrainingSequenceCache {
        seq_len: n_pred,
        token_ids,
        embed_sum,
        final_x,
        final_norm_x,
        d_logits,
        layers: gpu_layers,
    });

    Ok(SequenceForwardCache {
        tokens: token_caches,
        mean_loss,
        loss_weight_sum,
        grad_scale,
        device_sequence: Some(handle),
    })
}

pub(super) fn accumulate_gpu_training_batch(
    runtime: &GpuRuntime,
    matrices: &HashMap<ParameterId, GpuMatrix>,
    model: &Model,
    batch_examples: &[SequenceExample],
    capture_loss: bool,
    grad_scale: f32,
    profile: Option<&RuntimeProfile>,
) -> Result<Option<f32>> {
    let Some(seq_len) = shared_batch_seq_len(model, batch_examples)? else {
        return Ok(None);
    };
    let batch_size = batch_examples.len();
    let total_rows = batch_size
        .checked_mul(seq_len)
        .ok_or_else(|| RustGptError::Gpu("training batch row size overflow".to_string()))?;

    let mut embed_sum_host = vec![0.0; total_rows * model.cfg.n_embd];
    let mut token_ids_host = Vec::with_capacity(total_rows);
    let mut target_ids_host = Vec::with_capacity(total_rows);
    let mut loss_weights_host = Vec::with_capacity(total_rows);
    let mut loss_weight_sum = 0.0;
    for (batch_idx, example) in batch_examples.iter().enumerate() {
        for pos_id in 0..seq_len {
            let token_id = example.input_ids[pos_id];
            let target_id = example.target_ids[pos_id];
            if token_id >= model.cfg.vocab_size || target_id >= model.cfg.vocab_size {
                return Err(RustGptError::Gpu(
                    "sequence token id out of range".to_string(),
                ));
            }
            token_ids_host.push(token_id as u32);
            target_ids_host.push(target_id as u32);
            let loss_weight = example.loss_mask[pos_id];
            loss_weights_host.push(loss_weight);
            loss_weight_sum += loss_weight;
            let row_idx = batch_idx * seq_len + pos_id;
            let out_row =
                &mut embed_sum_host[row_idx * model.cfg.n_embd..(row_idx + 1) * model.cfg.n_embd];
            let token_row = model.wte.row(token_id);
            match model.cfg.position_encoding {
                PositionEncodingKind::LearnedAbsolute => {
                    let pos_row = model
                        .position_embedding()
                        .expect("learned position encoding requires wpe")
                        .row(pos_id);
                    for feature_idx in 0..model.cfg.n_embd {
                        out_row[feature_idx] = token_row[feature_idx] + pos_row[feature_idx];
                    }
                }
                PositionEncodingKind::Rope => out_row.copy_from_slice(token_row),
            }
        }
    }

    let embed_sum = measure(profile, "forward.upload_input", || {
        runtime.upload_vector(&embed_sum_host, "rustgpt-train-batch-embed-sum")
    })?;
    let token_ids = measure(profile, "forward.upload_input", || {
        runtime.upload_u32_vector(&token_ids_host, "rustgpt-train-batch-token-ids")
    })?;
    let target_ids = measure(profile, "forward.upload_input", || {
        runtime.upload_u32_vector(&target_ids_host, "rustgpt-train-batch-target-ids")
    })?;
    let loss_weights = measure(profile, "forward.upload_input", || {
        runtime.upload_vector(&loss_weights_host, "rustgpt-train-batch-loss-weights")
    })?;
    let mut x = measure(profile, "forward.rmsnorm", || {
        runtime.rmsnorm_rows(&embed_sum, total_rows, model.cfg.n_embd)
    })?;
    let mut gpu_layers = Vec::with_capacity(model.cfg.n_layer);

    for layer_idx in 0..model.cfg.n_layer {
        let attn_wq = lookup_parameter(matrices, ParameterId::AttnWq(layer_idx))?;
        let attn_wk = lookup_parameter(matrices, ParameterId::AttnWk(layer_idx))?;
        let attn_wv = lookup_parameter(matrices, ParameterId::AttnWv(layer_idx))?;
        let attn_wo = lookup_parameter(matrices, ParameterId::AttnWo(layer_idx))?;
        let mlp_fc1 = lookup_parameter(matrices, ParameterId::MlpFc1(layer_idx))?;
        let mlp_fc_gate = if model.cfg.activation == ActivationKind::SwiGlu {
            Some(lookup_parameter(
                matrices,
                ParameterId::MlpFcGate(layer_idx),
            )?)
        } else {
            None
        };
        let mlp_fc2 = lookup_parameter(matrices, ParameterId::MlpFc2(layer_idx))?;

        let x_residual_attn = measure(profile, "forward.copy", || {
            runtime.copy_vector(&x, "rustgpt-train-batch-attn-residual")
        })?;
        let x_norm_attn = measure(profile, "forward.rmsnorm", || {
            runtime.rmsnorm_rows(&x, total_rows, model.cfg.n_embd)
        })?;
        let q = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_attn, total_rows, attn_wq)
        })?;
        let k_compact = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_attn, total_rows, attn_wk)
        })?;
        let q = match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => q,
            PositionEncodingKind::Rope => measure(profile, "forward.position", || {
                runtime.rope_rows(&q, total_rows, model.cfg.n_embd, seq_len, model.head_dim())
            })?,
        };
        let k_compact = match model.cfg.position_encoding {
            PositionEncodingKind::LearnedAbsolute => k_compact,
            PositionEncodingKind::Rope => measure(profile, "forward.position", || {
                runtime.rope_rows(
                    &k_compact,
                    total_rows,
                    model.cfg.kv_dim(),
                    seq_len,
                    model.head_dim(),
                )
            })?,
        };
        let v_compact = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_attn, total_rows, attn_wv)
        })?;
        let k = measure(profile, "forward.attention_expand", || {
            runtime.expand_grouped_heads_rows(
                &k_compact,
                total_rows,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let v = measure(profile, "forward.attention_expand", || {
            runtime.expand_grouped_heads_rows(
                &v_compact,
                total_rows,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let scores = measure(profile, "forward.attention_scores", || {
            runtime.attn_scores_batch(
                &q,
                &k,
                batch_size,
                seq_len,
                model.cfg.n_head,
                model.head_dim(),
                model.cfg.n_embd,
            )
        })?;
        let weights = measure(profile, "forward.softmax", || {
            runtime.softmax_rows(&scores, total_rows * model.cfg.n_head, seq_len)
        })?;
        let x_attn = measure(profile, "forward.attention_values", || {
            runtime.attn_values_batch(
                &weights,
                &v,
                batch_size,
                seq_len,
                model.cfg.n_head,
                model.head_dim(),
                model.cfg.n_embd,
            )
        })?;
        let attn_out = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_attn, total_rows, attn_wo)
        })?;
        x = measure(profile, "forward.add", || {
            runtime.add(&attn_out, &x_residual_attn)
        })?;

        let x_residual_mlp = measure(profile, "forward.copy", || {
            runtime.copy_vector(&x, "rustgpt-train-batch-mlp-residual")
        })?;
        let x_norm_mlp = measure(profile, "forward.rmsnorm", || {
            runtime.rmsnorm_rows(&x, total_rows, model.cfg.n_embd)
        })?;
        let mlp_hidden_pre = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&x_norm_mlp, total_rows, mlp_fc1)
        })?;
        let mlp_gate_pre = if let Some(mlp_fc_gate) = mlp_fc_gate {
            Some(measure(profile, "forward.matvec", || {
                runtime.matmul_rows(&x_norm_mlp, total_rows, mlp_fc_gate)
            })?)
        } else {
            None
        };
        let mlp_hidden_act = measure(profile, "forward.activation", || {
            match model.cfg.activation {
                ActivationKind::Relu => runtime.relu(&mlp_hidden_pre),
                ActivationKind::Gelu => runtime.gelu(&mlp_hidden_pre),
                ActivationKind::SwiGlu => {
                    let gate_act = runtime.silu(
                        mlp_gate_pre
                            .as_ref()
                            .expect("swiglu activation requires gate pre-activations"),
                    )?;
                    runtime.mul(&mlp_hidden_pre, &gate_act)
                }
            }
        })?;
        let mlp_out = measure(profile, "forward.matvec", || {
            runtime.matmul_rows(&mlp_hidden_act, total_rows, mlp_fc2)
        })?;
        x = measure(profile, "forward.add", || {
            runtime.add(&mlp_out, &x_residual_mlp)
        })?;

        gpu_layers.push(GpuTrainingLayerCache {
            x_residual_attn,
            x_norm_attn,
            q,
            k,
            v,
            attn_weights: weights,
            x_attn,
            x_residual_mlp,
            x_norm_mlp,
            mlp_hidden_pre,
            mlp_gate_pre,
            mlp_hidden_act,
        });
    }

    let final_x = measure(profile, "forward.copy", || {
        runtime.copy_vector(&x, "rustgpt-train-batch-final-x")
    })?;
    let final_norm_x = measure(profile, "forward.rmsnorm", || {
        runtime.rmsnorm_rows(&final_x, total_rows, model.cfg.n_embd)
    })?;
    let lm_head = lookup_parameter(matrices, model.output_parameter_id())?;
    let logits = measure(profile, "forward.matvec", || {
        runtime.matmul_rows(&final_norm_x, total_rows, lm_head)
    })?;
    let probs_gpu = measure(profile, "forward.softmax", || {
        runtime.softmax_rows(&logits, total_rows, model.cfg.vocab_size)
    })?;
    let (d_logits, losses_gpu) = measure(profile, "forward.loss", || {
        runtime.cross_entropy_rows(
            &probs_gpu,
            &target_ids,
            &loss_weights,
            total_rows,
            model.cfg.vocab_size,
            if loss_weight_sum > 0.0 {
                grad_scale / loss_weight_sum
            } else {
                0.0
            },
        )
    })?;
    let mean_loss = if capture_loss {
        let losses = measure(profile, "forward.readback", || {
            runtime.readback_vector(&losses_gpu)
        })?;
        if loss_weight_sum > 0.0 {
            losses.iter().sum::<f32>() / loss_weight_sum
        } else {
            0.0
        }
    } else {
        0.0
    };

    let d_key_cache = (0..model.cfg.n_layer)
        .map(|layer_idx| {
            runtime.zeroed_vector(
                total_rows * model.cfg.n_embd,
                &format!("rustgpt-batch-dkeys-layer{layer_idx}"),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let d_value_cache = (0..model.cfg.n_layer)
        .map(|layer_idx| {
            runtime.zeroed_vector(
                total_rows * model.cfg.n_embd,
                &format!("rustgpt-batch-dvalues-layer{layer_idx}"),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let d_q_cache = (0..model.cfg.n_layer)
        .map(|layer_idx| {
            runtime.zeroed_vector(
                total_rows * model.cfg.n_embd,
                &format!("rustgpt-batch-dq-layer{layer_idx}"),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let wte = lookup_parameter(matrices, ParameterId::Wte)?;
    let wpe = if model.position_embedding().is_some() {
        Some(lookup_parameter(matrices, ParameterId::Wpe)?)
    } else {
        None
    };
    measure(profile, "backward.grad_accum", || {
        runtime.outer_product_rows_accum(&final_norm_x, &d_logits, total_rows, lm_head)
    })?;
    let d_final_norm_x = measure(profile, "backward.matvec_t", || {
        runtime.matmul_rows_transposed(&d_logits, total_rows, lm_head)
    })?;
    let mut dx = measure(profile, "backward.rmsnorm", || {
        runtime.rmsnorm_backward_rows(&final_x, &d_final_norm_x, total_rows, model.cfg.n_embd)
    })?;

    for layer_idx in (0..model.cfg.n_layer).rev() {
        let gpu_layer = &gpu_layers[layer_idx];
        let attn_wo = lookup_parameter(matrices, ParameterId::AttnWo(layer_idx))?;
        let mlp_fc1 = lookup_parameter(matrices, ParameterId::MlpFc1(layer_idx))?;
        let mlp_fc_gate = if model.cfg.activation == ActivationKind::SwiGlu {
            Some(lookup_parameter(
                matrices,
                ParameterId::MlpFcGate(layer_idx),
            )?)
        } else {
            None
        };
        let mlp_fc2 = lookup_parameter(matrices, ParameterId::MlpFc2(layer_idx))?;
        let attn_wq = lookup_parameter(matrices, ParameterId::AttnWq(layer_idx))?;
        let attn_wk = lookup_parameter(matrices, ParameterId::AttnWk(layer_idx))?;
        let attn_wv = lookup_parameter(matrices, ParameterId::AttnWv(layer_idx))?;

        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.mlp_hidden_act, &dx, total_rows, mlp_fc2)
        })?;
        let d_hidden_act = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(&dx, total_rows, mlp_fc2)
        })?;
        let d_x_norm_mlp = match model.cfg.activation {
            ActivationKind::Relu => {
                let d_hidden_pre = measure(profile, "backward.activation", || {
                    runtime.relu_backward_vector(&gpu_layer.mlp_hidden_pre, &d_hidden_act)
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_hidden_pre,
                        total_rows,
                        mlp_fc1,
                    )
                })?;
                measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(&d_hidden_pre, total_rows, mlp_fc1)
                })?
            }
            ActivationKind::Gelu => {
                let d_hidden_pre = measure(profile, "backward.activation", || {
                    runtime.gelu_backward_vector(&gpu_layer.mlp_hidden_pre, &d_hidden_act)
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_hidden_pre,
                        total_rows,
                        mlp_fc1,
                    )
                })?;
                measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(&d_hidden_pre, total_rows, mlp_fc1)
                })?
            }
            ActivationKind::SwiGlu => {
                let gate_pre = gpu_layer
                    .mlp_gate_pre
                    .as_ref()
                    .expect("swiglu activation requires gate pre-activations");
                let gate_act = measure(profile, "backward.activation", || runtime.silu(gate_pre))?;
                let d_value = measure(profile, "backward.activation", || {
                    runtime.mul(&d_hidden_act, &gate_act)
                })?;
                let d_gate_act = measure(profile, "backward.activation", || {
                    runtime.mul(&d_hidden_act, &gpu_layer.mlp_hidden_pre)
                })?;
                let d_gate = measure(profile, "backward.activation", || {
                    runtime.silu_backward_vector(gate_pre, &d_gate_act)
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_value,
                        total_rows,
                        mlp_fc1,
                    )
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_gate,
                        total_rows,
                        mlp_fc_gate.expect("swiglu activation requires gate weights"),
                    )
                })?;
                let d_x_norm_value = measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(&d_value, total_rows, mlp_fc1)
                })?;
                let d_x_norm_gate = measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(
                        &d_gate,
                        total_rows,
                        mlp_fc_gate.expect("swiglu activation requires gate weights"),
                    )
                })?;
                measure(profile, "backward.add", || {
                    runtime.add(&d_x_norm_value, &d_x_norm_gate)
                })?
            }
        };
        let d_x_from_norm_mlp = measure(profile, "backward.mlp_block", || {
            runtime.rmsnorm_backward_rows(
                &gpu_layer.x_residual_mlp,
                &d_x_norm_mlp,
                total_rows,
                model.cfg.n_embd,
            )
        })?;
        let d_attn_out = measure(profile, "backward.add", || {
            runtime.add(&dx, &d_x_from_norm_mlp)
        })?;

        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.x_attn, &d_attn_out, total_rows, attn_wo)
        })?;
        let d_x_attn = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(&d_attn_out, total_rows, attn_wo)
        })?;

        for batch_idx in 0..batch_size {
            for pos in (0..seq_len).rev() {
                measure(profile, "backward.attention", || {
                    runtime.attention_backward_batch_sequence(
                        &d_x_attn,
                        &gpu_layer.q,
                        &gpu_layer.attn_weights,
                        &gpu_layer.k,
                        &gpu_layer.v,
                        &d_q_cache[layer_idx],
                        &d_key_cache[layer_idx],
                        &d_value_cache[layer_idx],
                        batch_idx,
                        seq_len,
                        pos,
                        model.cfg.n_head,
                        model.head_dim(),
                        model.cfg.n_embd,
                    )
                })?;
            }
        }

        let d_q_rope = if model.cfg.position_encoding == PositionEncodingKind::Rope {
            Some(measure(profile, "backward.position", || {
                runtime.rope_rows_inverse(
                    &d_q_cache[layer_idx],
                    total_rows,
                    model.cfg.n_embd,
                    seq_len,
                    model.head_dim(),
                )
            })?)
        } else {
            None
        };
        let d_k_compact = measure(profile, "backward.attention_collapse", || {
            runtime.collapse_grouped_heads_rows(
                &d_key_cache[layer_idx],
                total_rows,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let d_v_compact = measure(profile, "backward.attention_collapse", || {
            runtime.collapse_grouped_heads_rows(
                &d_value_cache[layer_idx],
                total_rows,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let d_k_rope = if model.cfg.position_encoding == PositionEncodingKind::Rope {
            Some(measure(profile, "backward.position", || {
                runtime.rope_rows_inverse(
                    &d_k_compact,
                    total_rows,
                    model.cfg.kv_dim(),
                    seq_len,
                    model.head_dim(),
                )
            })?)
        } else {
            None
        };
        let d_q_input = d_q_rope.as_ref().unwrap_or(&d_q_cache[layer_idx]);
        let d_k_input = d_k_rope.as_ref().unwrap_or(&d_k_compact);

        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.x_norm_attn, d_q_input, total_rows, attn_wq)
        })?;
        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.x_norm_attn, d_k_input, total_rows, attn_wk)
        })?;
        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(
                &gpu_layer.x_norm_attn,
                &d_v_compact,
                total_rows,
                attn_wv,
            )
        })?;
        let d_x_norm_q = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(d_q_input, total_rows, attn_wq)
        })?;
        let d_x_norm_k = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(d_k_input, total_rows, attn_wk)
        })?;
        let d_x_norm_v = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(&d_v_compact, total_rows, attn_wv)
        })?;
        let d_x_norm_qk = measure(profile, "backward.add", || {
            runtime.add(&d_x_norm_q, &d_x_norm_k)
        })?;
        let d_x_norm_attn = measure(profile, "backward.add", || {
            runtime.add(&d_x_norm_qk, &d_x_norm_v)
        })?;
        let d_x_from_norm_attn = measure(profile, "backward.attn_proj_block", || {
            runtime.rmsnorm_backward_rows(
                &gpu_layer.x_residual_attn,
                &d_x_norm_attn,
                total_rows,
                model.cfg.n_embd,
            )
        })?;
        dx = measure(profile, "backward.add", || {
            runtime.add(&d_attn_out, &d_x_from_norm_attn)
        })?;
    }

    let d_embed_sum = measure(profile, "backward.rmsnorm", || {
        runtime.rmsnorm_backward_rows(&embed_sum, &dx, total_rows, model.cfg.n_embd)
    })?;
    measure(profile, "backward.row_grad", || {
        runtime.scatter_embed_grads(
            &token_ids,
            &d_embed_sum,
            wte,
            wpe,
            total_rows,
            model.cfg.n_embd,
        )
    })?;

    Ok(Some(mean_loss))
}

pub(super) fn backward_gpu_training_sequence(
    runtime: &GpuRuntime,
    matrices: &HashMap<ParameterId, GpuMatrix>,
    _matrix_index: &HashMap<usize, ParameterId>,
    model: &mut Model,
    _sequence: &SequenceForwardCache,
    handle: DeviceSequenceHandle,
    profile: Option<&RuntimeProfile>,
) -> Result<()> {
    let gpu_sequence = runtime.take_training_sequence(handle).ok_or_else(|| {
        RustGptError::Gpu(format!(
            "missing GPU training sequence cache for handle {}",
            handle.0
        ))
    })?;
    let seq_len = gpu_sequence.seq_len;
    let d_key_cache = (0..model.cfg.n_layer)
        .map(|layer_idx| {
            runtime.zeroed_vector(
                seq_len * model.cfg.n_embd,
                &format!("rustgpt-dkeys-layer{layer_idx}"),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let d_value_cache = (0..model.cfg.n_layer)
        .map(|layer_idx| {
            runtime.zeroed_vector(
                seq_len * model.cfg.n_embd,
                &format!("rustgpt-dvalues-layer{layer_idx}"),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let d_q_cache = (0..model.cfg.n_layer)
        .map(|layer_idx| {
            runtime.zeroed_vector(
                seq_len * model.cfg.n_embd,
                &format!("rustgpt-dq-layer{layer_idx}"),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let lm_head = lookup_parameter(matrices, model.output_parameter_id())?;
    let wte = lookup_parameter(matrices, ParameterId::Wte)?;
    let wpe = if model.position_embedding().is_some() {
        Some(lookup_parameter(matrices, ParameterId::Wpe)?)
    } else {
        None
    };
    measure(profile, "backward.grad_accum", || {
        runtime.outer_product_rows_accum(
            &gpu_sequence.final_norm_x,
            &gpu_sequence.d_logits,
            seq_len,
            lm_head,
        )
    })?;
    let d_final_norm_x = measure(profile, "backward.matvec_t", || {
        runtime.matmul_rows_transposed(&gpu_sequence.d_logits, seq_len, lm_head)
    })?;
    let mut dx = measure(profile, "backward.rmsnorm", || {
        runtime.rmsnorm_backward_rows(
            &gpu_sequence.final_x,
            &d_final_norm_x,
            seq_len,
            model.cfg.n_embd,
        )
    })?;

    for layer_idx in (0..model.cfg.n_layer).rev() {
        let gpu_layer = &gpu_sequence.layers[layer_idx];
        let attn_wo = lookup_parameter(matrices, ParameterId::AttnWo(layer_idx))?;
        let mlp_fc1 = lookup_parameter(matrices, ParameterId::MlpFc1(layer_idx))?;
        let mlp_fc_gate = if model.cfg.activation == ActivationKind::SwiGlu {
            Some(lookup_parameter(
                matrices,
                ParameterId::MlpFcGate(layer_idx),
            )?)
        } else {
            None
        };
        let mlp_fc2 = lookup_parameter(matrices, ParameterId::MlpFc2(layer_idx))?;
        let attn_wq = lookup_parameter(matrices, ParameterId::AttnWq(layer_idx))?;
        let attn_wk = lookup_parameter(matrices, ParameterId::AttnWk(layer_idx))?;
        let attn_wv = lookup_parameter(matrices, ParameterId::AttnWv(layer_idx))?;

        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.mlp_hidden_act, &dx, seq_len, mlp_fc2)
        })?;
        let d_hidden_act = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(&dx, seq_len, mlp_fc2)
        })?;
        let d_x_norm_mlp = match model.cfg.activation {
            ActivationKind::Relu => {
                let d_hidden_pre = measure(profile, "backward.activation", || {
                    runtime.relu_backward_vector(&gpu_layer.mlp_hidden_pre, &d_hidden_act)
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_hidden_pre,
                        seq_len,
                        mlp_fc1,
                    )
                })?;
                measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(&d_hidden_pre, seq_len, mlp_fc1)
                })?
            }
            ActivationKind::Gelu => {
                let d_hidden_pre = measure(profile, "backward.activation", || {
                    runtime.gelu_backward_vector(&gpu_layer.mlp_hidden_pre, &d_hidden_act)
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_hidden_pre,
                        seq_len,
                        mlp_fc1,
                    )
                })?;
                measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(&d_hidden_pre, seq_len, mlp_fc1)
                })?
            }
            ActivationKind::SwiGlu => {
                let gate_pre = gpu_layer
                    .mlp_gate_pre
                    .as_ref()
                    .expect("swiglu activation requires gate pre-activations");
                let gate_act = measure(profile, "backward.activation", || runtime.silu(gate_pre))?;
                let d_value = measure(profile, "backward.activation", || {
                    runtime.mul(&d_hidden_act, &gate_act)
                })?;
                let d_gate_act = measure(profile, "backward.activation", || {
                    runtime.mul(&d_hidden_act, &gpu_layer.mlp_hidden_pre)
                })?;
                let d_gate = measure(profile, "backward.activation", || {
                    runtime.silu_backward_vector(gate_pre, &d_gate_act)
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_value,
                        seq_len,
                        mlp_fc1,
                    )
                })?;
                measure(profile, "backward.grad_accum", || {
                    runtime.outer_product_rows_accum(
                        &gpu_layer.x_norm_mlp,
                        &d_gate,
                        seq_len,
                        mlp_fc_gate.expect("swiglu activation requires gate weights"),
                    )
                })?;
                let d_x_norm_value = measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(&d_value, seq_len, mlp_fc1)
                })?;
                let d_x_norm_gate = measure(profile, "backward.matvec_t", || {
                    runtime.matmul_rows_transposed(
                        &d_gate,
                        seq_len,
                        mlp_fc_gate.expect("swiglu activation requires gate weights"),
                    )
                })?;
                measure(profile, "backward.add", || {
                    runtime.add(&d_x_norm_value, &d_x_norm_gate)
                })?
            }
        };
        let d_x_from_norm_mlp = measure(profile, "backward.mlp_block", || {
            runtime.rmsnorm_backward_rows(
                &gpu_layer.x_residual_mlp,
                &d_x_norm_mlp,
                seq_len,
                model.cfg.n_embd,
            )
        })?;
        let d_attn_out = measure(profile, "backward.add", || {
            runtime.add(&dx, &d_x_from_norm_mlp)
        })?;

        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.x_attn, &d_attn_out, seq_len, attn_wo)
        })?;
        let d_x_attn = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(&d_attn_out, seq_len, attn_wo)
        })?;

        for pos in (0..seq_len).rev() {
            measure(profile, "backward.attention", || {
                runtime.attention_backward(
                    &d_x_attn,
                    &gpu_layer.q,
                    &gpu_layer.attn_weights,
                    &gpu_layer.k.buffer,
                    &gpu_layer.v.buffer,
                    &d_q_cache[layer_idx],
                    &d_key_cache[layer_idx],
                    &d_value_cache[layer_idx],
                    pos,
                    pos + 1,
                    seq_len,
                    model.cfg.n_head,
                    model.head_dim(),
                    model.cfg.n_embd,
                )
            })?;
        }

        let d_q_rope = if model.cfg.position_encoding == PositionEncodingKind::Rope {
            Some(measure(profile, "backward.position", || {
                runtime.rope_rows_inverse(
                    &d_q_cache[layer_idx],
                    seq_len,
                    model.cfg.n_embd,
                    seq_len,
                    model.head_dim(),
                )
            })?)
        } else {
            None
        };
        let d_k_compact = measure(profile, "backward.attention_collapse", || {
            runtime.collapse_grouped_heads_rows(
                &d_key_cache[layer_idx],
                seq_len,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let d_v_compact = measure(profile, "backward.attention_collapse", || {
            runtime.collapse_grouped_heads_rows(
                &d_value_cache[layer_idx],
                seq_len,
                model.cfg.n_head,
                model.cfg.n_kv_head,
                model.head_dim(),
            )
        })?;
        let d_k_rope = if model.cfg.position_encoding == PositionEncodingKind::Rope {
            Some(measure(profile, "backward.position", || {
                runtime.rope_rows_inverse(
                    &d_k_compact,
                    seq_len,
                    model.cfg.kv_dim(),
                    seq_len,
                    model.head_dim(),
                )
            })?)
        } else {
            None
        };
        let d_q_input = d_q_rope.as_ref().unwrap_or(&d_q_cache[layer_idx]);
        let d_k_input = d_k_rope.as_ref().unwrap_or(&d_k_compact);

        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.x_norm_attn, d_q_input, seq_len, attn_wq)
        })?;
        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.x_norm_attn, d_k_input, seq_len, attn_wk)
        })?;
        measure(profile, "backward.grad_accum", || {
            runtime.outer_product_rows_accum(&gpu_layer.x_norm_attn, &d_v_compact, seq_len, attn_wv)
        })?;
        let d_x_norm_q = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(d_q_input, seq_len, attn_wq)
        })?;
        let d_x_norm_k = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(d_k_input, seq_len, attn_wk)
        })?;
        let d_x_norm_v = measure(profile, "backward.matvec_t", || {
            runtime.matmul_rows_transposed(&d_v_compact, seq_len, attn_wv)
        })?;
        let d_x_norm_qk = measure(profile, "backward.add", || {
            runtime.add(&d_x_norm_q, &d_x_norm_k)
        })?;
        let d_x_norm_attn = measure(profile, "backward.add", || {
            runtime.add(&d_x_norm_qk, &d_x_norm_v)
        })?;
        let d_x_from_norm_attn = measure(profile, "backward.attn_proj_block", || {
            runtime.rmsnorm_backward_rows(
                &gpu_layer.x_residual_attn,
                &d_x_norm_attn,
                seq_len,
                model.cfg.n_embd,
            )
        })?;
        dx = measure(profile, "backward.add", || {
            runtime.add(&d_attn_out, &d_x_from_norm_attn)
        })?;
    }

    let d_embed_sum = measure(profile, "backward.rmsnorm", || {
        runtime.rmsnorm_backward_rows(&gpu_sequence.embed_sum, &dx, seq_len, model.cfg.n_embd)
    })?;
    measure(profile, "backward.row_grad", || {
        runtime.scatter_embed_grads(
            &gpu_sequence.token_ids,
            &d_embed_sum,
            wte,
            wpe,
            seq_len,
            model.cfg.n_embd,
        )
    })?;

    Ok(())
}

pub(super) fn shared_batch_seq_len(
    model: &Model,
    batch_examples: &[SequenceExample],
) -> Result<Option<usize>> {
    if batch_examples.len() <= 1 {
        return Ok(None);
    }
    let mut shared = None;
    for example in batch_examples {
        if example.input_ids.is_empty() {
            return Err(RustGptError::Data(
                "need at least two tokens to compute a next-token loss".to_string(),
            ));
        }
        let seq_len = usize::min(model.cfg.block_size, example.input_ids.len());
        match shared {
            None => shared = Some(seq_len),
            Some(existing) if existing == seq_len => {}
            Some(_) => return Ok(None),
        }
    }
    Ok(shared)
}
