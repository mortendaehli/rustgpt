use crate::core::config::{BoundaryMode, DeviceKind, ModelConfig};
use crate::core::rng::Rng;
use crate::core::tensor::{
    Matrix, accumulate_linear_grad, add_in_place, linear, linear_transposed,
};
use crate::data::tokenizer::Tokenizer;
use crate::runtime::forward::forward_token;
use crate::runtime::workspace::new_kv_cache;

use super::inference::run_gpu_forward_token;
use super::training::shared_batch_seq_len;
use super::{ComputeBackend, bytes_to_f32, f32s_to_bytes};

#[test]
fn float_byte_roundtrip_is_lossless() {
    let values = vec![0.25, -3.5, 8.0];
    let bytes = f32s_to_bytes(&values);
    assert_eq!(bytes_to_f32(&bytes).unwrap(), values);
}

#[test]
fn auto_backend_falls_back_to_cpu_or_uses_gpu() {
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
    let model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let backend = ComputeBackend::from_model(&model, DeviceKind::Auto).unwrap();
    assert!(!backend.description().is_empty());
}

#[test]
fn cpu_backend_uses_reference_linear_path() {
    let matrix = Matrix {
        rows: 2,
        cols: 3,
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        grad: vec![0.0; 6],
        m: vec![0.0; 6],
        v: vec![0.0; 6],
    };
    let backend = ComputeBackend::cpu();
    let out = backend.matvec(&[1.0, 0.5, -1.0], &matrix).unwrap();
    assert_eq!(out, linear(&[1.0, 0.5, -1.0], &matrix).unwrap());
}

#[test]
fn gpu_matvec_matches_cpu_when_adapter_is_available() {
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
    let model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let Ok(backend) = ComputeBackend::from_model(&model, DeviceKind::Gpu) else {
        return;
    };

    let input = vec![0.5; model.cfg.n_embd];
    let cpu = linear(&input, &model.layers[0].attn_wq).unwrap();
    let gpu = backend.matvec(&input, &model.layers[0].attn_wq).unwrap();
    for (left, right) in cpu.iter().zip(&gpu) {
        assert!((left - right).abs() < 1e-4);
    }
}

#[test]
fn gpu_transposed_matvec_matches_cpu_when_adapter_is_available() {
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
    let model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let Ok(backend) = ComputeBackend::from_model(&model, DeviceKind::Gpu) else {
        return;
    };

    let input = vec![0.25; model.layers[0].attn_wq.rows];
    let cpu = linear_transposed(&input, &model.layers[0].attn_wq).unwrap();
    let gpu = backend
        .matvec_transposed(&input, &model.layers[0].attn_wq)
        .unwrap();
    for (left, right) in cpu.iter().zip(&gpu) {
        assert!((left - right).abs() < 1e-4);
    }
}

#[test]
fn gpu_backend_syncs_updated_weights_when_adapter_is_available() {
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
    let mut model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let Ok(mut backend) = ComputeBackend::from_model(&model, DeviceKind::Gpu) else {
        return;
    };

    let input = vec![0.5; model.cfg.n_embd];
    let before = backend.matvec(&input, &model.layers[0].attn_wq).unwrap();
    model.layers[0].attn_wq.data[0] += 1.0;
    backend.sync_model(&model).unwrap();
    let after = backend.matvec(&input, &model.layers[0].attn_wq).unwrap();
    assert_ne!(before, after);
}

#[test]
fn gpu_outer_product_grad_accum_matches_cpu_when_adapter_is_available() {
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
    let mut model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let Ok(backend) = ComputeBackend::from_model(&model, DeviceKind::Gpu) else {
        return;
    };

    let mut cpu_matrix = model.layers[0].attn_wq.clone();
    let x = vec![0.25; cpu_matrix.cols];
    let dout = vec![0.5; cpu_matrix.rows];
    accumulate_linear_grad(&x, &dout, &mut cpu_matrix).unwrap();

    backend
        .accumulate_linear_grad(&x, &dout, &mut model.layers[0].attn_wq)
        .unwrap();
    let mut downloaded = model.clone();
    backend.download_model(&mut downloaded).unwrap();
    assert_eq!(downloaded.layers[0].attn_wq.grad, cpu_matrix.grad);
}

#[test]
fn gpu_row_grad_accum_matches_cpu_when_adapter_is_available() {
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
    let mut model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let Ok(backend) = ComputeBackend::from_model(&model, DeviceKind::Gpu) else {
        return;
    };

    let row_idx = 1;
    let row_grad = vec![0.75; model.wte.cols];
    let mut cpu_matrix = model.wte.clone();
    add_in_place(cpu_matrix.grad_row_mut(row_idx), &row_grad);

    backend
        .add_row_grad(row_idx, &row_grad, &mut model.wte)
        .unwrap();
    let mut downloaded = model.clone();
    backend.download_model(&mut downloaded).unwrap();
    assert_eq!(downloaded.wte.grad, cpu_matrix.grad);
}

#[test]
fn gpu_device_resident_forward_matches_cpu_when_adapter_is_available() {
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
    let model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let Ok(backend) = ComputeBackend::from_model(&model, DeviceKind::Gpu) else {
        return;
    };

    let tokens = tokenizer.encode_with_boundaries("em").unwrap();
    let mut cpu_kv_cache = new_kv_cache(model.cfg.n_layer, model.cfg.n_embd);
    let cpu = forward_token(&model, tokens[0], tokens[1], 0, &mut cpu_kv_cache)
        .unwrap()
        .logits;

    let Some((runtime, matrices)) = backend.raw_gpu_state() else {
        panic!("expected gpu backend");
    };
    let mut gpu_layer_caches = (0..model.cfg.n_layer)
        .map(|layer_idx| runtime.create_kv_cache(model.cfg.block_size, model.cfg.n_embd, layer_idx))
        .collect::<crate::core::error::Result<Vec<_>>>()
        .unwrap();
    let gpu = run_gpu_forward_token(
        runtime,
        matrices,
        &model,
        &mut gpu_layer_caches,
        tokens[0],
        0,
        None,
    )
    .unwrap();

    for (left, right) in cpu.iter().zip(&gpu) {
        assert!((left - right).abs() < 1e-4);
    }
}

#[test]
fn shared_batch_seq_len_requires_matching_effective_lengths() {
    let cfg = ModelConfig {
        vocab_size: 8,
        block_size: 4,
        n_layer: 1,
        n_embd: 8,
        n_head: 2,
        boundary_mode: BoundaryMode::SharedBos,
    };
    let mut rng = Rng::from_seed(7);
    let model = crate::model::Model::new(cfg, &mut rng).unwrap();
    let matching = vec![vec![0, 1, 2, 3], vec![0, 4, 5, 6]];
    let mismatched = vec![vec![0, 1, 2, 3], vec![0, 4, 5]];

    assert_eq!(shared_batch_seq_len(&model, &matching).unwrap(), Some(3));
    assert_eq!(shared_batch_seq_len(&model, &mismatched).unwrap(), None);
}
