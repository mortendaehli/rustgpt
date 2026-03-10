//! Public GPU backend façade.
//! This is the layer the rest of the runtime calls; low-level wgpu execution stays in
//! `mod.rs`, `training.rs`, `inference.rs`, and `storage.rs`.

use std::collections::HashMap;

use crate::app::cli::GpuInfoCommand;
use crate::core::config::DeviceKind;
use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::core::tensor::{
    Matrix, accumulate_linear_grad, add_in_place, linear, linear_transposed, relu_backward,
    rmsnorm_backward,
};
use crate::data::tokenizer::Tokenizer;
use crate::model::{Model, ParameterId};
use crate::runtime::profile::RuntimeProfile;
use crate::runtime::train_cache::SequenceForwardCache;

use super::inference::run_gpu_completion;
use super::shaders::AdapterSummary;
use super::training::{
    accumulate_gpu_training_batch, backward_gpu_training_sequence, run_gpu_training_sequence,
};
use super::{GpuMatrix, GpuRuntime};

impl AdapterSummary {
    pub(super) fn from_info(info: wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            driver: info.driver,
            driver_info: info.driver_info,
        }
    }

    pub(super) fn is_cpu_adapter(&self) -> bool {
        self.device_type == "Cpu"
    }
}

pub struct ComputeBackend {
    kind: ComputeBackendKind,
    description: String,
}

enum ComputeBackendKind {
    Cpu,
    Gpu {
        runtime: GpuRuntime,
        matrices: HashMap<ParameterId, GpuMatrix>,
        matrix_index: HashMap<usize, ParameterId>,
    },
}

impl ComputeBackend {
    pub fn cpu() -> Self {
        Self {
            kind: ComputeBackendKind::Cpu,
            description: "cpu".to_string(),
        }
    }

    pub fn from_model(model: &Model, device_kind: DeviceKind) -> Result<Self> {
        match device_kind {
            DeviceKind::Cpu => Ok(Self::cpu()),
            DeviceKind::Auto => match GpuRuntime::new(DeviceKind::Auto) {
                Ok(runtime) => {
                    let description =
                        format!("gpu:{} ({})", runtime.adapter.backend, runtime.adapter.name);
                    let mut backend = Self {
                        kind: ComputeBackendKind::Gpu {
                            runtime,
                            matrices: HashMap::new(),
                            matrix_index: HashMap::new(),
                        },
                        description,
                    };
                    backend.sync_model(model)?;
                    Ok(backend)
                }
                Err(err) => Ok(Self {
                    kind: ComputeBackendKind::Cpu,
                    description: format!("cpu (auto fallback: {err})"),
                }),
            },
            DeviceKind::Gpu => {
                let runtime = GpuRuntime::new(DeviceKind::Gpu)?;
                let description =
                    format!("gpu:{} ({})", runtime.adapter.backend, runtime.adapter.name);
                let mut backend = Self {
                    kind: ComputeBackendKind::Gpu {
                        runtime,
                        matrices: HashMap::new(),
                        matrix_index: HashMap::new(),
                    },
                    description,
                };
                backend.sync_model(model)?;
                Ok(backend)
            }
        }
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn uses_device_optimizer(&self) -> bool {
        matches!(self.kind, ComputeBackendKind::Gpu { .. })
    }

    pub fn forward_training_sequence(
        &self,
        model: &Model,
        tokens: &[usize],
        capture_loss: bool,
        grad_scale: f32,
        profile: Option<&RuntimeProfile>,
    ) -> Result<Option<SequenceForwardCache>> {
        match &self.kind {
            ComputeBackendKind::Cpu => Ok(None),
            ComputeBackendKind::Gpu {
                runtime, matrices, ..
            } => Ok(Some(run_gpu_training_sequence(
                runtime,
                matrices,
                model,
                tokens,
                capture_loss,
                grad_scale,
                profile,
            )?)),
        }
    }

    pub fn backward_training_sequence(
        &self,
        model: &mut Model,
        sequence: &SequenceForwardCache,
        profile: Option<&RuntimeProfile>,
    ) -> Result<bool> {
        match &self.kind {
            ComputeBackendKind::Cpu => Ok(false),
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                let Some(handle) = sequence.device_sequence else {
                    return Ok(false);
                };
                backward_gpu_training_sequence(
                    runtime,
                    matrices,
                    matrix_index,
                    model,
                    sequence,
                    handle,
                    profile,
                )?;
                Ok(true)
            }
        }
    }

    pub fn accumulate_training_batch(
        &self,
        model: &Model,
        batch_tokens: &[Vec<usize>],
        capture_loss: bool,
        grad_scale: f32,
        profile: Option<&RuntimeProfile>,
    ) -> Result<Option<f32>> {
        match &self.kind {
            ComputeBackendKind::Cpu => Ok(None),
            ComputeBackendKind::Gpu {
                runtime, matrices, ..
            } => accumulate_gpu_training_batch(
                runtime,
                matrices,
                model,
                batch_tokens,
                capture_loss,
                grad_scale,
                profile,
            ),
        }
    }

    pub fn sync_model(&mut self, model: &Model) -> Result<()> {
        if let ComputeBackendKind::Gpu {
            runtime,
            matrices,
            matrix_index,
        } = &mut self.kind
        {
            matrix_index.clear();
            model.visit_parameters(|parameter_id, matrix| {
                matrix_index.insert(super::wgpu_utils::matrix_key(matrix), parameter_id);
                runtime.sync_parameter(matrices, parameter_id, matrix)
            })?;
        }
        Ok(())
    }

    pub fn adam_step(
        &mut self,
        model: &mut Model,
        lr_t: f32,
        beta1: f32,
        beta2: f32,
        eps_adam: f32,
        optimizer_step_num: usize,
    ) -> Result<()> {
        match &mut self.kind {
            ComputeBackendKind::Cpu => {
                model.adam_step(lr_t, beta1, beta2, eps_adam, optimizer_step_num);
                Ok(())
            }
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                matrix_index.clear();
                model.visit_parameters_mut(|parameter_id, matrix| {
                    matrix_index.insert(super::wgpu_utils::matrix_key(matrix), parameter_id);
                    let gpu_matrix = matrices.get(&parameter_id).ok_or_else(|| {
                        RustGptError::Gpu(format!(
                            "GPU state missing for parameter {}",
                            parameter_id.label()
                        ))
                    })?;
                    runtime.apply_adam(
                        gpu_matrix,
                        lr_t,
                        beta1,
                        beta2,
                        eps_adam,
                        optimizer_step_num,
                    )?;
                    matrix.grad.fill(0.0);
                    Ok(())
                })
            }
        }
    }

    pub fn download_model(&self, model: &mut Model) -> Result<()> {
        if let ComputeBackendKind::Gpu {
            runtime, matrices, ..
        } = &self.kind
        {
            model.visit_parameters_mut(|parameter_id, matrix| {
                let gpu_matrix = matrices.get(&parameter_id).ok_or_else(|| {
                    RustGptError::Gpu(format!(
                        "GPU state missing for parameter {}",
                        parameter_id.label()
                    ))
                })?;
                runtime.download_parameter(gpu_matrix, matrix)
            })?;
        }
        Ok(())
    }

    pub fn matvec(&self, x: &[f32], matrix: &Matrix) -> Result<Vec<f32>> {
        match &self.kind {
            ComputeBackendKind::Cpu => linear(x, matrix),
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                let gpu_matrix = runtime.lookup_matrix(matrices, matrix_index, matrix)?;
                runtime.matvec(x, gpu_matrix)
            }
        }
    }

    pub fn matvec_transposed(&self, x: &[f32], matrix: &Matrix) -> Result<Vec<f32>> {
        match &self.kind {
            ComputeBackendKind::Cpu => linear_transposed(x, matrix),
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                let gpu_matrix = runtime.lookup_matrix(matrices, matrix_index, matrix)?;
                runtime.matvec_transposed(x, gpu_matrix)
            }
        }
    }

    pub fn accumulate_linear_grad(
        &self,
        x: &[f32],
        dout: &[f32],
        matrix: &mut Matrix,
    ) -> Result<()> {
        match &self.kind {
            ComputeBackendKind::Cpu => accumulate_linear_grad(x, dout, matrix),
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                let gpu_matrix = runtime.lookup_matrix(matrices, matrix_index, matrix)?;
                runtime.accumulate_outer_product(x, dout, gpu_matrix)
            }
        }
    }

    pub fn add_row_grad(&self, row_idx: usize, grad: &[f32], matrix: &mut Matrix) -> Result<()> {
        match &self.kind {
            ComputeBackendKind::Cpu => {
                add_in_place(matrix.grad_row_mut(row_idx), grad);
                Ok(())
            }
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                let gpu_matrix = runtime.lookup_matrix(matrices, matrix_index, matrix)?;
                runtime.accumulate_row_grad(row_idx, grad, gpu_matrix)
            }
        }
    }

    pub fn backward_mlp_residual(
        &self,
        x_residual: &[f32],
        x_norm: &[f32],
        rms_inv: f32,
        mlp_hidden_pre: &[f32],
        mlp_hidden_act: &[f32],
        d_mlp_out: &[f32],
        mlp_fc1: &mut Matrix,
        mlp_fc2: &mut Matrix,
    ) -> Result<Vec<f32>> {
        match &self.kind {
            ComputeBackendKind::Cpu => {
                let d_mlp_hidden_act = linear_transposed(d_mlp_out, mlp_fc2)?;
                accumulate_linear_grad(mlp_hidden_act, d_mlp_out, mlp_fc2)?;
                let d_mlp_hidden_pre = relu_backward(mlp_hidden_pre, &d_mlp_hidden_act);
                let d_x_norm_mlp = linear_transposed(&d_mlp_hidden_pre, mlp_fc1)?;
                accumulate_linear_grad(x_norm, &d_mlp_hidden_pre, mlp_fc1)?;
                Ok(rmsnorm_backward(x_residual, rms_inv, &d_x_norm_mlp))
            }
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                let gpu_fc1 = runtime.lookup_matrix(matrices, matrix_index, mlp_fc1)?;
                let gpu_fc2 = runtime.lookup_matrix(matrices, matrix_index, mlp_fc2)?;
                runtime.backward_mlp_residual(
                    x_residual,
                    x_norm,
                    rms_inv,
                    mlp_hidden_pre,
                    mlp_hidden_act,
                    d_mlp_out,
                    gpu_fc1,
                    gpu_fc2,
                )
            }
        }
    }

    pub fn backward_attention_projections(
        &self,
        x_residual: &[f32],
        x_norm: &[f32],
        rms_inv: f32,
        d_q: &[f32],
        d_k: &[f32],
        d_v: &[f32],
        attn_wq: &mut Matrix,
        attn_wk: &mut Matrix,
        attn_wv: &mut Matrix,
    ) -> Result<Vec<f32>> {
        match &self.kind {
            ComputeBackendKind::Cpu => {
                let d_x_norm_q = linear_transposed(d_q, attn_wq)?;
                accumulate_linear_grad(x_norm, d_q, attn_wq)?;
                let d_x_norm_k = linear_transposed(d_k, attn_wk)?;
                accumulate_linear_grad(x_norm, d_k, attn_wk)?;
                let d_x_norm_v = linear_transposed(d_v, attn_wv)?;
                accumulate_linear_grad(x_norm, d_v, attn_wv)?;
                let mut d_x_norm = d_x_norm_q;
                add_in_place(&mut d_x_norm, &d_x_norm_k);
                add_in_place(&mut d_x_norm, &d_x_norm_v);
                Ok(rmsnorm_backward(x_residual, rms_inv, &d_x_norm))
            }
            ComputeBackendKind::Gpu {
                runtime,
                matrices,
                matrix_index,
            } => {
                let gpu_wq = runtime.lookup_matrix(matrices, matrix_index, attn_wq)?;
                let gpu_wk = runtime.lookup_matrix(matrices, matrix_index, attn_wk)?;
                let gpu_wv = runtime.lookup_matrix(matrices, matrix_index, attn_wv)?;
                runtime.backward_attention_projections(
                    x_residual, x_norm, rms_inv, d_q, d_k, d_v, gpu_wq, gpu_wk, gpu_wv,
                )
            }
        }
    }

    pub fn generate_completion_on_device(
        &self,
        model: &Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        stop_sequences: &[&str],
        rng: &mut Rng,
        profile: Option<&RuntimeProfile>,
    ) -> Result<Option<String>> {
        match &self.kind {
            ComputeBackendKind::Cpu => Ok(None),
            ComputeBackendKind::Gpu {
                runtime, matrices, ..
            } => run_gpu_completion(
                runtime,
                matrices,
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                temperature,
                stop_sequences,
                rng,
                profile,
            )
            .map(Some),
        }
    }
}

#[cfg(test)]
impl ComputeBackend {
    pub(super) fn raw_gpu_state(&self) -> Option<(&GpuRuntime, &HashMap<ParameterId, GpuMatrix>)> {
        match &self.kind {
            ComputeBackendKind::Cpu => None,
            ComputeBackendKind::Gpu {
                runtime, matrices, ..
            } => Some((runtime, matrices)),
        }
    }
}

pub fn run_gpu_info(command: GpuInfoCommand) -> Result<()> {
    if command.gpu.device == DeviceKind::Cpu {
        println!("RustGPT gpu-info");
        println!("device=cpu");
        println!("CPU mode bypasses wgpu and uses the existing host-side implementation.");
        return Ok(());
    }

    let runtime = match GpuRuntime::new(command.gpu.device) {
        Ok(runtime) => runtime,
        Err(err) if command.gpu.device == DeviceKind::Auto => {
            println!("RustGPT gpu-info");
            println!("requested_device=auto");
            println!("status=no-compatible-gpu-adapter");
            println!("note={err}");
            println!("fallback=cpu");
            return Ok(());
        }
        Err(err) => return Err(err),
    };
    println!("RustGPT gpu-info");
    println!("requested_device={}", command.gpu.device);
    println!("adapter_name={}", runtime.adapter.name);
    println!("backend={}", runtime.adapter.backend);
    println!("device_type={}", runtime.adapter.device_type);
    println!("driver={}", runtime.adapter.driver);
    println!("driver_info={}", runtime.adapter.driver_info);
    Ok(())
}
