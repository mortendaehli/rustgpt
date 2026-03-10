use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::mpsc;

use crate::core::config::DeviceKind;
use crate::core::error::{Result, RustGptError};
use crate::core::tensor::Matrix;
use crate::model::ParameterId;
use crate::runtime::train_cache::DeviceSequenceHandle;

const WORKGROUP_SIZE: u32 = 64;
const GPU_ATTN_BACKWARD_MAX_SEQ: usize = 1024;

mod api;
mod inference;
mod shaders;
mod storage;
#[cfg(test)]
mod tests;
mod training;
mod wgpu_utils;

pub use self::api::{ComputeBackend, run_gpu_info};
use self::shaders::*;
use self::storage::{
    GpuIndexVector, GpuLayerCache, GpuMatrix, GpuTrainingLayerCache, GpuTrainingSequenceCache,
    GpuVector,
};
use self::wgpu_utils::{
    adam_params_to_bytes, buffer_f32_range, byte_len, bytes_to_f32, create_adam_bind_group_layout,
    create_attn_backward_bind_group_layout, create_attn_bind_group_layout,
    create_binary_bind_group_layout, create_gather_bind_group_layout,
    create_loss_bind_group_layout, create_matvec_bind_group_layout, create_pipeline,
    create_scatter_embed_bind_group_layout, create_unary_bind_group_layout, f32s_to_bytes,
    len_scalar_params_to_bytes, matrix_key, u32s_to_bytes,
};

struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matvec_bind_group_layout: wgpu::BindGroupLayout,
    adam_bind_group_layout: wgpu::BindGroupLayout,
    unary_bind_group_layout: wgpu::BindGroupLayout,
    binary_bind_group_layout: wgpu::BindGroupLayout,
    gather_bind_group_layout: wgpu::BindGroupLayout,
    attn_bind_group_layout: wgpu::BindGroupLayout,
    attn_backward_bind_group_layout: wgpu::BindGroupLayout,
    scatter_embed_bind_group_layout: wgpu::BindGroupLayout,
    loss_bind_group_layout: wgpu::BindGroupLayout,
    matvec_pipeline: wgpu::ComputePipeline,
    matvec_transposed_pipeline: wgpu::ComputePipeline,
    matmul_rows_pipeline: wgpu::ComputePipeline,
    matmul_rows_transposed_pipeline: wgpu::ComputePipeline,
    adam_pipeline: wgpu::ComputePipeline,
    gather_row_pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
    relu_pipeline: wgpu::ComputePipeline,
    relu_backward_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    rmsnorm_backward_pipeline: wgpu::ComputePipeline,
    rmsnorm_rows_pipeline: wgpu::ComputePipeline,
    rmsnorm_backward_rows_pipeline: wgpu::ComputePipeline,
    softmax_rows_pipeline: wgpu::ComputePipeline,
    attn_scores_pipeline: wgpu::ComputePipeline,
    attn_values_pipeline: wgpu::ComputePipeline,
    attn_scores_seq_pipeline: wgpu::ComputePipeline,
    attn_values_seq_pipeline: wgpu::ComputePipeline,
    attn_scores_batch_pipeline: wgpu::ComputePipeline,
    attn_values_batch_pipeline: wgpu::ComputePipeline,
    attn_backward_pipeline: wgpu::ComputePipeline,
    outer_product_accum_pipeline: wgpu::ComputePipeline,
    outer_product_rows_accum_pipeline: wgpu::ComputePipeline,
    row_add_accum_pipeline: wgpu::ComputePipeline,
    scatter_embed_grads_pipeline: wgpu::ComputePipeline,
    cross_entropy_rows_pipeline: wgpu::ComputePipeline,
    adapter: AdapterSummary,
    training_sequences: RefCell<HashMap<u64, GpuTrainingSequenceCache>>,
    next_training_sequence_id: Cell<u64>,
}

impl GpuRuntime {
    fn new(device_kind: DeviceKind) -> Result<Self> {
        if device_kind == DeviceKind::Cpu {
            return Err(RustGptError::Gpu(
                "CPU mode does not initialize a GPU runtime".to_string(),
            ));
        }

        let instance = wgpu::Instance::default();
        let adapter = request_adapter(&instance, device_kind)?;
        let adapter_info = AdapterSummary::from_info(adapter.get_info());
        if adapter_info.is_cpu_adapter() {
            return Err(RustGptError::Gpu(format!(
                "requested {:?} compute but wgpu selected a CPU adapter",
                device_kind
            )));
        }

        let supported_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_storage_buffers_per_shader_stage = required_limits
            .max_storage_buffers_per_shader_stage
            .max(8)
            .min(supported_limits.max_storage_buffers_per_shader_stage);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rustgpt-compute-device"),
                required_features: wgpu::Features::empty(),
                required_limits,
            },
            None,
        ))
        .map_err(|err| RustGptError::Gpu(format!("failed to create GPU device: {err}")))?;

        let matvec_bind_group_layout = create_matvec_bind_group_layout(&device);
        let adam_bind_group_layout = create_adam_bind_group_layout(&device);
        let unary_bind_group_layout = create_unary_bind_group_layout(&device);
        let binary_bind_group_layout = create_binary_bind_group_layout(&device);
        let gather_bind_group_layout = create_gather_bind_group_layout(&device);
        let attn_bind_group_layout = create_attn_bind_group_layout(&device);
        let attn_backward_bind_group_layout = create_attn_backward_bind_group_layout(&device);
        let scatter_embed_bind_group_layout = create_scatter_embed_bind_group_layout(&device);
        let loss_bind_group_layout = create_loss_bind_group_layout(&device);
        let matvec_pipeline = create_pipeline(
            &device,
            &matvec_bind_group_layout,
            "rustgpt-matvec-pipeline",
            "rustgpt-matvec-shader",
            MATVEC_SHADER,
        );
        let matvec_transposed_pipeline = create_pipeline(
            &device,
            &matvec_bind_group_layout,
            "rustgpt-matvec-transposed-pipeline",
            "rustgpt-matvec-transposed-shader",
            MATVEC_TRANSPOSED_SHADER,
        );
        let matmul_rows_pipeline = create_pipeline(
            &device,
            &matvec_bind_group_layout,
            "rustgpt-matmul-rows-pipeline",
            "rustgpt-matmul-rows-shader",
            MATMUL_ROWS_SHADER,
        );
        let matmul_rows_transposed_pipeline = create_pipeline(
            &device,
            &matvec_bind_group_layout,
            "rustgpt-matmul-rows-transposed-pipeline",
            "rustgpt-matmul-rows-transposed-shader",
            MATMUL_ROWS_TRANSPOSED_SHADER,
        );
        let adam_pipeline = create_pipeline(
            &device,
            &adam_bind_group_layout,
            "rustgpt-adam-pipeline",
            "rustgpt-adam-shader",
            ADAM_SHADER,
        );
        let gather_row_pipeline = create_pipeline(
            &device,
            &gather_bind_group_layout,
            "rustgpt-gather-row-pipeline",
            "rustgpt-gather-row-shader",
            GATHER_ROW_SHADER,
        );
        let add_pipeline = create_pipeline(
            &device,
            &binary_bind_group_layout,
            "rustgpt-add-pipeline",
            "rustgpt-add-shader",
            ADD_SHADER,
        );
        let relu_pipeline = create_pipeline(
            &device,
            &unary_bind_group_layout,
            "rustgpt-relu-pipeline",
            "rustgpt-relu-shader",
            RELU_SHADER,
        );
        let relu_backward_pipeline = create_pipeline(
            &device,
            &binary_bind_group_layout,
            "rustgpt-relu-backward-pipeline",
            "rustgpt-relu-backward-shader",
            RELU_BACKWARD_SHADER,
        );
        let rmsnorm_pipeline = create_pipeline(
            &device,
            &unary_bind_group_layout,
            "rustgpt-rmsnorm-pipeline",
            "rustgpt-rmsnorm-shader",
            RMSNORM_SHADER,
        );
        let rmsnorm_backward_pipeline = create_pipeline(
            &device,
            &binary_bind_group_layout,
            "rustgpt-rmsnorm-backward-pipeline",
            "rustgpt-rmsnorm-backward-shader",
            RMSNORM_BACKWARD_SHADER,
        );
        let rmsnorm_rows_pipeline = create_pipeline(
            &device,
            &unary_bind_group_layout,
            "rustgpt-rmsnorm-rows-pipeline",
            "rustgpt-rmsnorm-rows-shader",
            RMSNORM_ROWS_SHADER,
        );
        let rmsnorm_backward_rows_pipeline = create_pipeline(
            &device,
            &binary_bind_group_layout,
            "rustgpt-rmsnorm-backward-rows-pipeline",
            "rustgpt-rmsnorm-backward-rows-shader",
            RMSNORM_BACKWARD_ROWS_SHADER,
        );
        let softmax_rows_pipeline = create_pipeline(
            &device,
            &unary_bind_group_layout,
            "rustgpt-softmax-rows-pipeline",
            "rustgpt-softmax-rows-shader",
            SOFTMAX_ROWS_SHADER,
        );
        let attn_scores_pipeline = create_pipeline(
            &device,
            &attn_bind_group_layout,
            "rustgpt-attn-scores-pipeline",
            "rustgpt-attn-scores-shader",
            ATTN_SCORES_SHADER,
        );
        let attn_values_pipeline = create_pipeline(
            &device,
            &attn_bind_group_layout,
            "rustgpt-attn-values-pipeline",
            "rustgpt-attn-values-shader",
            ATTN_VALUES_SHADER,
        );
        let attn_scores_seq_pipeline = create_pipeline(
            &device,
            &attn_bind_group_layout,
            "rustgpt-attn-scores-seq-pipeline",
            "rustgpt-attn-scores-seq-shader",
            ATTN_SCORES_SEQ_SHADER,
        );
        let attn_values_seq_pipeline = create_pipeline(
            &device,
            &attn_bind_group_layout,
            "rustgpt-attn-values-seq-pipeline",
            "rustgpt-attn-values-seq-shader",
            ATTN_VALUES_SEQ_SHADER,
        );
        let attn_scores_batch_pipeline = create_pipeline(
            &device,
            &attn_bind_group_layout,
            "rustgpt-attn-scores-batch-pipeline",
            "rustgpt-attn-scores-batch-shader",
            ATTN_SCORES_BATCH_SHADER,
        );
        let attn_values_batch_pipeline = create_pipeline(
            &device,
            &attn_bind_group_layout,
            "rustgpt-attn-values-batch-pipeline",
            "rustgpt-attn-values-batch-shader",
            ATTN_VALUES_BATCH_SHADER,
        );
        let attn_backward_pipeline = create_pipeline(
            &device,
            &attn_backward_bind_group_layout,
            "rustgpt-attn-backward-pipeline",
            "rustgpt-attn-backward-shader",
            ATTN_BACKWARD_SHADER,
        );
        let outer_product_accum_pipeline = create_pipeline(
            &device,
            &matvec_bind_group_layout,
            "rustgpt-outer-product-accum-pipeline",
            "rustgpt-outer-product-accum-shader",
            OUTER_PRODUCT_ACCUM_SHADER,
        );
        let outer_product_rows_accum_pipeline = create_pipeline(
            &device,
            &matvec_bind_group_layout,
            "rustgpt-outer-product-rows-accum-pipeline",
            "rustgpt-outer-product-rows-accum-shader",
            OUTER_PRODUCT_ROWS_ACCUM_SHADER,
        );
        let row_add_accum_pipeline = create_pipeline(
            &device,
            &gather_bind_group_layout,
            "rustgpt-row-add-accum-pipeline",
            "rustgpt-row-add-accum-shader",
            ROW_ADD_ACCUM_SHADER,
        );
        let scatter_embed_grads_pipeline = create_pipeline(
            &device,
            &scatter_embed_bind_group_layout,
            "rustgpt-scatter-embed-grads-pipeline",
            "rustgpt-scatter-embed-grads-shader",
            SCATTER_EMBED_GRADS_SHADER,
        );
        let cross_entropy_rows_pipeline = create_pipeline(
            &device,
            &loss_bind_group_layout,
            "rustgpt-cross-entropy-rows-pipeline",
            "rustgpt-cross-entropy-rows-shader",
            CROSS_ENTROPY_ROWS_SHADER,
        );

        Ok(Self {
            device,
            queue,
            matvec_bind_group_layout,
            adam_bind_group_layout,
            unary_bind_group_layout,
            binary_bind_group_layout,
            gather_bind_group_layout,
            attn_bind_group_layout,
            attn_backward_bind_group_layout,
            scatter_embed_bind_group_layout,
            loss_bind_group_layout,
            matvec_pipeline,
            matvec_transposed_pipeline,
            matmul_rows_pipeline,
            matmul_rows_transposed_pipeline,
            adam_pipeline,
            gather_row_pipeline,
            add_pipeline,
            relu_pipeline,
            relu_backward_pipeline,
            rmsnorm_pipeline,
            rmsnorm_backward_pipeline,
            rmsnorm_rows_pipeline,
            rmsnorm_backward_rows_pipeline,
            softmax_rows_pipeline,
            attn_scores_pipeline,
            attn_values_pipeline,
            attn_scores_seq_pipeline,
            attn_values_seq_pipeline,
            attn_scores_batch_pipeline,
            attn_values_batch_pipeline,
            attn_backward_pipeline,
            outer_product_accum_pipeline,
            outer_product_rows_accum_pipeline,
            row_add_accum_pipeline,
            scatter_embed_grads_pipeline,
            cross_entropy_rows_pipeline,
            adapter: adapter_info,
            training_sequences: RefCell::new(HashMap::new()),
            next_training_sequence_id: Cell::new(1),
        })
    }

    fn sync_parameter(
        &self,
        matrices: &mut HashMap<ParameterId, GpuMatrix>,
        parameter_id: ParameterId,
        matrix: &Matrix,
    ) -> Result<()> {
        let label = parameter_id.label();
        let data_bytes = f32s_to_bytes(&matrix.data);
        let grad_bytes = f32s_to_bytes(&matrix.grad);
        let m_bytes = f32s_to_bytes(&matrix.m);
        let v_bytes = f32s_to_bytes(&matrix.v);

        match matrices.get(&parameter_id) {
            Some(existing) if existing.rows == matrix.rows && existing.cols == matrix.cols => {
                self.queue
                    .write_buffer(&existing.weight_buffer, 0, &data_bytes);
                self.queue
                    .write_buffer(&existing.grad_buffer, 0, &grad_bytes);
                self.queue.write_buffer(&existing.m_buffer, 0, &m_bytes);
                self.queue.write_buffer(&existing.v_buffer, 0, &v_bytes);
            }
            _ => {
                let uploaded = self.create_matrix_state(matrix, &label)?;
                matrices.insert(parameter_id, uploaded);
            }
        }
        Ok(())
    }

    fn lookup_matrix<'a>(
        &self,
        matrices: &'a HashMap<ParameterId, GpuMatrix>,
        matrix_index: &HashMap<usize, ParameterId>,
        matrix: &Matrix,
    ) -> Result<&'a GpuMatrix> {
        let parameter_id = matrix_index.get(&matrix_key(matrix)).ok_or_else(|| {
            RustGptError::Gpu("GPU backend was used before model weights were synced".to_string())
        })?;
        matrices.get(parameter_id).ok_or_else(|| {
            RustGptError::Gpu(format!(
                "GPU backend is missing state for parameter {}",
                parameter_id.label()
            ))
        })
    }

    fn create_matrix_state(&self, matrix: &Matrix, label: &str) -> Result<GpuMatrix> {
        let weight_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len(matrix.data.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&weight_buffer, 0, &f32s_to_bytes(&matrix.data));
        let grad_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-grad")),
            size: byte_len(matrix.grad.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&grad_buffer, 0, &f32s_to_bytes(&matrix.grad));
        let m_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-m")),
            size: byte_len(matrix.m.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&m_buffer, 0, &f32s_to_bytes(&matrix.m));
        let v_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-v")),
            size: byte_len(matrix.v.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&v_buffer, 0, &f32s_to_bytes(&matrix.v));
        Ok(GpuMatrix {
            rows: matrix.rows,
            cols: matrix.cols,
            weight_buffer,
            grad_buffer,
            m_buffer,
            v_buffer,
        })
    }

    fn matvec(&self, x: &[f32], matrix: &GpuMatrix) -> Result<Vec<f32>> {
        if x.len() != matrix.cols {
            return Err(RustGptError::Gpu(format!(
                "GPU matvec shape mismatch: input has {} elements, matrix expects {}",
                x.len(),
                matrix.cols
            )));
        }

        self.run_kernel(
            x,
            matrix,
            matrix.rows,
            &matrix.weight_buffer,
            &self.matvec_pipeline,
            "rustgpt-matvec",
        )
    }

    fn matvec_transposed(&self, x: &[f32], matrix: &GpuMatrix) -> Result<Vec<f32>> {
        if x.len() != matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "GPU matvec_transposed shape mismatch: input has {} elements, matrix expects {} rows",
                x.len(),
                matrix.rows
            )));
        }

        self.run_kernel(
            x,
            matrix,
            matrix.cols,
            &matrix.weight_buffer,
            &self.matvec_transposed_pipeline,
            "rustgpt-matvec-transposed",
        )
    }

    fn run_kernel(
        &self,
        x: &[f32],
        matrix: &GpuMatrix,
        output_len: usize,
        weight_buffer: &wgpu::Buffer,
        pipeline: &wgpu::ComputePipeline,
        label_prefix: &str,
    ) -> Result<Vec<f32>> {
        let x_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-input")),
            size: byte_len(x.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&x_buffer, 0, &f32s_to_bytes(x));

        let y_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-output")),
            size: byte_len(output_len, std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-staging")),
            size: byte_len(output_len, std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = [matrix.rows as u32, matrix.cols as u32, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-params")),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label_prefix}-bind-group")),
            layout: &self.matvec_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label_prefix}-encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label_prefix}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (output_len as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &y_buffer,
            0,
            &staging_buffer,
            0,
            byte_len(output_len, std::mem::size_of::<f32>())?,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let map_result = rx
            .recv()
            .map_err(|err| RustGptError::Gpu(format!("failed waiting for GPU readback: {err}")))?;
        map_result.map_err(|err| RustGptError::Gpu(format!("failed mapping GPU buffer: {err}")))?;

        let data = slice.get_mapped_range();
        let out = bytes_to_f32(&data)?;
        drop(data);
        staging_buffer.unmap();
        Ok(out)
    }

    fn apply_adam(
        &self,
        matrix: &GpuMatrix,
        lr_t: f32,
        beta1: f32,
        beta2: f32,
        eps_adam: f32,
        optimizer_step_num: usize,
    ) -> Result<()> {
        let len = matrix.rows * matrix.cols;
        let max_elements_per_dispatch = (u16::MAX as usize) * (WORKGROUP_SIZE as usize);

        for offset in (0..len).step_by(max_elements_per_dispatch) {
            let chunk_len = usize::min(max_elements_per_dispatch, len - offset);
            let params_bytes = adam_params_to_bytes(
                &[len as u32, optimizer_step_num as u32, offset as u32, 0_u32],
                &[lr_t, beta1, beta2, eps_adam],
            );
            let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rustgpt-adam-params"),
                size: u64::try_from(params_bytes.len())
                    .map_err(|_| RustGptError::Gpu("buffer size overflow".to_string()))?,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&params_buffer, 0, &params_bytes);

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rustgpt-adam-bind-group"),
                layout: &self.adam_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: matrix.weight_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: matrix.m_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: matrix.v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: matrix.grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rustgpt-adam-encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rustgpt-adam-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.adam_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = (chunk_len as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
        Ok(())
    }

    fn accumulate_outer_product(&self, x: &[f32], dout: &[f32], matrix: &GpuMatrix) -> Result<()> {
        if x.len() != matrix.cols {
            return Err(RustGptError::Gpu(format!(
                "outer-product input shape mismatch: {} vs {}",
                x.len(),
                matrix.cols
            )));
        }
        if dout.len() != matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "outer-product output shape mismatch: {} vs {}",
                dout.len(),
                matrix.rows
            )));
        }

        let x_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-outer-product-input"),
            size: byte_len(x.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&x_buffer, 0, &f32s_to_bytes(x));
        let dout_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-outer-product-dout"),
            size: byte_len(dout.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&dout_buffer, 0, &f32s_to_bytes(dout));
        let params = [matrix.rows as u32, matrix.cols as u32, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-outer-product-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-outer-product-bind-group"),
            layout: &self.matvec_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dout_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix.grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-outer-product-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-outer-product-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.outer_product_accum_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (matrix.cols as u32).div_ceil(8);
            let workgroups_y = (matrix.rows as u32).div_ceil(8);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn accumulate_row_grad(&self, row_idx: usize, grad: &[f32], matrix: &GpuMatrix) -> Result<()> {
        if row_idx >= matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "row grad index {row_idx} out of range for {} rows",
                matrix.rows
            )));
        }
        if grad.len() != matrix.cols {
            return Err(RustGptError::Gpu(format!(
                "row grad shape mismatch: {} vs {}",
                grad.len(),
                matrix.cols
            )));
        }

        let grad_input = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-row-grad-input"),
            size: byte_len(grad.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&grad_input, 0, &f32s_to_bytes(grad));
        let params = [row_idx as u32, matrix.cols as u32, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-row-grad-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-row-grad-bind-group"),
            layout: &self.gather_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix.grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-row-grad-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-row-grad-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.row_add_accum_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (matrix.cols as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn download_parameter(&self, gpu_matrix: &GpuMatrix, matrix: &mut Matrix) -> Result<()> {
        matrix.data = self.readback_f32_buffer(&gpu_matrix.weight_buffer, matrix.data.len())?;
        matrix.grad = self.readback_f32_buffer(&gpu_matrix.grad_buffer, matrix.grad.len())?;
        matrix.m = self.readback_f32_buffer(&gpu_matrix.m_buffer, matrix.m.len())?;
        matrix.v = self.readback_f32_buffer(&gpu_matrix.v_buffer, matrix.v.len())?;
        Ok(())
    }

    fn readback_f32_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> Result<Vec<f32>> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-readback-staging"),
            size: byte_len(len, std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            byte_len(len, std::mem::size_of::<f32>())?,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let map_result = rx
            .recv()
            .map_err(|err| RustGptError::Gpu(format!("failed waiting for GPU readback: {err}")))?;
        map_result.map_err(|err| RustGptError::Gpu(format!("failed mapping GPU buffer: {err}")))?;

        let data = slice.get_mapped_range();
        let out = bytes_to_f32(&data)?;
        drop(data);
        staging_buffer.unmap();
        Ok(out)
    }

    fn create_storage_buffer(
        &self,
        len: usize,
        label: &str,
        extra_usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer> {
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len(len, std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | extra_usage,
            mapped_at_creation: false,
        }))
    }

    fn readback_vector(&self, vector: &GpuVector) -> Result<Vec<f32>> {
        self.readback_f32_buffer(&vector.buffer, vector.len)
    }

    fn store_training_sequence(&self, sequence: GpuTrainingSequenceCache) -> DeviceSequenceHandle {
        let next = self.next_training_sequence_id.get();
        self.next_training_sequence_id.set(next + 1);
        self.training_sequences.borrow_mut().insert(next, sequence);
        DeviceSequenceHandle(next)
    }

    fn take_training_sequence(
        &self,
        handle: DeviceSequenceHandle,
    ) -> Option<GpuTrainingSequenceCache> {
        self.training_sequences.borrow_mut().remove(&handle.0)
    }

    fn upload_vector(&self, values: &[f32], label: &str) -> Result<GpuVector> {
        let out = self.create_storage_buffer(values.len(), label, wgpu::BufferUsages::empty())?;
        self.queue.write_buffer(&out, 0, &f32s_to_bytes(values));
        Ok(GpuVector {
            len: values.len(),
            buffer: out,
        })
    }

    fn upload_u32_vector(&self, values: &[u32], label: &str) -> Result<GpuIndexVector> {
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len(values.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&out, 0, &u32s_to_bytes(values));
        Ok(GpuIndexVector {
            len: values.len(),
            buffer: out,
        })
    }

    fn zeroed_vector(&self, len: usize, label: &str) -> Result<GpuVector> {
        let out = self.create_storage_buffer(len, label, wgpu::BufferUsages::empty())?;
        let zeros = vec![0.0; len];
        self.queue.write_buffer(&out, 0, &f32s_to_bytes(&zeros));
        Ok(GpuVector { len, buffer: out })
    }

    fn copy_vector(&self, vector: &GpuVector, label: &str) -> Result<GpuVector> {
        let out = self.create_storage_buffer(vector.len, label, wgpu::BufferUsages::empty())?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-copy-vector-encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &vector.buffer,
            0,
            &out,
            0,
            byte_len(vector.len, std::mem::size_of::<f32>())?,
        );
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: vector.len,
            buffer: out,
        })
    }

    fn create_kv_cache(
        &self,
        capacity: usize,
        n_embd: usize,
        layer_idx: usize,
    ) -> Result<GpuLayerCache> {
        let len = capacity
            .checked_mul(n_embd)
            .ok_or_else(|| RustGptError::Gpu("kv cache size overflow".to_string()))?;
        Ok(GpuLayerCache {
            keys: self.create_storage_buffer(
                len,
                &format!("rustgpt-layer{layer_idx}-keys"),
                wgpu::BufferUsages::empty(),
            )?,
            values: self.create_storage_buffer(
                len,
                &format!("rustgpt-layer{layer_idx}-values"),
                wgpu::BufferUsages::empty(),
            )?,
            len: 0,
            capacity,
            n_embd,
        })
    }

    fn append_to_cache(
        &self,
        cache: &mut GpuLayerCache,
        key: &GpuVector,
        value: &GpuVector,
    ) -> Result<()> {
        if key.len != cache.n_embd || value.len != cache.n_embd {
            return Err(RustGptError::Gpu(format!(
                "kv cache append shape mismatch: expected {}, got key={} value={}",
                cache.n_embd, key.len, value.len
            )));
        }
        if cache.len >= cache.capacity {
            return Err(RustGptError::Gpu("kv cache capacity exceeded".to_string()));
        }
        let offset = byte_len(cache.len * cache.n_embd, std::mem::size_of::<f32>())?;
        let copy_len = byte_len(cache.n_embd, std::mem::size_of::<f32>())?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-kv-append-encoder"),
            });
        encoder.copy_buffer_to_buffer(&key.buffer, 0, &cache.keys, offset, copy_len);
        encoder.copy_buffer_to_buffer(&value.buffer, 0, &cache.values, offset, copy_len);
        self.queue.submit(Some(encoder.finish()));
        cache.len += 1;
        Ok(())
    }

    fn gather_row(&self, matrix: &GpuMatrix, row_idx: usize) -> Result<GpuVector> {
        if row_idx >= matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "gather_row index {row_idx} out of range for {} rows",
                matrix.rows
            )));
        }
        let out = self.create_storage_buffer(
            matrix.cols,
            "rustgpt-gather-row-output",
            wgpu::BufferUsages::empty(),
        )?;
        let params = [row_idx as u32, matrix.cols as u32, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-gather-row-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-gather-row-bind-group"),
            layout: &self.gather_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix.weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-gather-row-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-gather-row-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather_row_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (matrix.cols as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: matrix.cols,
            buffer: out,
        })
    }

    fn add(&self, left: &GpuVector, right: &GpuVector) -> Result<GpuVector> {
        self.run_binary_vector_op(
            left,
            right,
            &self.add_pipeline,
            "rustgpt-add",
            WORKGROUP_SIZE,
        )
    }

    fn relu(&self, x: &GpuVector) -> Result<GpuVector> {
        self.run_unary_vector_op(
            x,
            &self.relu_pipeline,
            "rustgpt-relu",
            &[x.len as u32, 0, 0, 0],
            WORKGROUP_SIZE,
        )
    }

    fn rmsnorm(&self, x: &GpuVector) -> Result<GpuVector> {
        self.run_unary_vector_op(
            x,
            &self.rmsnorm_pipeline,
            "rustgpt-rmsnorm",
            &[x.len as u32, 0, 0, 0],
            1,
        )
    }

    fn softmax_rows(&self, x: &GpuVector, rows: usize, cols: usize) -> Result<GpuVector> {
        if x.len != rows * cols {
            return Err(RustGptError::Gpu(format!(
                "softmax_rows shape mismatch: len={} rows={} cols={}",
                x.len, rows, cols
            )));
        }
        self.run_unary_vector_op(
            x,
            &self.softmax_rows_pipeline,
            "rustgpt-softmax-rows",
            &[rows as u32, cols as u32, 0, 0],
            rows as u32,
        )
    }

    fn rmsnorm_rows(&self, x: &GpuVector, rows: usize, cols: usize) -> Result<GpuVector> {
        if x.len != rows * cols {
            return Err(RustGptError::Gpu(format!(
                "rmsnorm_rows shape mismatch: len={} rows={} cols={}",
                x.len, rows, cols
            )));
        }
        self.run_unary_vector_op(
            x,
            &self.rmsnorm_rows_pipeline,
            "rustgpt-rmsnorm-rows",
            &[rows as u32, cols as u32, 0, 0],
            rows as u32,
        )
    }

    fn rmsnorm_backward_rows(
        &self,
        x: &GpuVector,
        dy: &GpuVector,
        rows: usize,
        cols: usize,
    ) -> Result<GpuVector> {
        if x.len != rows * cols || dy.len != rows * cols {
            return Err(RustGptError::Gpu(format!(
                "rmsnorm_backward_rows shape mismatch: x={} dy={} rows={} cols={}",
                x.len, dy.len, rows, cols
            )));
        }
        self.run_binary_vector_op_with_params(
            x,
            dy,
            &self.rmsnorm_backward_rows_pipeline,
            "rustgpt-rmsnorm-backward-rows",
            &u32s_to_bytes(&[rows as u32, cols as u32, 0, 0]),
            rows as u32,
        )
    }

    fn outer_product_rows_accum(
        &self,
        x: &GpuVector,
        dout: &GpuVector,
        batch_rows: usize,
        matrix: &GpuMatrix,
    ) -> Result<()> {
        if x.len != batch_rows * matrix.cols {
            return Err(RustGptError::Gpu(format!(
                "outer_product_rows_accum input shape mismatch: len={} batch_rows={} cols={}",
                x.len, batch_rows, matrix.cols
            )));
        }
        if dout.len != batch_rows * matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "outer_product_rows_accum dout shape mismatch: len={} batch_rows={} rows={}",
                dout.len, batch_rows, matrix.rows
            )));
        }
        let params = [
            batch_rows as u32,
            matrix.rows as u32,
            matrix.cols as u32,
            0_u32,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-outer-product-rows-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-outer-product-rows-bind-group"),
            layout: &self.matvec_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dout.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix.grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-outer-product-rows-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-outer-product-rows-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.outer_product_rows_accum_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (matrix.cols as u32).div_ceil(8);
            let wg_y = (matrix.rows as u32).div_ceil(8);
            pass.dispatch_workgroups(wg_x.max(1), wg_y.max(1), 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn attn_scores(
        &self,
        q: &GpuVector,
        keys_buffer: &wgpu::Buffer,
        time_len: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Result<GpuVector> {
        let out_len = time_len
            .checked_mul(n_head)
            .ok_or_else(|| RustGptError::Gpu("attention score size overflow".to_string()))?;
        let out = self.create_storage_buffer(
            out_len,
            "rustgpt-attn-scores-output",
            wgpu::BufferUsages::empty(),
        )?;
        let params = [time_len as u32, n_head as u32, head_dim as u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-attn-scores-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-attn-scores-bind-group"),
            layout: &self.attn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-attn-scores-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-attn-scores-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_scores_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: out_len,
            buffer: out,
        })
    }

    fn attn_values(
        &self,
        weights: &GpuVector,
        values_buffer: &wgpu::Buffer,
        time_len: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Result<GpuVector> {
        let out_len = n_head
            .checked_mul(head_dim)
            .ok_or_else(|| RustGptError::Gpu("attention value size overflow".to_string()))?;
        let out = self.create_storage_buffer(
            out_len,
            "rustgpt-attn-values-output",
            wgpu::BufferUsages::empty(),
        )?;
        let params = [time_len as u32, n_head as u32, head_dim as u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-attn-values-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-attn-values-bind-group"),
            layout: &self.attn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-attn-values-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-attn-values-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_values_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: out_len,
            buffer: out,
        })
    }

    fn attn_scores_seq(
        &self,
        q: &GpuVector,
        k: &GpuVector,
        seq_len: usize,
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> Result<GpuVector> {
        let out_len = seq_len
            .checked_mul(n_head)
            .and_then(|v| v.checked_mul(seq_len))
            .ok_or_else(|| RustGptError::Gpu("attention score seq size overflow".to_string()))?;
        let out = self.create_storage_buffer(
            out_len,
            "rustgpt-attn-scores-seq-output",
            wgpu::BufferUsages::empty(),
        )?;
        let params = [
            seq_len as u32,
            n_head as u32,
            head_dim as u32,
            n_embd as u32,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-attn-scores-seq-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-attn-scores-seq-bind-group"),
            layout: &self.attn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-attn-scores-seq-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-attn-scores-seq-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_scores_seq_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((seq_len * n_head) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: out_len,
            buffer: out,
        })
    }

    fn attn_values_seq(
        &self,
        weights: &GpuVector,
        values: &GpuVector,
        seq_len: usize,
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> Result<GpuVector> {
        let out_len = seq_len
            .checked_mul(n_embd)
            .ok_or_else(|| RustGptError::Gpu("attention value seq size overflow".to_string()))?;
        let out = self.create_storage_buffer(
            out_len,
            "rustgpt-attn-values-seq-output",
            wgpu::BufferUsages::empty(),
        )?;
        let params = [
            seq_len as u32,
            n_head as u32,
            head_dim as u32,
            n_embd as u32,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-attn-values-seq-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-attn-values-seq-bind-group"),
            layout: &self.attn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-attn-values-seq-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-attn-values-seq-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_values_seq_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((seq_len * n_head) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: out_len,
            buffer: out,
        })
    }

    fn attn_scores_batch(
        &self,
        q: &GpuVector,
        k: &GpuVector,
        batch_size: usize,
        seq_len: usize,
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> Result<GpuVector> {
        let total_rows = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| RustGptError::Gpu("batched attention row size overflow".to_string()))?;
        let expected_len = total_rows.checked_mul(n_embd).ok_or_else(|| {
            RustGptError::Gpu("batched attention input size overflow".to_string())
        })?;
        if q.len != expected_len || k.len != expected_len {
            return Err(RustGptError::Gpu(format!(
                "batched attention score shape mismatch: q={} k={} expected={expected_len}",
                q.len, k.len
            )));
        }
        let out_len = total_rows
            .checked_mul(n_head)
            .and_then(|v| v.checked_mul(seq_len))
            .ok_or_else(|| {
                RustGptError::Gpu("batched attention score size overflow".to_string())
            })?;
        let out = self.create_storage_buffer(
            out_len,
            "rustgpt-attn-scores-batch-output",
            wgpu::BufferUsages::empty(),
        )?;
        let params = [
            batch_size as u32,
            seq_len as u32,
            n_head as u32,
            head_dim as u32,
            n_embd as u32,
            0_u32,
            0_u32,
            0_u32,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-attn-scores-batch-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-attn-scores-batch-bind-group"),
            layout: &self.attn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-attn-scores-batch-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-attn-scores-batch-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_scores_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((total_rows * n_head) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: out_len,
            buffer: out,
        })
    }

    fn attn_values_batch(
        &self,
        weights: &GpuVector,
        values: &GpuVector,
        batch_size: usize,
        seq_len: usize,
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> Result<GpuVector> {
        let total_rows = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| RustGptError::Gpu("batched attention row size overflow".to_string()))?;
        let expected_values = total_rows.checked_mul(n_embd).ok_or_else(|| {
            RustGptError::Gpu("batched attention value size overflow".to_string())
        })?;
        let expected_weights = total_rows
            .checked_mul(n_head)
            .and_then(|v| v.checked_mul(seq_len))
            .ok_or_else(|| {
                RustGptError::Gpu("batched attention weight size overflow".to_string())
            })?;
        if values.len != expected_values || weights.len != expected_weights {
            return Err(RustGptError::Gpu(format!(
                "batched attention value shape mismatch: weights={} expected_weights={} values={} expected_values={}",
                weights.len, expected_weights, values.len, expected_values
            )));
        }
        let out = self.create_storage_buffer(
            expected_values,
            "rustgpt-attn-values-batch-output",
            wgpu::BufferUsages::empty(),
        )?;
        let params = [
            batch_size as u32,
            seq_len as u32,
            n_head as u32,
            head_dim as u32,
            n_embd as u32,
            0_u32,
            0_u32,
            0_u32,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-attn-values-batch-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-attn-values-batch-bind-group"),
            layout: &self.attn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-attn-values-batch-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-attn-values-batch-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_values_batch_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((total_rows * n_head) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: expected_values,
            buffer: out,
        })
    }

    fn attention_backward(
        &self,
        d_attn: &GpuVector,
        q: &GpuVector,
        weights: &GpuVector,
        keys_buffer: &wgpu::Buffer,
        values_buffer: &wgpu::Buffer,
        d_q: &GpuVector,
        d_keys: &GpuVector,
        d_values: &GpuVector,
        query: usize,
        time_len: usize,
        seq_len: usize,
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> Result<()> {
        self.attention_backward_range(
            d_attn,
            0,
            q,
            0,
            weights,
            0,
            keys_buffer,
            0,
            values_buffer,
            0,
            d_q,
            0,
            d_keys,
            0,
            d_values,
            0,
            query,
            time_len,
            seq_len,
            n_head,
            head_dim,
            n_embd,
        )
    }

    fn attention_backward_batch_sequence(
        &self,
        d_attn: &GpuVector,
        q: &GpuVector,
        weights: &GpuVector,
        keys: &GpuVector,
        values: &GpuVector,
        d_q: &GpuVector,
        d_keys: &GpuVector,
        d_values: &GpuVector,
        batch_idx: usize,
        seq_len: usize,
        query: usize,
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> Result<()> {
        let vector_rows = batch_idx
            .checked_mul(seq_len)
            .ok_or_else(|| RustGptError::Gpu("batched attention offset overflow".to_string()))?;
        let vector_offset = vector_rows.checked_mul(n_embd).ok_or_else(|| {
            RustGptError::Gpu("batched attention vector offset overflow".to_string())
        })?;
        let weights_offset = batch_idx
            .checked_mul(seq_len)
            .and_then(|v| v.checked_mul(n_head))
            .and_then(|v| v.checked_mul(seq_len))
            .ok_or_else(|| {
                RustGptError::Gpu("batched attention weight offset overflow".to_string())
            })?;
        self.attention_backward_range(
            d_attn,
            vector_offset,
            q,
            vector_offset,
            weights,
            weights_offset,
            &keys.buffer,
            vector_offset,
            &values.buffer,
            vector_offset,
            d_q,
            vector_offset,
            d_keys,
            vector_offset,
            d_values,
            vector_offset,
            query,
            query + 1,
            seq_len,
            n_head,
            head_dim,
            n_embd,
        )
    }

    fn attention_backward_range(
        &self,
        d_attn: &GpuVector,
        d_attn_offset: usize,
        q: &GpuVector,
        q_offset: usize,
        weights: &GpuVector,
        weights_offset: usize,
        keys_buffer: &wgpu::Buffer,
        keys_offset: usize,
        values_buffer: &wgpu::Buffer,
        values_offset: usize,
        d_q: &GpuVector,
        d_q_offset: usize,
        d_keys: &GpuVector,
        d_keys_offset: usize,
        d_values: &GpuVector,
        d_values_offset: usize,
        query: usize,
        time_len: usize,
        seq_len: usize,
        n_head: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> Result<()> {
        if seq_len > GPU_ATTN_BACKWARD_MAX_SEQ {
            return Err(RustGptError::Gpu(format!(
                "GPU attention backward supports seq_len <= {GPU_ATTN_BACKWARD_MAX_SEQ}, got {seq_len}"
            )));
        }
        let vector_len = seq_len.checked_mul(n_embd).ok_or_else(|| {
            RustGptError::Gpu("attention backward vector size overflow".to_string())
        })?;
        let weights_len = seq_len
            .checked_mul(n_head)
            .and_then(|v| v.checked_mul(seq_len))
            .ok_or_else(|| {
                RustGptError::Gpu("attention backward weight size overflow".to_string())
            })?;
        let params = [
            query as u32,
            time_len as u32,
            seq_len as u32,
            n_head as u32,
            head_dim as u32,
            n_embd as u32,
            0_u32,
            0_u32,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-attn-backward-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-attn-backward-bind-group"),
            layout: &self.attn_backward_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_f32_range(&d_attn.buffer, d_attn_offset, vector_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_f32_range(&q.buffer, q_offset, vector_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_f32_range(&weights.buffer, weights_offset, weights_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_f32_range(keys_buffer, keys_offset, vector_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_f32_range(values_buffer, values_offset, vector_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffer_f32_range(&d_q.buffer, d_q_offset, vector_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffer_f32_range(&d_keys.buffer, d_keys_offset, vector_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffer_f32_range(&d_values.buffer, d_values_offset, vector_len)?,
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-attn-backward-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-attn-backward-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.attn_backward_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_head as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn scatter_embed_grads(
        &self,
        token_ids: &GpuIndexVector,
        d_embed: &GpuVector,
        wte: &GpuMatrix,
        wpe: &GpuMatrix,
        seq_len: usize,
        n_embd: usize,
    ) -> Result<()> {
        if token_ids.len != seq_len {
            return Err(RustGptError::Gpu(format!(
                "token id length mismatch: {} vs {}",
                token_ids.len, seq_len
            )));
        }
        if d_embed.len != seq_len * n_embd {
            return Err(RustGptError::Gpu(format!(
                "embed grad shape mismatch: {} vs {}",
                d_embed.len,
                seq_len * n_embd
            )));
        }

        let params = [seq_len as u32, n_embd as u32, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-scatter-embed-grads-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-scatter-embed-grads-bind-group"),
            layout: &self.scatter_embed_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_ids.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: d_embed.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wte.grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wpe.grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-scatter-embed-grads-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-scatter-embed-grads-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_embed_grads_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn cross_entropy_rows(
        &self,
        probs: &GpuVector,
        target_ids: &GpuIndexVector,
        rows: usize,
        cols: usize,
        norm: f32,
    ) -> Result<(GpuVector, GpuVector)> {
        if probs.len != rows * cols {
            return Err(RustGptError::Gpu(format!(
                "cross entropy shape mismatch: probs={} rows={} cols={}",
                probs.len, rows, cols
            )));
        }
        if target_ids.len != rows {
            return Err(RustGptError::Gpu(format!(
                "cross entropy target mismatch: {} vs {}",
                target_ids.len, rows
            )));
        }

        let d_logits = self.create_storage_buffer(
            probs.len,
            "rustgpt-cross-entropy-dlogits",
            wgpu::BufferUsages::empty(),
        )?;
        let losses = self.create_storage_buffer(
            rows,
            "rustgpt-cross-entropy-losses",
            wgpu::BufferUsages::empty(),
        )?;
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-cross-entropy-params"),
            size: byte_len(8, std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &params_buffer,
            0,
            &adam_params_to_bytes(
                &[rows as u32, cols as u32, 0_u32, 0_u32],
                &[norm, 0.0, 0.0, 0.0],
            ),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-cross-entropy-bind-group"),
            layout: &self.loss_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: probs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_ids.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: d_logits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: losses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-cross-entropy-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-cross-entropy-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cross_entropy_rows_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok((
            GpuVector {
                len: probs.len,
                buffer: d_logits,
            },
            GpuVector {
                len: rows,
                buffer: losses,
            },
        ))
    }

    fn matvec_vector(&self, x: &GpuVector, matrix: &GpuMatrix) -> Result<GpuVector> {
        if x.len != matrix.cols {
            return Err(RustGptError::Gpu(format!(
                "GPU matvec shape mismatch: input has {} elements, matrix expects {}",
                x.len, matrix.cols
            )));
        }
        self.run_matvec_vector(
            x,
            matrix.rows,
            matrix.rows as u32,
            matrix.cols as u32,
            &matrix.weight_buffer,
            &self.matvec_pipeline,
            "rustgpt-matvec-device",
        )
    }

    fn matvec_transposed_vector(&self, x: &GpuVector, matrix: &GpuMatrix) -> Result<GpuVector> {
        if x.len != matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "GPU matvec_transposed shape mismatch: input has {} elements, matrix expects {} rows",
                x.len, matrix.rows
            )));
        }
        self.run_matvec_vector(
            x,
            matrix.cols,
            matrix.rows as u32,
            matrix.cols as u32,
            &matrix.weight_buffer,
            &self.matvec_transposed_pipeline,
            "rustgpt-matvec-transposed-device",
        )
    }

    fn matmul_rows(
        &self,
        x: &GpuVector,
        batch_rows: usize,
        matrix: &GpuMatrix,
    ) -> Result<GpuVector> {
        if x.len != batch_rows * matrix.cols {
            return Err(RustGptError::Gpu(format!(
                "GPU matmul_rows shape mismatch: input len={} batch_rows={} matrix.cols={}",
                x.len, batch_rows, matrix.cols
            )));
        }
        self.run_matmul_rows_vector(
            x,
            batch_rows,
            matrix.rows,
            matrix.cols,
            &matrix.weight_buffer,
            &self.matmul_rows_pipeline,
            "rustgpt-matmul-rows",
        )
    }

    fn matmul_rows_transposed(
        &self,
        x: &GpuVector,
        batch_rows: usize,
        matrix: &GpuMatrix,
    ) -> Result<GpuVector> {
        if x.len != batch_rows * matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "GPU matmul_rows_transposed shape mismatch: input len={} batch_rows={} matrix.rows={}",
                x.len, batch_rows, matrix.rows
            )));
        }
        self.run_matmul_rows_vector(
            x,
            batch_rows,
            matrix.cols,
            matrix.rows,
            &matrix.weight_buffer,
            &self.matmul_rows_transposed_pipeline,
            "rustgpt-matmul-rows-transposed",
        )
    }

    fn run_matmul_rows_vector(
        &self,
        x: &GpuVector,
        batch_rows: usize,
        out_cols: usize,
        in_cols: usize,
        weight_buffer: &wgpu::Buffer,
        pipeline: &wgpu::ComputePipeline,
        label_prefix: &str,
    ) -> Result<GpuVector> {
        let out_len = batch_rows
            .checked_mul(out_cols)
            .ok_or_else(|| RustGptError::Gpu("batched matmul output size overflow".to_string()))?;
        let out = self.create_storage_buffer(out_len, label_prefix, wgpu::BufferUsages::empty())?;
        let params = [batch_rows as u32, out_cols as u32, in_cols as u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-params")),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label_prefix}-bind-group")),
            layout: &self.matvec_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label_prefix}-encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label_prefix}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (out_cols as u32).div_ceil(8);
            let wg_y = (batch_rows as u32).div_ceil(8);
            pass.dispatch_workgroups(wg_x.max(1), wg_y.max(1), 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: out_len,
            buffer: out,
        })
    }

    fn run_matvec_vector(
        &self,
        x: &GpuVector,
        output_len: usize,
        rows: u32,
        cols: u32,
        weight_buffer: &wgpu::Buffer,
        pipeline: &wgpu::ComputePipeline,
        label_prefix: &str,
    ) -> Result<GpuVector> {
        let out =
            self.create_storage_buffer(output_len, label_prefix, wgpu::BufferUsages::empty())?;
        let params = [rows, cols, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-params")),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label_prefix}-bind-group")),
            layout: &self.matvec_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label_prefix}-encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label_prefix}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (output_len as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: output_len,
            buffer: out,
        })
    }

    fn run_unary_vector_op(
        &self,
        x: &GpuVector,
        pipeline: &wgpu::ComputePipeline,
        label_prefix: &str,
        params: &[u32; 4],
        dispatch_x: u32,
    ) -> Result<GpuVector> {
        let out = self.create_storage_buffer(x.len, label_prefix, wgpu::BufferUsages::empty())?;
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-params")),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label_prefix}-bind-group")),
            layout: &self.unary_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label_prefix}-encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label_prefix}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: x.len,
            buffer: out,
        })
    }

    fn run_binary_vector_op(
        &self,
        left: &GpuVector,
        right: &GpuVector,
        pipeline: &wgpu::ComputePipeline,
        label_prefix: &str,
        workgroup_size: u32,
    ) -> Result<GpuVector> {
        if left.len != right.len {
            return Err(RustGptError::Gpu(format!(
                "binary op shape mismatch: left={} right={}",
                left.len, right.len
            )));
        }
        let out =
            self.create_storage_buffer(left.len, label_prefix, wgpu::BufferUsages::empty())?;
        let params = [left.len as u32, 0_u32, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-params")),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label_prefix}-bind-group")),
            layout: &self.binary_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label_prefix}-encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label_prefix}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (left.len as u32).div_ceil(workgroup_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: left.len,
            buffer: out,
        })
    }

    fn run_binary_vector_op_with_params(
        &self,
        left: &GpuVector,
        right: &GpuVector,
        pipeline: &wgpu::ComputePipeline,
        label_prefix: &str,
        params_bytes: &[u8],
        dispatch_x: u32,
    ) -> Result<GpuVector> {
        if left.len != right.len {
            return Err(RustGptError::Gpu(format!(
                "binary op shape mismatch: left={} right={}",
                left.len, right.len
            )));
        }
        let out =
            self.create_storage_buffer(left.len, label_prefix, wgpu::BufferUsages::empty())?;
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}-params")),
            size: u64::try_from(params_bytes.len())
                .map_err(|_| RustGptError::Gpu("buffer size overflow".to_string()))?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, params_bytes);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label_prefix}-bind-group")),
            layout: &self.binary_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label_prefix}-encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label_prefix}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuVector {
            len: left.len,
            buffer: out,
        })
    }

    fn relu_backward_vector(
        &self,
        pre_activation: &GpuVector,
        dout: &GpuVector,
    ) -> Result<GpuVector> {
        self.run_binary_vector_op_with_params(
            pre_activation,
            dout,
            &self.relu_backward_pipeline,
            "rustgpt-relu-backward",
            &u32s_to_bytes(&[pre_activation.len as u32, 0, 0, 0]),
            (pre_activation.len as u32).div_ceil(WORKGROUP_SIZE),
        )
    }

    fn rmsnorm_backward_vector(
        &self,
        x: &GpuVector,
        rms_inv: f32,
        dy: &GpuVector,
    ) -> Result<GpuVector> {
        self.run_binary_vector_op_with_params(
            x,
            dy,
            &self.rmsnorm_backward_pipeline,
            "rustgpt-rmsnorm-backward",
            &len_scalar_params_to_bytes(x.len as u32, rms_inv),
            1,
        )
    }

    fn accumulate_outer_product_from_vector(
        &self,
        x: &[f32],
        dout: &GpuVector,
        matrix: &GpuMatrix,
    ) -> Result<()> {
        if x.len() != matrix.cols {
            return Err(RustGptError::Gpu(format!(
                "outer-product input shape mismatch: {} vs {}",
                x.len(),
                matrix.cols
            )));
        }
        if dout.len != matrix.rows {
            return Err(RustGptError::Gpu(format!(
                "outer-product output shape mismatch: {} vs {}",
                dout.len, matrix.rows
            )));
        }
        let x_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-outer-product-input"),
            size: byte_len(x.len(), std::mem::size_of::<f32>())?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&x_buffer, 0, &f32s_to_bytes(x));
        let params = [matrix.rows as u32, matrix.cols as u32, 0_u32, 0_u32];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustgpt-outer-product-params"),
            size: byte_len(params.len(), std::mem::size_of::<u32>())?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, &u32s_to_bytes(&params));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustgpt-outer-product-bind-group"),
            layout: &self.matvec_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dout.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix.grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rustgpt-outer-product-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustgpt-outer-product-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.outer_product_accum_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (matrix.cols as u32).div_ceil(8);
            let workgroups_y = (matrix.rows as u32).div_ceil(8);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn backward_mlp_residual(
        &self,
        x_residual: &[f32],
        x_norm: &[f32],
        rms_inv: f32,
        mlp_hidden_pre: &[f32],
        mlp_hidden_act: &[f32],
        d_mlp_out: &[f32],
        mlp_fc1: &GpuMatrix,
        mlp_fc2: &GpuMatrix,
    ) -> Result<Vec<f32>> {
        let d_out = self.upload_vector(d_mlp_out, "rustgpt-mlp-dout")?;
        self.accumulate_outer_product_from_vector(mlp_hidden_act, &d_out, mlp_fc2)?;
        let d_hidden_act = self.matvec_transposed_vector(&d_out, mlp_fc2)?;
        let pre = self.upload_vector(mlp_hidden_pre, "rustgpt-mlp-pre")?;
        let d_hidden_pre = self.relu_backward_vector(&pre, &d_hidden_act)?;
        self.accumulate_outer_product_from_vector(x_norm, &d_hidden_pre, mlp_fc1)?;
        let d_x_norm = self.matvec_transposed_vector(&d_hidden_pre, mlp_fc1)?;
        let x_residual_vec = self.upload_vector(x_residual, "rustgpt-mlp-residual")?;
        let d_x = self.rmsnorm_backward_vector(&x_residual_vec, rms_inv, &d_x_norm)?;
        self.readback_vector(&d_x)
    }

    fn backward_attention_projections(
        &self,
        x_residual: &[f32],
        x_norm: &[f32],
        rms_inv: f32,
        d_q: &[f32],
        d_k: &[f32],
        d_v: &[f32],
        attn_wq: &GpuMatrix,
        attn_wk: &GpuMatrix,
        attn_wv: &GpuMatrix,
    ) -> Result<Vec<f32>> {
        let d_q_vec = self.upload_vector(d_q, "rustgpt-attn-dq")?;
        let d_k_vec = self.upload_vector(d_k, "rustgpt-attn-dk")?;
        let d_v_vec = self.upload_vector(d_v, "rustgpt-attn-dv")?;
        self.accumulate_outer_product_from_vector(x_norm, &d_q_vec, attn_wq)?;
        self.accumulate_outer_product_from_vector(x_norm, &d_k_vec, attn_wk)?;
        self.accumulate_outer_product_from_vector(x_norm, &d_v_vec, attn_wv)?;
        let d_x_norm_q = self.matvec_transposed_vector(&d_q_vec, attn_wq)?;
        let d_x_norm_k = self.matvec_transposed_vector(&d_k_vec, attn_wk)?;
        let d_x_norm_v = self.matvec_transposed_vector(&d_v_vec, attn_wv)?;
        let d_x_norm_qk = self.add(&d_x_norm_q, &d_x_norm_k)?;
        let d_x_norm = self.add(&d_x_norm_qk, &d_x_norm_v)?;
        let x_residual_vec = self.upload_vector(x_residual, "rustgpt-attn-residual")?;
        let d_x = self.rmsnorm_backward_vector(&x_residual_vec, rms_inv, &d_x_norm)?;
        self.readback_vector(&d_x)
    }
}

fn lookup_parameter(
    matrices: &HashMap<ParameterId, GpuMatrix>,
    parameter_id: ParameterId,
) -> Result<&GpuMatrix> {
    matrices.get(&parameter_id).ok_or_else(|| {
        RustGptError::Gpu(format!(
            "GPU runtime is missing parameter {}",
            parameter_id.label()
        ))
    })
}

fn request_adapter(instance: &wgpu::Instance, device_kind: DeviceKind) -> Result<wgpu::Adapter> {
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| {
        RustGptError::Gpu(format!(
            "wgpu could not find a compatible adapter for {} mode",
            device_kind
        ))
    })
}
