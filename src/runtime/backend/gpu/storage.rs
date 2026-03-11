//! Device-resident buffers and cached activations used by the GPU backend.

pub(super) struct GpuMatrix {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) weight_buffer: wgpu::Buffer,
    pub(super) grad_buffer: wgpu::Buffer,
    pub(super) m_buffer: wgpu::Buffer,
    pub(super) v_buffer: wgpu::Buffer,
}

pub(super) struct GpuVector {
    pub(super) len: usize,
    pub(super) buffer: wgpu::Buffer,
}

pub(super) struct GpuIndexVector {
    pub(super) len: usize,
    pub(super) buffer: wgpu::Buffer,
}

pub(super) struct GpuTrainingSequenceCache {
    pub(super) seq_len: usize,
    pub(super) token_ids: GpuIndexVector,
    pub(super) embed_sum: GpuVector,
    pub(super) final_x: GpuVector,
    pub(super) final_norm_x: GpuVector,
    pub(super) d_logits: GpuVector,
    pub(super) layers: Vec<GpuTrainingLayerCache>,
}

pub(super) struct GpuTrainingLayerCache {
    pub(super) x_residual_attn: GpuVector,
    pub(super) x_norm_attn: GpuVector,
    pub(super) q: GpuVector,
    pub(super) k: GpuVector,
    pub(super) v: GpuVector,
    pub(super) attn_weights: GpuVector,
    pub(super) x_attn: GpuVector,
    pub(super) x_residual_mlp: GpuVector,
    pub(super) x_norm_mlp: GpuVector,
    pub(super) mlp_hidden_pre: GpuVector,
    pub(super) mlp_gate_pre: Option<GpuVector>,
    pub(super) mlp_hidden_act: GpuVector,
}

pub(super) struct GpuLayerCache {
    pub(super) keys: wgpu::Buffer,
    pub(super) values: wgpu::Buffer,
    pub(super) len: usize,
    pub(super) capacity: usize,
    pub(super) n_embd: usize,
}
