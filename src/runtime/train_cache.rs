use crate::runtime::workspace::AttentionWeights;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceSequenceHandle(pub u64);

#[derive(Clone, Debug, PartialEq)]
pub struct LayerForwardCache {
    pub x_residual_attn: Vec<f32>,
    pub x_norm_attn: Vec<f32>,
    pub rms_inv_attn: f32,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_weights: AttentionWeights,
    pub x_attn: Vec<f32>,
    pub x_residual_mlp: Vec<f32>,
    pub x_norm_mlp: Vec<f32>,
    pub rms_inv_mlp: f32,
    pub mlp_hidden_pre: Vec<f32>,
    pub mlp_gate_pre: Option<Vec<f32>>,
    pub mlp_hidden_act: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenForwardCache {
    pub token_id: usize,
    pub target_id: usize,
    pub pos_id: usize,
    pub loss_weight: f32,
    pub embed_sum: Vec<f32>,
    pub embed_rms_inv: f32,
    pub layers: Vec<LayerForwardCache>,
    pub final_x: Vec<f32>,
    pub final_norm_x: Vec<f32>,
    pub final_rms_inv: f32,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequenceForwardCache {
    pub tokens: Vec<TokenForwardCache>,
    pub mean_loss: f32,
    pub loss_weight_sum: f32,
    pub grad_scale: f32,
    pub device_sequence: Option<DeviceSequenceHandle>,
}
