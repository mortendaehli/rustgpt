pub mod attention;
pub mod position;

use crate::core::config::{ModelConfig, PositionEncodingKind};
use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::core::tensor::Matrix;

const INIT_STD: f32 = 0.08;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ParameterId {
    Wte,
    Wpe,
    LmHead,
    AttnWq(usize),
    AttnWk(usize),
    AttnWv(usize),
    AttnWo(usize),
    MlpFc1(usize),
    MlpFcGate(usize),
    MlpFc2(usize),
}

impl ParameterId {
    pub fn label(self) -> String {
        match self {
            Self::Wte => "rustgpt-wte".to_string(),
            Self::Wpe => "rustgpt-wpe".to_string(),
            Self::LmHead => "rustgpt-lm-head".to_string(),
            Self::AttnWq(layer_idx) => format!("rustgpt-layer{layer_idx}-attn-wq"),
            Self::AttnWk(layer_idx) => format!("rustgpt-layer{layer_idx}-attn-wk"),
            Self::AttnWv(layer_idx) => format!("rustgpt-layer{layer_idx}-attn-wv"),
            Self::AttnWo(layer_idx) => format!("rustgpt-layer{layer_idx}-attn-wo"),
            Self::MlpFc1(layer_idx) => format!("rustgpt-layer{layer_idx}-mlp-fc1"),
            Self::MlpFcGate(layer_idx) => format!("rustgpt-layer{layer_idx}-mlp-fc-gate"),
            Self::MlpFc2(layer_idx) => format!("rustgpt-layer{layer_idx}-mlp-fc2"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransformerLayer {
    pub attn_wq: Matrix,
    pub attn_wk: Matrix,
    pub attn_wv: Matrix,
    pub attn_wo: Matrix,
    pub mlp_fc1: Matrix,
    pub mlp_fc_gate: Option<Matrix>,
    pub mlp_fc2: Matrix,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub cfg: ModelConfig,
    pub wte: Matrix,
    pub wpe: Option<Matrix>,
    pub lm_head: Option<Matrix>,
    pub layers: Vec<TransformerLayer>,
}

impl Model {
    pub fn new(cfg: ModelConfig, rng: &mut Rng) -> Result<Self> {
        Self::validate_config(&cfg)?;

        let wte = Matrix::from_gaussian(cfg.vocab_size, cfg.n_embd, INIT_STD, rng);
        let wpe = if cfg.position_encoding == PositionEncodingKind::LearnedAbsolute {
            Some(Matrix::from_gaussian(
                cfg.block_size,
                cfg.n_embd,
                INIT_STD,
                rng,
            ))
        } else {
            None
        };
        let lm_head = if cfg.tie_embeddings {
            None
        } else {
            Some(Matrix::from_gaussian(
                cfg.vocab_size,
                cfg.n_embd,
                INIT_STD,
                rng,
            ))
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            layers.push(TransformerLayer {
                attn_wq: Matrix::from_gaussian(cfg.n_embd, cfg.n_embd, INIT_STD, rng),
                attn_wk: Matrix::from_gaussian(cfg.kv_dim(), cfg.n_embd, INIT_STD, rng),
                attn_wv: Matrix::from_gaussian(cfg.kv_dim(), cfg.n_embd, INIT_STD, rng),
                attn_wo: Matrix::from_gaussian(cfg.n_embd, cfg.n_embd, INIT_STD, rng),
                mlp_fc1: Matrix::from_gaussian(4 * cfg.n_embd, cfg.n_embd, INIT_STD, rng),
                mlp_fc_gate: cfg
                    .activation
                    .eq(&crate::core::config::ActivationKind::SwiGlu)
                    .then(|| Matrix::from_gaussian(4 * cfg.n_embd, cfg.n_embd, INIT_STD, rng)),
                mlp_fc2: Matrix::from_gaussian(cfg.n_embd, 4 * cfg.n_embd, INIT_STD, rng),
            });
        }

        Ok(Self {
            cfg,
            wte,
            wpe,
            lm_head,
            layers,
        })
    }

    pub fn from_parts(
        cfg: ModelConfig,
        wte: Matrix,
        wpe: Option<Matrix>,
        lm_head: Option<Matrix>,
        layers: Vec<TransformerLayer>,
    ) -> Result<Self> {
        Self::validate_config(&cfg)?;
        if wte.rows != cfg.vocab_size || wte.cols != cfg.n_embd {
            return Err(RustGptError::Config(
                "wte shape does not match model config".to_string(),
            ));
        }
        if let Some(wpe) = &wpe {
            if wpe.rows != cfg.block_size || wpe.cols != cfg.n_embd {
                return Err(RustGptError::Config(
                    "wpe shape does not match model config".to_string(),
                ));
            }
        } else if cfg.position_encoding == PositionEncodingKind::LearnedAbsolute {
            return Err(RustGptError::Config(
                "learned position encoding requires wpe weights".to_string(),
            ));
        }
        if let Some(lm_head) = &lm_head {
            if lm_head.rows != cfg.vocab_size || lm_head.cols != cfg.n_embd {
                return Err(RustGptError::Config(
                    "lm_head shape does not match model config".to_string(),
                ));
            }
        } else if !cfg.tie_embeddings {
            return Err(RustGptError::Config(
                "untied model requires lm_head weights".to_string(),
            ));
        }
        if layers.len() != cfg.n_layer {
            return Err(RustGptError::Config(
                "checkpoint layer count does not match model config".to_string(),
            ));
        }
        for layer in &layers {
            if layer.attn_wk.rows != cfg.kv_dim() || layer.attn_wk.cols != cfg.n_embd {
                return Err(RustGptError::Config(
                    "attn_wk shape does not match model config".to_string(),
                ));
            }
            if layer.attn_wv.rows != cfg.kv_dim() || layer.attn_wv.cols != cfg.n_embd {
                return Err(RustGptError::Config(
                    "attn_wv shape does not match model config".to_string(),
                ));
            }
            if layer.mlp_fc1.rows != 4 * cfg.n_embd || layer.mlp_fc1.cols != cfg.n_embd {
                return Err(RustGptError::Config(
                    "mlp_fc1 shape does not match model config".to_string(),
                ));
            }
            if layer.mlp_fc2.rows != cfg.n_embd || layer.mlp_fc2.cols != 4 * cfg.n_embd {
                return Err(RustGptError::Config(
                    "mlp_fc2 shape does not match model config".to_string(),
                ));
            }
            match (&layer.mlp_fc_gate, cfg.activation) {
                (Some(gate), crate::core::config::ActivationKind::SwiGlu) => {
                    if gate.rows != 4 * cfg.n_embd || gate.cols != cfg.n_embd {
                        return Err(RustGptError::Config(
                            "mlp_fc_gate shape does not match model config".to_string(),
                        ));
                    }
                }
                (None, crate::core::config::ActivationKind::SwiGlu) => {
                    return Err(RustGptError::Config(
                        "swiglu requires mlp_fc_gate weights".to_string(),
                    ));
                }
                (Some(_), _) => {
                    return Err(RustGptError::Config(
                        "mlp_fc_gate weights require swiglu activation".to_string(),
                    ));
                }
                (None, _) => {}
            }
        }
        Ok(Self {
            cfg,
            wte,
            wpe,
            lm_head,
            layers,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.cfg.head_dim()
    }

    pub fn uses_tied_embeddings(&self) -> bool {
        self.cfg.tie_embeddings
    }

    pub fn uses_rope(&self) -> bool {
        self.cfg.position_encoding == PositionEncodingKind::Rope
    }

    pub fn position_embedding(&self) -> Option<&Matrix> {
        self.wpe.as_ref()
    }

    pub fn position_embedding_mut(&mut self) -> Option<&mut Matrix> {
        self.wpe.as_mut()
    }

    pub fn output_projection(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.wte)
    }

    pub fn output_projection_mut(&mut self) -> &mut Matrix {
        if self.cfg.tie_embeddings {
            &mut self.wte
        } else {
            self.lm_head
                .as_mut()
                .expect("untied model must contain lm_head")
        }
    }

    pub fn output_parameter_id(&self) -> ParameterId {
        if self.cfg.tie_embeddings {
            ParameterId::Wte
        } else {
            ParameterId::LmHead
        }
    }

    pub fn visit_parameters<E, F>(&self, mut visit: F) -> std::result::Result<(), E>
    where
        F: FnMut(ParameterId, &Matrix) -> std::result::Result<(), E>,
    {
        visit(ParameterId::Wte, &self.wte)?;
        if let Some(wpe) = &self.wpe {
            visit(ParameterId::Wpe, wpe)?;
        }
        if let Some(lm_head) = &self.lm_head {
            visit(ParameterId::LmHead, lm_head)?;
        }
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            visit(ParameterId::AttnWq(layer_idx), &layer.attn_wq)?;
            visit(ParameterId::AttnWk(layer_idx), &layer.attn_wk)?;
            visit(ParameterId::AttnWv(layer_idx), &layer.attn_wv)?;
            visit(ParameterId::AttnWo(layer_idx), &layer.attn_wo)?;
            visit(ParameterId::MlpFc1(layer_idx), &layer.mlp_fc1)?;
            if let Some(mlp_fc_gate) = &layer.mlp_fc_gate {
                visit(ParameterId::MlpFcGate(layer_idx), mlp_fc_gate)?;
            }
            visit(ParameterId::MlpFc2(layer_idx), &layer.mlp_fc2)?;
        }
        Ok(())
    }

    pub fn visit_parameters_mut<E, F>(&mut self, mut visit: F) -> std::result::Result<(), E>
    where
        F: FnMut(ParameterId, &mut Matrix) -> std::result::Result<(), E>,
    {
        visit(ParameterId::Wte, &mut self.wte)?;
        if let Some(wpe) = &mut self.wpe {
            visit(ParameterId::Wpe, wpe)?;
        }
        if let Some(lm_head) = &mut self.lm_head {
            visit(ParameterId::LmHead, lm_head)?;
        }
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            visit(ParameterId::AttnWq(layer_idx), &mut layer.attn_wq)?;
            visit(ParameterId::AttnWk(layer_idx), &mut layer.attn_wk)?;
            visit(ParameterId::AttnWv(layer_idx), &mut layer.attn_wv)?;
            visit(ParameterId::AttnWo(layer_idx), &mut layer.attn_wo)?;
            visit(ParameterId::MlpFc1(layer_idx), &mut layer.mlp_fc1)?;
            if let Some(mlp_fc_gate) = &mut layer.mlp_fc_gate {
                visit(ParameterId::MlpFcGate(layer_idx), mlp_fc_gate)?;
            }
            visit(ParameterId::MlpFc2(layer_idx), &mut layer.mlp_fc2)?;
        }
        Ok(())
    }

    pub fn num_parameters(&self) -> usize {
        let mut total = self.wte.parameter_count();
        if let Some(wpe) = &self.wpe {
            total += wpe.parameter_count();
        }
        if let Some(lm_head) = &self.lm_head {
            total += lm_head.parameter_count();
        }
        for layer in &self.layers {
            total += layer.attn_wq.parameter_count();
            total += layer.attn_wk.parameter_count();
            total += layer.attn_wv.parameter_count();
            total += layer.attn_wo.parameter_count();
            total += layer.mlp_fc1.parameter_count();
            if let Some(mlp_fc_gate) = &layer.mlp_fc_gate {
                total += mlp_fc_gate.parameter_count();
            }
            total += layer.mlp_fc2.parameter_count();
        }
        total
    }

    pub fn zero_grads(&mut self) {
        self.wte.zero_grad();
        if let Some(wpe) = &mut self.wpe {
            wpe.zero_grad();
        }
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.zero_grad();
        }
        for layer in &mut self.layers {
            layer.attn_wq.zero_grad();
            layer.attn_wk.zero_grad();
            layer.attn_wv.zero_grad();
            layer.attn_wo.zero_grad();
            layer.mlp_fc1.zero_grad();
            if let Some(mlp_fc_gate) = &mut layer.mlp_fc_gate {
                mlp_fc_gate.zero_grad();
            }
            layer.mlp_fc2.zero_grad();
        }
    }

    pub fn grad_squared_sum(&self) -> f32 {
        let mut total = self.wte.grad_squared_sum();
        if let Some(wpe) = &self.wpe {
            total += wpe.grad_squared_sum();
        }
        if let Some(lm_head) = &self.lm_head {
            total += lm_head.grad_squared_sum();
        }
        for layer in &self.layers {
            total += layer.attn_wq.grad_squared_sum();
            total += layer.attn_wk.grad_squared_sum();
            total += layer.attn_wv.grad_squared_sum();
            total += layer.attn_wo.grad_squared_sum();
            total += layer.mlp_fc1.grad_squared_sum();
            if let Some(mlp_fc_gate) = &layer.mlp_fc_gate {
                total += mlp_fc_gate.grad_squared_sum();
            }
            total += layer.mlp_fc2.grad_squared_sum();
        }
        total
    }

    pub fn scale_gradients(&mut self, scale: f32) {
        self.wte.scale_gradients(scale);
        if let Some(wpe) = &mut self.wpe {
            wpe.scale_gradients(scale);
        }
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.scale_gradients(scale);
        }
        for layer in &mut self.layers {
            layer.attn_wq.scale_gradients(scale);
            layer.attn_wk.scale_gradients(scale);
            layer.attn_wv.scale_gradients(scale);
            layer.attn_wo.scale_gradients(scale);
            layer.mlp_fc1.scale_gradients(scale);
            if let Some(mlp_fc_gate) = &mut layer.mlp_fc_gate {
                mlp_fc_gate.scale_gradients(scale);
            }
            layer.mlp_fc2.scale_gradients(scale);
        }
    }

    pub fn adam_step(
        &mut self,
        lr_t: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step_num: usize,
    ) {
        self.wte
            .adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
        if let Some(wpe) = &mut self.wpe {
            wpe.adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
        }
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
        }
        for layer in &mut self.layers {
            layer
                .attn_wq
                .adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
            layer
                .attn_wk
                .adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
            layer
                .attn_wv
                .adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
            layer
                .attn_wo
                .adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
            layer
                .mlp_fc1
                .adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
            if let Some(mlp_fc_gate) = &mut layer.mlp_fc_gate {
                mlp_fc_gate.adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
            }
            layer
                .mlp_fc2
                .adam_step(lr_t, beta1, beta2, eps, weight_decay, step_num);
        }
    }

    fn validate_config(cfg: &ModelConfig) -> Result<()> {
        if cfg.n_head == 0 {
            return Err(RustGptError::Config("n_head must be > 0".to_string()));
        }
        if cfg.n_kv_head == 0 {
            return Err(RustGptError::Config("n_kv_head must be > 0".to_string()));
        }
        if cfg.n_embd == 0 {
            return Err(RustGptError::Config("n_embd must be > 0".to_string()));
        }
        if cfg.n_embd % cfg.n_head != 0 {
            return Err(RustGptError::Config(format!(
                "n_embd ({}) must be divisible by n_head ({})",
                cfg.n_embd, cfg.n_head
            )));
        }
        if cfg.n_head % cfg.n_kv_head != 0 {
            return Err(RustGptError::Config(format!(
                "n_head ({}) must be divisible by n_kv_head ({})",
                cfg.n_head, cfg.n_kv_head
            )));
        }
        if cfg.position_encoding == PositionEncodingKind::Rope && cfg.head_dim() % 2 != 0 {
            return Err(RustGptError::Config(format!(
                "rope requires an even head_dim, got {}",
                cfg.head_dim()
            )));
        }
        if cfg.vocab_size == 0 {
            return Err(RustGptError::Config("vocab_size must be > 0".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::{ActivationKind, BoundaryMode, ModelConfig, PositionEncodingKind};
    use crate::core::rng::Rng;

    use super::Model;

    #[test]
    fn model_parameter_count_matches_python_layout() {
        let mut rng = Rng::from_seed(42);
        let cfg = ModelConfig {
            vocab_size: 27,
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
        let model = Model::new(cfg, &mut rng).unwrap();
        assert_eq!(model.num_parameters(), 4_192);
    }

    #[test]
    fn tied_embeddings_reduce_parameter_count() {
        let mut rng = Rng::from_seed(42);
        let cfg = ModelConfig {
            vocab_size: 27,
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 4,
            tie_embeddings: true,
            activation: ActivationKind::Relu,
            position_encoding: PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
        };
        let model = Model::new(cfg, &mut rng).unwrap();
        assert!(model.uses_tied_embeddings());
        assert!(model.lm_head.is_none());
        assert_eq!(model.num_parameters(), 3_760);
    }

    #[test]
    fn rope_model_omits_learned_position_table() {
        let mut rng = Rng::from_seed(42);
        let cfg = ModelConfig {
            vocab_size: 27,
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
        let model = Model::new(cfg, &mut rng).unwrap();
        assert!(model.position_embedding().is_none());
        assert_eq!(model.num_parameters(), 3_936);
    }

    #[test]
    fn swiglu_model_adds_gate_projection() {
        let mut rng = Rng::from_seed(42);
        let cfg = ModelConfig {
            vocab_size: 27,
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
        let model = Model::new(cfg, &mut rng).unwrap();
        assert!(model.layers[0].mlp_fc_gate.is_some());
    }

    #[test]
    fn grouped_query_attention_reduces_kv_parameter_count() {
        let mut rng = Rng::from_seed(42);
        let cfg = ModelConfig {
            vocab_size: 27,
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 2,
            tie_embeddings: false,
            activation: ActivationKind::Relu,
            position_encoding: PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
        };
        let model = Model::new(cfg, &mut rng).unwrap();
        assert_eq!(model.layers[0].attn_wk.rows, 8);
        assert_eq!(model.layers[0].attn_wv.rows, 8);
    }
}
