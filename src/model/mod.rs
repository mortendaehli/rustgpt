use crate::core::config::ModelConfig;
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
    pub mlp_fc2: Matrix,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub cfg: ModelConfig,
    pub wte: Matrix,
    pub wpe: Matrix,
    pub lm_head: Matrix,
    pub layers: Vec<TransformerLayer>,
}

impl Model {
    pub fn new(cfg: ModelConfig, rng: &mut Rng) -> Result<Self> {
        Self::validate_config(&cfg)?;

        let wte = Matrix::from_gaussian(cfg.vocab_size, cfg.n_embd, INIT_STD, rng);
        let wpe = Matrix::from_gaussian(cfg.block_size, cfg.n_embd, INIT_STD, rng);
        let lm_head = Matrix::from_gaussian(cfg.vocab_size, cfg.n_embd, INIT_STD, rng);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            layers.push(TransformerLayer {
                attn_wq: Matrix::from_gaussian(cfg.n_embd, cfg.n_embd, INIT_STD, rng),
                attn_wk: Matrix::from_gaussian(cfg.n_embd, cfg.n_embd, INIT_STD, rng),
                attn_wv: Matrix::from_gaussian(cfg.n_embd, cfg.n_embd, INIT_STD, rng),
                attn_wo: Matrix::from_gaussian(cfg.n_embd, cfg.n_embd, INIT_STD, rng),
                mlp_fc1: Matrix::from_gaussian(4 * cfg.n_embd, cfg.n_embd, INIT_STD, rng),
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
        wpe: Matrix,
        lm_head: Matrix,
        layers: Vec<TransformerLayer>,
    ) -> Result<Self> {
        Self::validate_config(&cfg)?;
        if wte.rows != cfg.vocab_size || wte.cols != cfg.n_embd {
            return Err(RustGptError::Config(
                "wte shape does not match model config".to_string(),
            ));
        }
        if wpe.rows != cfg.block_size || wpe.cols != cfg.n_embd {
            return Err(RustGptError::Config(
                "wpe shape does not match model config".to_string(),
            ));
        }
        if lm_head.rows != cfg.vocab_size || lm_head.cols != cfg.n_embd {
            return Err(RustGptError::Config(
                "lm_head shape does not match model config".to_string(),
            ));
        }
        if layers.len() != cfg.n_layer {
            return Err(RustGptError::Config(
                "checkpoint layer count does not match model config".to_string(),
            ));
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

    pub fn visit_parameters<E, F>(&self, mut visit: F) -> std::result::Result<(), E>
    where
        F: FnMut(ParameterId, &Matrix) -> std::result::Result<(), E>,
    {
        visit(ParameterId::Wte, &self.wte)?;
        visit(ParameterId::Wpe, &self.wpe)?;
        visit(ParameterId::LmHead, &self.lm_head)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            visit(ParameterId::AttnWq(layer_idx), &layer.attn_wq)?;
            visit(ParameterId::AttnWk(layer_idx), &layer.attn_wk)?;
            visit(ParameterId::AttnWv(layer_idx), &layer.attn_wv)?;
            visit(ParameterId::AttnWo(layer_idx), &layer.attn_wo)?;
            visit(ParameterId::MlpFc1(layer_idx), &layer.mlp_fc1)?;
            visit(ParameterId::MlpFc2(layer_idx), &layer.mlp_fc2)?;
        }
        Ok(())
    }

    pub fn visit_parameters_mut<E, F>(&mut self, mut visit: F) -> std::result::Result<(), E>
    where
        F: FnMut(ParameterId, &mut Matrix) -> std::result::Result<(), E>,
    {
        visit(ParameterId::Wte, &mut self.wte)?;
        visit(ParameterId::Wpe, &mut self.wpe)?;
        visit(ParameterId::LmHead, &mut self.lm_head)?;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            visit(ParameterId::AttnWq(layer_idx), &mut layer.attn_wq)?;
            visit(ParameterId::AttnWk(layer_idx), &mut layer.attn_wk)?;
            visit(ParameterId::AttnWv(layer_idx), &mut layer.attn_wv)?;
            visit(ParameterId::AttnWo(layer_idx), &mut layer.attn_wo)?;
            visit(ParameterId::MlpFc1(layer_idx), &mut layer.mlp_fc1)?;
            visit(ParameterId::MlpFc2(layer_idx), &mut layer.mlp_fc2)?;
        }
        Ok(())
    }

    pub fn num_parameters(&self) -> usize {
        let mut total = self.wte.parameter_count()
            + self.wpe.parameter_count()
            + self.lm_head.parameter_count();
        for layer in &self.layers {
            total += layer.attn_wq.parameter_count();
            total += layer.attn_wk.parameter_count();
            total += layer.attn_wv.parameter_count();
            total += layer.attn_wo.parameter_count();
            total += layer.mlp_fc1.parameter_count();
            total += layer.mlp_fc2.parameter_count();
        }
        total
    }

    pub fn zero_grads(&mut self) {
        self.wte.zero_grad();
        self.wpe.zero_grad();
        self.lm_head.zero_grad();
        for layer in &mut self.layers {
            layer.attn_wq.zero_grad();
            layer.attn_wk.zero_grad();
            layer.attn_wv.zero_grad();
            layer.attn_wo.zero_grad();
            layer.mlp_fc1.zero_grad();
            layer.mlp_fc2.zero_grad();
        }
    }

    pub fn adam_step(&mut self, lr_t: f32, beta1: f32, beta2: f32, eps: f32, step_num: usize) {
        self.wte.adam_step(lr_t, beta1, beta2, eps, step_num);
        self.wpe.adam_step(lr_t, beta1, beta2, eps, step_num);
        self.lm_head.adam_step(lr_t, beta1, beta2, eps, step_num);
        for layer in &mut self.layers {
            layer.attn_wq.adam_step(lr_t, beta1, beta2, eps, step_num);
            layer.attn_wk.adam_step(lr_t, beta1, beta2, eps, step_num);
            layer.attn_wv.adam_step(lr_t, beta1, beta2, eps, step_num);
            layer.attn_wo.adam_step(lr_t, beta1, beta2, eps, step_num);
            layer.mlp_fc1.adam_step(lr_t, beta1, beta2, eps, step_num);
            layer.mlp_fc2.adam_step(lr_t, beta1, beta2, eps, step_num);
        }
    }

    fn validate_config(cfg: &ModelConfig) -> Result<()> {
        if cfg.n_head == 0 {
            return Err(RustGptError::Config("n_head must be > 0".to_string()));
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
        if cfg.vocab_size == 0 {
            return Err(RustGptError::Config("vocab_size must be > 0".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::{BoundaryMode, ModelConfig};
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
            boundary_mode: BoundaryMode::SharedBos,
        };
        let model = Model::new(cfg, &mut rng).unwrap();
        assert_eq!(model.num_parameters(), 4_192);
    }
}
