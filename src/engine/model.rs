use burn::module::{Ignored, Module};
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, grad_clipping::GradientClippingConfig};
use burn::tensor::activation::{gelu, quiet_softmax, relu, sigmoid, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Int, Tensor, TensorData};

use crate::core::config::{ActivationKind, ModelConfig, PositionEncodingKind, TrainConfig};
use crate::core::error::{Result, RustGptError};

pub type LanguageModelOptimizer<AD> = OptimizerAdaptor<AdamW, LanguageModel<AD>, AD>;

#[derive(Module, Debug)]
pub struct LanguageModel<B: Backend> {
    cfg: Ignored<ModelConfig>,
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<DecoderBlock<B>>,
    final_norm: RmsNorm<B>,
    lm_head: Option<Linear<B>>,
}

#[derive(Module, Debug)]
struct DecoderBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    norm_attn: RmsNorm<B>,
    norm_ff: RmsNorm<B>,
    ff_inner: Linear<B>,
    ff_gate: Option<Linear<B>>,
    ff_outer: Linear<B>,
    activation: Ignored<ActivationKind>,
}

pub struct DecodeCache<B: Backend> {
    layers: Vec<DecoderBlockCache<B>>,
    sequence_len: usize,
}

struct DecoderBlockCache<B: Backend> {
    attention: AttentionKvCache<B>,
}

struct AttentionKvCache<B: Backend> {
    key: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
}

impl<B: Backend> LanguageModel<B> {
    pub fn new(cfg: ModelConfig, device: &B::Device) -> Result<Self> {
        validate_model_config(&cfg)?;

        let ff_hidden = cfg.n_embd * 4;
        let token_embedding = EmbeddingConfig::new(cfg.vocab_size, cfg.n_embd).init(device);
        let position_embedding = EmbeddingConfig::new(cfg.block_size, cfg.n_embd).init(device);
        let blocks = (0..cfg.n_layer)
            .map(|_| DecoderBlock::new(&cfg, ff_hidden, device))
            .collect();
        let final_norm = RmsNormConfig::new(cfg.n_embd).init(device);
        let lm_head = (!cfg.tie_embeddings)
            .then(|| LinearConfig::new(cfg.n_embd, cfg.vocab_size).init(device));

        Ok(Self {
            cfg: Ignored(cfg),
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            lm_head,
        })
    }

    pub fn num_parameters(&self) -> usize {
        self.num_params()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn new_decode_cache(&self) -> DecodeCache<B> {
        DecodeCache {
            layers: self
                .blocks
                .iter()
                .map(|_| DecoderBlockCache::empty())
                .collect(),
            sequence_len: 0,
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length] = input_ids.dims();
        let position_ids = position_ids_tensor::<B>(batch_size, seq_length, &input_ids.device());
        let mut hidden =
            self.token_embedding.forward(input_ids) + self.position_embedding.forward(position_ids);
        let attn_mask =
            Tensor::<B, 2, Bool>::tril_mask([seq_length, seq_length], 0, &hidden.device())
                .expand([batch_size, seq_length, seq_length]);

        for block in &self.blocks {
            hidden = block.forward(hidden, pad_mask.clone(), attn_mask.clone());
        }

        self.project_logits(self.final_norm.forward(hidden))
    }

    pub fn prefill(&self, token_ids: &[usize], cache: &mut DecodeCache<B>) -> Tensor<B, 1> {
        assert!(
            !token_ids.is_empty(),
            "prefill requires at least one conditioning token"
        );

        cache.reset();
        let device = self.device();
        let values = token_ids
            .iter()
            .map(|token_id| *token_id as i64)
            .collect::<Vec<_>>();
        let seq_length = token_ids.len();
        let input_ids = int_tensor_2d::<B>(&values, 1, seq_length, &device);
        let position_ids = position_ids_tensor_with_offset::<B>(1, seq_length, 0, &device);
        let mut hidden =
            self.token_embedding.forward(input_ids) + self.position_embedding.forward(position_ids);
        let attn_mask = causal_mask::<B>(1, seq_length, &hidden.device());

        for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            hidden = block.forward_prefill(hidden, attn_mask.clone(), layer_cache);
        }

        cache.sequence_len = seq_length;
        self.last_token_logits(self.project_logits(self.final_norm.forward(hidden)))
    }

    pub fn forward_step(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut DecodeCache<B>,
    ) -> Tensor<B, 1> {
        debug_assert_eq!(
            cache.sequence_len, position,
            "decode cache and requested position diverged"
        );

        let device = self.device();
        let input_ids = int_tensor_2d::<B>(&[token_id as i64], 1, 1, &device);
        let position_ids = position_ids_tensor_with_offset::<B>(1, 1, position, &device);
        let mut hidden =
            self.token_embedding.forward(input_ids) + self.position_embedding.forward(position_ids);

        for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            hidden = block.forward_decode_step(hidden, layer_cache);
        }

        cache.sequence_len = position + 1;
        self.last_token_logits(self.project_logits(self.final_norm.forward(hidden)))
    }

    fn project_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.lm_head {
            Some(lm_head) => lm_head.forward(hidden),
            None => {
                let [batch_size, seq_length, hidden_dim] = hidden.dims();
                hidden
                    .reshape([batch_size * seq_length, hidden_dim])
                    .matmul(self.token_embedding.weight.val().swap_dims(0, 1))
                    .reshape([batch_size, seq_length, self.config().vocab_size])
            }
        }
    }

    fn last_token_logits(&self, logits: Tensor<B, 3>) -> Tensor<B, 1> {
        let [_batch_size, seq_length, _vocab] = logits.dims();
        logits
            .slice([
                0..1,
                seq_length.saturating_sub(1)..seq_length,
                0..self.config().vocab_size,
            ])
            .reshape([self.config().vocab_size])
    }

    fn device(&self) -> B::Device {
        self.token_embedding
            .devices()
            .into_iter()
            .next()
            .expect("language model must own a device")
    }
}

impl<B: Backend> DecoderBlock<B> {
    fn new(cfg: &ModelConfig, ff_hidden: usize, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(cfg.n_embd, cfg.n_head)
            .with_dropout(0.0)
            .init(device);
        let norm_attn = RmsNormConfig::new(cfg.n_embd).init(device);
        let norm_ff = RmsNormConfig::new(cfg.n_embd).init(device);
        let ff_inner = LinearConfig::new(cfg.n_embd, ff_hidden).init(device);
        let ff_gate = matches!(cfg.activation, ActivationKind::SwiGlu)
            .then(|| LinearConfig::new(cfg.n_embd, ff_hidden).init(device));
        let ff_outer = LinearConfig::new(ff_hidden, cfg.n_embd).init(device);

        Self {
            attention,
            norm_attn,
            norm_ff,
            ff_inner,
            ff_gate,
            ff_outer,
            activation: Ignored(cfg.activation),
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
        attn_mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let mut attn_input = MhaInput::self_attn(self.norm_attn.forward(input));
        if let Some(pad_mask) = pad_mask {
            attn_input = attn_input.mask_pad(pad_mask);
        }
        let attn = self
            .attention
            .forward(attn_input.mask_attn(attn_mask))
            .context;
        let hidden = residual + attn;
        let residual = hidden.clone();
        residual + self.feed_forward(self.norm_ff.forward(hidden))
    }

    fn forward_prefill(
        &self,
        input: Tensor<B, 3>,
        attn_mask: Tensor<B, 3, Bool>,
        cache: &mut DecoderBlockCache<B>,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let attn = self.self_attention_prefill(self.norm_attn.forward(input), attn_mask, cache);
        let hidden = residual + attn;
        let residual = hidden.clone();
        residual + self.feed_forward(self.norm_ff.forward(hidden))
    }

    fn forward_decode_step(
        &self,
        input: Tensor<B, 3>,
        cache: &mut DecoderBlockCache<B>,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let attn = self.self_attention_decode_step(self.norm_attn.forward(input), cache);
        let hidden = residual + attn;
        let residual = hidden.clone();
        residual + self.feed_forward(self.norm_ff.forward(hidden))
    }

    fn feed_forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.ff_inner.forward(input.clone());
        let activated = match *self.activation {
            ActivationKind::Relu => relu(hidden),
            ActivationKind::Gelu => gelu(hidden),
            ActivationKind::SwiGlu => {
                let gate = self
                    .ff_gate
                    .as_ref()
                    .expect("SwiGLU requires a gate projection")
                    .forward(input);
                hidden * (gate.clone() * sigmoid(gate))
            }
        };
        self.ff_outer.forward(activated)
    }

    fn self_attention_prefill(
        &self,
        input: Tensor<B, 3>,
        attn_mask: Tensor<B, 3, Bool>,
        cache: &mut DecoderBlockCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length, _d_model] = input.dims();
        let query = self.attention_linear(input.clone(), &self.attention.query);
        let key = self.attention_linear(input.clone(), &self.attention.key);
        let value = self.attention_linear(input, &self.attention.value);
        cache.attention.set(key.clone(), value.clone());

        let attn_scores = self.attention_scores(query, key).mask_fill(
            attn_mask.reshape([batch_size, 1, seq_length, seq_length]),
            self.attention.min_float,
        );
        let weights = self.attention_weights(attn_scores);
        let context = weights.clone().matmul(value);
        self.project_attention_context(context, batch_size, seq_length)
    }

    fn self_attention_decode_step(
        &self,
        input: Tensor<B, 3>,
        cache: &mut DecoderBlockCache<B>,
    ) -> Tensor<B, 3> {
        let query = self.attention_linear(input.clone(), &self.attention.query);
        let key = self.attention_linear(input.clone(), &self.attention.key);
        let value = self.attention_linear(input, &self.attention.value);
        let (key, value) = cache.attention.append(key, value);

        let weights = self.attention_weights(self.attention_scores(query, key));
        let context = weights.matmul(value);
        self.project_attention_context(context, 1, 1)
    }

    fn attention_linear(&self, input: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = input.dims();
        linear
            .forward(input)
            .reshape([
                batch_size,
                seq_length,
                self.attention.n_heads,
                self.attention.d_k,
            ])
            .swap_dims(1, 2)
    }

    fn attention_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        let scores = query
            .matmul(key.transpose())
            .div_scalar((self.attention.d_k as f32).sqrt());
        self.attention.dropout.forward(scores)
    }

    fn attention_weights(&self, attn_scores: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.attention.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        }
    }

    fn project_attention_context(
        &self,
        context: Tensor<B, 4>,
        batch_size: usize,
        seq_length: usize,
    ) -> Tensor<B, 3> {
        self.attention
            .output
            .forward(context.swap_dims(1, 2).reshape([
                batch_size,
                seq_length,
                self.attention.d_model,
            ]))
    }
}

impl<B: Backend> DecodeCache<B> {
    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.attention.clear();
        }
        self.sequence_len = 0;
    }
}

impl<B: Backend> DecoderBlockCache<B> {
    fn empty() -> Self {
        Self {
            attention: AttentionKvCache::empty(),
        }
    }
}

impl<B: Backend> AttentionKvCache<B> {
    fn empty() -> Self {
        Self {
            key: None,
            value: None,
        }
    }

    fn clear(&mut self) {
        self.key = None;
        self.value = None;
    }

    fn set(&mut self, key: Tensor<B, 4>, value: Tensor<B, 4>) {
        self.key = Some(key);
        self.value = Some(value);
    }

    fn append(&mut self, key: Tensor<B, 4>, value: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let key = match self.key.take() {
            Some(cache) => Tensor::cat(vec![cache, key], 2),
            None => key,
        };
        let value = match self.value.take() {
            Some(cache) => Tensor::cat(vec![cache, value], 2),
            None => value,
        };
        self.key = Some(key.clone());
        self.value = Some(value.clone());
        (key, value)
    }
}

pub fn validate_model_config(cfg: &ModelConfig) -> Result<()> {
    if cfg.block_size == 0 {
        return Err(RustGptError::Config(
            "block_size must be at least 1".to_string(),
        ));
    }
    if cfg.n_layer == 0 || cfg.n_embd == 0 || cfg.n_head == 0 {
        return Err(RustGptError::Config(
            "n_layer, n_embd, and n_head must all be at least 1".to_string(),
        ));
    }
    if cfg.n_embd % cfg.n_head != 0 {
        return Err(RustGptError::Config(format!(
            "n_embd ({}) must be divisible by n_head ({})",
            cfg.n_embd, cfg.n_head
        )));
    }
    if cfg.n_kv_head != cfg.n_head {
        return Err(RustGptError::Config(
            "Burn migration currently supports only n_kv_head == n_head".to_string(),
        ));
    }
    if cfg.position_encoding != PositionEncodingKind::LearnedAbsolute {
        return Err(RustGptError::Config(
            "Burn migration currently supports only learned absolute position embeddings"
                .to_string(),
        ));
    }

    Ok(())
}

pub fn init_model<B: Backend>(
    cfg: &ModelConfig,
    seed: u64,
    device: &B::Device,
) -> Result<LanguageModel<B>> {
    B::seed(device, seed);
    LanguageModel::new(cfg.clone(), device)
}

pub fn build_optimizer<AD: AutodiffBackend>(
    train_config: &TrainConfig,
) -> LanguageModelOptimizer<AD> {
    let mut config = AdamWConfig::new()
        .with_beta_1(train_config.beta1)
        .with_beta_2(train_config.beta2)
        .with_epsilon(train_config.eps_adam)
        .with_weight_decay(train_config.weight_decay);
    if train_config.grad_clip > 0.0 {
        config =
            config.with_grad_clipping(Some(GradientClippingConfig::Norm(train_config.grad_clip)));
    }
    config.init::<AD, LanguageModel<AD>>()
}

fn position_ids_tensor<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    position_ids_tensor_with_offset::<B>(batch_size, seq_length, 0, device)
}

fn position_ids_tensor_with_offset<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    offset: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let values = (0..batch_size)
        .flat_map(|_| (0..seq_length).map(move |position| (offset + position) as i64))
        .collect::<Vec<_>>();
    int_tensor_2d::<B>(&values, batch_size, seq_length, device)
}

fn causal_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    Tensor::<B, 2, Bool>::tril_mask([seq_length, seq_length], 0, device)
        .expand([batch_size, seq_length, seq_length])
}

fn int_tensor_2d<B: Backend>(
    values: &[i64],
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::<B, 2, Int>::from_data(TensorData::new(values.to_vec(), [rows, cols]), device)
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Int, Tensor, TensorData, Tolerance};

    use crate::core::config::{ActivationKind, BoundaryMode, ModelConfig, PositionEncodingKind};

    use super::{LanguageModel, validate_model_config};

    type TestBackend = NdArray<f32>;

    fn test_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32,
            block_size: 8,
            n_layer: 2,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 4,
            tie_embeddings: true,
            activation: ActivationKind::Gelu,
            position_encoding: PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
        }
    }

    #[test]
    fn validates_supported_configs() {
        validate_model_config(&test_config()).unwrap();
    }

    #[test]
    fn forward_preserves_expected_shape() {
        let cfg = test_config();
        let device = Default::default();
        let model = LanguageModel::<TestBackend>::new(cfg.clone(), &device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );
        let logits = model.forward(input, None);
        assert_eq!(logits.dims(), [1, 4, cfg.vocab_size]);
    }

    #[test]
    fn forward_step_keeps_single_token_shape_across_cached_steps() {
        let cfg = test_config();
        let device = Default::default();
        let model = LanguageModel::<TestBackend>::new(cfg.clone(), &device).unwrap();
        let mut cache = model.new_decode_cache();

        let logits_1 = model.forward_step(1, 0, &mut cache);
        let logits_2 = model.forward_step(2, 1, &mut cache);

        assert_eq!(logits_1.dims(), [cfg.vocab_size]);
        assert_eq!(logits_2.dims(), [cfg.vocab_size]);
    }

    #[test]
    fn prefill_and_decode_match_full_forward() {
        let cfg = test_config();
        let device = Default::default();
        TestBackend::seed(&device, 7);
        let model = LanguageModel::<TestBackend>::new(cfg.clone(), &device).unwrap();
        let mut cache = model.new_decode_cache();

        let prefix = [1_usize, 2, 3];
        let prefix_input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );
        let prefix_logits = model
            .forward(prefix_input, None)
            .slice([0..1, 2..3, 0..cfg.vocab_size])
            .reshape([cfg.vocab_size]);
        let cached_prefix_logits = model.prefill(&prefix, &mut cache);
        prefix_logits
            .into_data()
            .assert_approx_eq::<f32>(&cached_prefix_logits.into_data(), Tolerance::default());

        let extended_input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5], [1, 5]),
            &device,
        );
        let full_logits = model.forward(extended_input, None);
        let decode_logits_1 = model.forward_step(4, 3, &mut cache);
        let decode_logits_2 = model.forward_step(5, 4, &mut cache);

        full_logits
            .clone()
            .slice([0..1, 3..4, 0..cfg.vocab_size])
            .reshape([cfg.vocab_size])
            .into_data()
            .assert_approx_eq::<f32>(&decode_logits_1.into_data(), Tolerance::default());
        full_logits
            .slice([0..1, 4..5, 0..cfg.vocab_size])
            .reshape([cfg.vocab_size])
            .into_data()
            .assert_approx_eq::<f32>(&decode_logits_2.into_data(), Tolerance::default());
    }
}
