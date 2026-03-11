use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DataFormat {
    Lines,
    PlainText,
    JsonlText,
    JsonlChat,
    ParquetText,
    ParquetChat,
}

impl DataFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "lines" => Some(Self::Lines),
            "text" | "plain-text" => Some(Self::PlainText),
            "jsonl-text" => Some(Self::JsonlText),
            "jsonl-chat" => Some(Self::JsonlChat),
            "parquet-text" => Some(Self::ParquetText),
            "parquet-chat" => Some(Self::ParquetChat),
            _ => None,
        }
    }

    pub fn is_chat(self) -> bool {
        matches!(self, Self::JsonlChat | Self::ParquetChat)
    }
}

impl Display for DataFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lines => write!(f, "lines"),
            Self::PlainText => write!(f, "plain-text"),
            Self::JsonlText => write!(f, "jsonl-text"),
            Self::JsonlChat => write!(f, "jsonl-chat"),
            Self::ParquetText => write!(f, "parquet-text"),
            Self::ParquetChat => write!(f, "parquet-chat"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ChatTemplateKind {
    SimpleTranscript,
    ChatMl,
}

impl ChatTemplateKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "simple" | "simple-transcript" => Some(Self::SimpleTranscript),
            "chatml" => Some(Self::ChatMl),
            _ => None,
        }
    }
}

impl Display for ChatTemplateKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SimpleTranscript => write!(f, "simple-transcript"),
            Self::ChatMl => write!(f, "chatml"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TrainMode {
    Auto,
    Pretrain,
    Sft,
}

impl TrainMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "pretrain" => Some(Self::Pretrain),
            "sft" => Some(Self::Sft),
            _ => None,
        }
    }
}

impl Display for TrainMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Pretrain => write!(f, "pretrain"),
            Self::Sft => write!(f, "sft"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BoundaryMode {
    SharedBos,
    SeparateBosEos,
}

impl Display for BoundaryMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharedBos => write!(f, "shared-bos"),
            Self::SeparateBosEos => write!(f, "separate-bos-eos"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DeviceKind {
    Cpu,
    Auto,
    Gpu,
}

impl DeviceKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "cpu" => Some(Self::Cpu),
            "auto" => Some(Self::Auto),
            "gpu" => Some(Self::Gpu),
            _ => None,
        }
    }
}

impl Display for DeviceKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Auto => write!(f, "auto"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ActivationKind {
    Relu,
    Gelu,
    SwiGlu,
}

impl ActivationKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "relu" => Some(Self::Relu),
            "gelu" => Some(Self::Gelu),
            "swiglu" | "swi-glu" => Some(Self::SwiGlu),
            _ => None,
        }
    }
}

impl Display for ActivationKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Relu => write!(f, "relu"),
            Self::Gelu => write!(f, "gelu"),
            Self::SwiGlu => write!(f, "swiglu"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PositionEncodingKind {
    LearnedAbsolute,
    Rope,
}

impl PositionEncodingKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "learned" | "learned-absolute" => Some(Self::LearnedAbsolute),
            "rope" => Some(Self::Rope),
            _ => None,
        }
    }
}

impl Display for PositionEncodingKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LearnedAbsolute => write!(f, "learned-absolute"),
            Self::Rope => write!(f, "rope"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum LrScheduleKind {
    Linear,
    Cosine,
}

impl LrScheduleKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "linear" => Some(Self::Linear),
            "cosine" => Some(Self::Cosine),
            _ => None,
        }
    }
}

impl Display for LrScheduleKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear => write!(f, "linear"),
            Self::Cosine => write!(f, "cosine"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TokenizerModelKind {
    Bpe,
}

impl TokenizerModelKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "bpe" => Some(Self::Bpe),
            _ => None,
        }
    }
}

impl Display for TokenizerModelKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bpe => write!(f, "bpe"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataConfig {
    pub data_path: PathBuf,
    pub format: DataFormat,
    pub shuffle: bool,
    pub lowercase: bool,
    pub tokenizer_path: Option<PathBuf>,
    pub tokenizer_bos: Option<String>,
    pub tokenizer_eos: Option<String>,
    pub chat_template: ChatTemplateKind,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("input.txt"),
            format: DataFormat::Lines,
            shuffle: true,
            lowercase: false,
            tokenizer_path: None,
            tokenizer_bos: None,
            tokenizer_eos: None,
            chat_template: ChatTemplateKind::SimpleTranscript,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub activation_checkpointing: bool,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps_adam: f32,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub grad_clip: f32,
    pub lr_schedule: LrScheduleKind,
    pub validation_ratio: f32,
    pub validation_max_examples: usize,
    pub seed: u64,
    pub sample_every: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub tie_embeddings: bool,
    pub activation: ActivationKind,
    pub position_encoding: PositionEncodingKind,
    pub boundary_mode: BoundaryMode,
    pub mode: TrainMode,
    pub device: DeviceKind,
    pub profile: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 1_000,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            activation_checkpointing: false,
            learning_rate: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            eps_adam: 1e-8,
            weight_decay: 0.0,
            warmup_steps: 0,
            grad_clip: 0.0,
            lr_schedule: LrScheduleKind::Linear,
            validation_ratio: 0.0,
            validation_max_examples: 64,
            seed: 42,
            sample_every: 100,
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 0,
            tie_embeddings: false,
            activation: ActivationKind::Relu,
            position_encoding: PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
            mode: TrainMode::Auto,
            device: DeviceKind::Cpu,
            profile: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SampleConfig {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub samples: usize,
    pub seed: u64,
    pub device: DeviceKind,
    pub profile: bool,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_new_tokens: 16,
            temperature: 0.5,
            top_k: 40,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            samples: 5,
            seed: 42,
            device: DeviceKind::Cpu,
            profile: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChatConfig {
    pub system_prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stream: bool,
    pub seed: u64,
    pub device: DeviceKind,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_new_tokens: 32,
            temperature: 0.5,
            top_k: 40,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stream: false,
            seed: 42,
            device: DeviceKind::Cpu,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GpuInfoConfig {
    pub device: DeviceKind,
}

impl Default for GpuInfoConfig {
    fn default() -> Self {
        Self {
            device: DeviceKind::Auto,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrepareDataConfig {
    pub output_path: PathBuf,
    pub output_format: DataFormat,
    pub pretty: bool,
    pub dedup: bool,
    pub min_chars: usize,
    pub max_chars: usize,
    pub min_messages: usize,
    pub require_assistant: bool,
}

impl Default for PrepareDataConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("prepared.jsonl"),
            output_format: DataFormat::JsonlText,
            pretty: false,
            dedup: false,
            min_chars: 0,
            max_chars: 0,
            min_messages: 0,
            require_assistant: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TrainTokenizerConfig {
    pub output_path: PathBuf,
    pub model: TokenizerModelKind,
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub show_progress: bool,
}

impl Default for TrainTokenizerConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("tokenizer.json"),
            model: TokenizerModelKind::Bpe,
            vocab_size: 2048,
            min_frequency: 2,
            show_progress: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchmarkConfig {
    pub iterations: usize,
    pub warmup: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 5,
            warmup: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvalConfig {
    pub max_examples: usize,
    pub prompts: Vec<String>,
    pub prompt_files: Vec<PathBuf>,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub max_new_tokens: usize,
    pub device: DeviceKind,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_examples: 64,
            prompts: Vec::new(),
            prompt_files: Vec::new(),
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            max_new_tokens: 32,
            device: DeviceKind::Cpu,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InspectVocabConfig {
    pub boundary_mode: BoundaryMode,
    pub show_tokens: usize,
}

impl Default for InspectVocabConfig {
    fn default() -> Self {
        Self {
            boundary_mode: BoundaryMode::SharedBos,
            show_tokens: 32,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub tie_embeddings: bool,
    pub activation: ActivationKind,
    pub position_encoding: PositionEncodingKind,
    pub boundary_mode: BoundaryMode,
}

impl TrainConfig {
    pub fn to_model_config(&self, vocab_size: usize) -> ModelConfig {
        let n_kv_head = if self.n_kv_head == 0 {
            self.n_head
        } else {
            self.n_kv_head
        };
        ModelConfig {
            vocab_size,
            block_size: self.block_size,
            n_layer: self.n_layer,
            n_embd: self.n_embd,
            n_head: self.n_head,
            n_kv_head,
            tie_embeddings: self.tie_embeddings,
            activation: self.activation,
            position_encoding: self.position_encoding,
            boundary_mode: self.boundary_mode,
        }
    }
}
