use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataFormat {
    Lines,
    PlainText,
}

impl DataFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "lines" => Some(Self::Lines),
            "text" | "plain-text" => Some(Self::PlainText),
            _ => None,
        }
    }
}

impl Display for DataFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lines => write!(f, "lines"),
            Self::PlainText => write!(f, "plain-text"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataConfig {
    pub data_path: PathBuf,
    pub format: DataFormat,
    pub shuffle: bool,
    pub lowercase: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("input.txt"),
            format: DataFormat::Lines,
            shuffle: true,
            lowercase: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps_adam: f32,
    pub seed: u64,
    pub sample_every: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub boundary_mode: BoundaryMode,
    pub device: DeviceKind,
    pub profile: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 1_000,
            batch_size: 1,
            learning_rate: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            eps_adam: 1e-8,
            seed: 42,
            sample_every: 100,
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            boundary_mode: BoundaryMode::SharedBos,
            device: DeviceKind::Cpu,
            profile: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampleConfig {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
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
            samples: 5,
            seed: 42,
            device: DeviceKind::Cpu,
            profile: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatConfig {
    pub system_prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
    pub device: DeviceKind,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_new_tokens: 32,
            temperature: 0.5,
            seed: 42,
            device: DeviceKind::Cpu,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub boundary_mode: BoundaryMode,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

impl TrainConfig {
    pub fn to_model_config(&self, vocab_size: usize) -> ModelConfig {
        ModelConfig {
            vocab_size,
            block_size: self.block_size,
            n_layer: self.n_layer,
            n_embd: self.n_embd,
            n_head: self.n_head,
            boundary_mode: self.boundary_mode,
        }
    }
}
