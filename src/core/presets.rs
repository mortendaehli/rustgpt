use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::core::config::{
    ActivationKind, DeviceKind, LrScheduleKind, PositionEncodingKind, TrainConfig, TrainMode,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TrainPreset {
    DebugTiny,
    ClassSmall,
    ClassSerious,
    ClassChat,
}

impl TrainPreset {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "debug-tiny" | "debug_tiny" => Some(Self::DebugTiny),
            "class-small" | "class_small" => Some(Self::ClassSmall),
            "class-serious" | "class_serious" => Some(Self::ClassSerious),
            "class-chat" | "class_chat" => Some(Self::ClassChat),
            _ => None,
        }
    }

    pub fn train_config(self) -> TrainConfig {
        match self {
            Self::DebugTiny => TrainConfig {
                steps: 64,
                batch_size: 4,
                gradient_accumulation_steps: 1,
                activation_checkpointing: false,
                learning_rate: 3e-4,
                beta1: 0.9,
                beta2: 0.95,
                weight_decay: 0.1,
                warmup_steps: 8,
                grad_clip: 1.0,
                lr_schedule: LrScheduleKind::Cosine,
                validation_max_examples: 32,
                sample_every: 8,
                block_size: 128,
                n_layer: 4,
                n_embd: 256,
                n_head: 4,
                tie_embeddings: true,
                activation: ActivationKind::SwiGlu,
                position_encoding: PositionEncodingKind::Rope,
                device: DeviceKind::Cpu,
                ..TrainConfig::default()
            },
            Self::ClassSmall => TrainConfig {
                steps: 2_000,
                batch_size: 4,
                gradient_accumulation_steps: 2,
                activation_checkpointing: false,
                learning_rate: 3e-4,
                beta1: 0.9,
                beta2: 0.95,
                weight_decay: 0.1,
                warmup_steps: 100,
                grad_clip: 1.0,
                lr_schedule: LrScheduleKind::Cosine,
                validation_max_examples: 128,
                sample_every: 100,
                block_size: 256,
                n_layer: 8,
                n_embd: 512,
                n_head: 8,
                tie_embeddings: true,
                activation: ActivationKind::SwiGlu,
                position_encoding: PositionEncodingKind::Rope,
                device: DeviceKind::Auto,
                ..TrainConfig::default()
            },
            Self::ClassSerious => TrainConfig {
                steps: 8_000,
                batch_size: 2,
                gradient_accumulation_steps: 4,
                activation_checkpointing: true,
                learning_rate: 2.5e-4,
                beta1: 0.9,
                beta2: 0.95,
                weight_decay: 0.1,
                warmup_steps: 200,
                grad_clip: 1.0,
                lr_schedule: LrScheduleKind::Cosine,
                validation_max_examples: 256,
                sample_every: 200,
                block_size: 512,
                n_layer: 12,
                n_embd: 768,
                n_head: 12,
                tie_embeddings: true,
                activation: ActivationKind::SwiGlu,
                position_encoding: PositionEncodingKind::Rope,
                device: DeviceKind::Auto,
                ..TrainConfig::default()
            },
            Self::ClassChat => TrainConfig {
                steps: 1_500,
                batch_size: 2,
                gradient_accumulation_steps: 4,
                activation_checkpointing: true,
                learning_rate: 1e-4,
                beta1: 0.9,
                beta2: 0.95,
                weight_decay: 0.1,
                warmup_steps: 50,
                grad_clip: 1.0,
                lr_schedule: LrScheduleKind::Cosine,
                validation_max_examples: 128,
                sample_every: 50,
                block_size: 512,
                n_layer: 12,
                n_embd: 768,
                n_head: 12,
                tie_embeddings: true,
                activation: ActivationKind::SwiGlu,
                position_encoding: PositionEncodingKind::Rope,
                mode: TrainMode::Sft,
                device: DeviceKind::Auto,
                ..TrainConfig::default()
            },
        }
    }

    pub fn names() -> &'static str {
        "debug-tiny, class-small, class-serious, or class-chat"
    }
}

pub fn train_presets_help() -> String {
    [
        "debug-tiny (tiny smoke-test profile for correctness checks)",
        "class-small (small pretraining profile for classroom labs)",
        "class-serious (serious M5/16 GB profile for longer instructor runs)",
        "class-chat (chat SFT profile for resuming a pretrained base checkpoint)",
    ]
    .join("; ")
}

impl Display for TrainPreset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DebugTiny => write!(f, "debug-tiny"),
            Self::ClassSmall => write!(f, "class-small"),
            Self::ClassSerious => write!(f, "class-serious"),
            Self::ClassChat => write!(f, "class-chat"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::{ActivationKind, DeviceKind, LrScheduleKind, PositionEncodingKind};

    use super::TrainPreset;

    #[test]
    fn serious_profile_matches_modern_baseline_shape() {
        let train = TrainPreset::ClassSerious.train_config();
        assert_eq!(train.n_layer, 12);
        assert_eq!(train.n_embd, 768);
        assert_eq!(train.n_head, 12);
        assert_eq!(train.block_size, 512);
        assert_eq!(train.activation, ActivationKind::SwiGlu);
        assert_eq!(train.position_encoding, PositionEncodingKind::Rope);
        assert_eq!(train.lr_schedule, LrScheduleKind::Cosine);
        assert_eq!(train.device, DeviceKind::Auto);
        assert!(train.tie_embeddings);
    }
}
