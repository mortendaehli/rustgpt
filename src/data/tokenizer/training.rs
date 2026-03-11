use std::path::PathBuf;

use tokenizers::AddedToken;
use tokenizers::TokenizerBuilder;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use crate::core::config::{ChatTemplateKind, TokenizerModelKind, TrainTokenizerConfig};
use crate::core::error::{Result, RustGptError};
use crate::data::corpus::Dataset;

use super::Tokenizer;

const DEFAULT_SPECIAL_TOKENS: &[&str] = &[
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerTrainingSummary {
    pub output_path: PathBuf,
    pub model: TokenizerModelKind,
    pub vocab_size: usize,
    pub bos_token: String,
    pub eos_token: String,
}

pub fn train_tokenizer_from_dataset(
    dataset: &Dataset,
    chat_template: ChatTemplateKind,
    config: &TrainTokenizerConfig,
) -> Result<TokenizerTrainingSummary> {
    match config.model {
        TokenizerModelKind::Bpe => train_bpe_tokenizer(dataset, chat_template, config),
    }
}

fn train_bpe_tokenizer(
    dataset: &Dataset,
    chat_template: ChatTemplateKind,
    config: &TrainTokenizerConfig,
) -> Result<TokenizerTrainingSummary> {
    let docs = dataset.docs_with_template(chat_template);
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(config.show_progress)
        .vocab_size(config.vocab_size)
        .min_frequency(config.min_frequency)
        .initial_alphabet(ByteLevel::alphabet())
        .special_tokens(
            DEFAULT_SPECIAL_TOKENS
                .iter()
                .map(|token| AddedToken::from((*token).to_string(), true))
                .collect(),
        )
        .build();

    let model = BPE::builder()
        .unk_token("<|unk|>".to_string())
        .build()
        .map_err(|source| {
            RustGptError::Tokenizer(format!("failed to build BPE model: {source}"))
        })?;
    let byte_level = ByteLevel::new(false, true, true);
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(model)
        .with_normalizer(Some(NFC))
        .with_pre_tokenizer(Some(byte_level))
        .with_post_processor(Some(byte_level))
        .with_decoder(Some(ByteLevelDecoder::default()))
        .build()
        .map_err(|source| {
            RustGptError::Tokenizer(format!("failed to build tokenizer: {source}"))
        })?;
    tokenizer
        .train(&mut trainer, docs.iter())
        .map_err(|source| {
            RustGptError::Tokenizer(format!(
                "failed to train tokenizer on {} documents: {source}",
                docs.len()
            ))
        })?;
    tokenizer
        .save(&config.output_path, false)
        .map_err(|source| {
            RustGptError::Tokenizer(format!(
                "failed to save tokenizer to {}: {source}",
                config.output_path.display()
            ))
        })?;

    let trained =
        Tokenizer::from_tokenizer_file(&config.output_path, Some("<|bos|>"), Some("<|eos|>"))?;

    Ok(TokenizerTrainingSummary {
        output_path: config.output_path.clone(),
        model: config.model,
        vocab_size: trained.vocab_size(),
        bos_token: "<|bos|>".to_string(),
        eos_token: "<|eos|>".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::config::{
        ChatTemplateKind, DataFormat, TokenizerModelKind, TrainTokenizerConfig,
    };
    use crate::data::corpus::Dataset;

    use super::train_tokenizer_from_dataset;

    #[test]
    fn trains_byte_level_bpe_that_roundtrips_utf8() {
        let dataset =
            Dataset::from_text("hello world\nHei 👋 verden", DataFormat::PlainText, false).unwrap();
        let path = temp_path("trained_tokenizer.json");
        let summary = train_tokenizer_from_dataset(
            &dataset,
            ChatTemplateKind::SimpleTranscript,
            &TrainTokenizerConfig {
                output_path: path.clone(),
                model: TokenizerModelKind::Bpe,
                vocab_size: 300,
                min_frequency: 1,
                show_progress: false,
            },
        )
        .unwrap();
        let tokenizer = crate::data::tokenizer::Tokenizer::from_tokenizer_file(
            &path,
            Some("<|bos|>"),
            Some("<|eos|>"),
        )
        .unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(summary.model, TokenizerModelKind::Bpe);
        let text = "Hej 👋 世界";
        let tokens = tokenizer.encode_text(text);
        assert_eq!(tokenizer.decode(&tokens, true).unwrap(), text);
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rustgpt_{label}_{nanos}"))
    }
}
