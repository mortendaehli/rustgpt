//! Tokenizer abstractions used by training, evaluation, and chat.
//! The default educational path still uses a byte tokenizer, but the project can
//! now also load a Hugging Face style `tokenizer.json` asset for modern BPE setups.

mod byte;
mod hf;
mod training;

use std::fmt::{Display, Formatter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::core::config::BoundaryMode;
use crate::core::error::{Result, RustGptError};

pub(crate) use self::byte::ByteTokenizer;
pub(crate) use self::hf::HfTokenizer;
pub use self::training::train_tokenizer_from_dataset;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum TokenSymbol {
    Byte(u8),
    Piece(String),
    Bos,
    Eos,
    Special(String),
}

impl Display for TokenSymbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Byte(byte) => write!(f, "{}", display_byte(*byte)),
            Self::Piece(piece) | Self::Special(piece) => write!(f, "{piece}"),
            Self::Bos => write!(f, "<BOS>"),
            Self::Eos => write!(f, "<EOS>"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TokenizerImpl {
    Byte(Box<ByteTokenizer>),
    HuggingFace(HfTokenizer),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tokenizer {
    inner: TokenizerImpl,
}

impl Tokenizer {
    pub fn from_docs(docs: &[String], boundary_mode: BoundaryMode) -> Result<Self> {
        Ok(Self {
            inner: TokenizerImpl::Byte(Box::new(ByteTokenizer::from_docs(docs, boundary_mode)?)),
        })
    }

    pub fn from_tokenizer_file(
        path: impl AsRef<Path>,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<Self> {
        Ok(Self {
            inner: TokenizerImpl::HuggingFace(HfTokenizer::from_file(path, bos_token, eos_token)?),
        })
    }

    pub(crate) fn from_checkpoint_symbols(symbols: Vec<TokenSymbol>) -> Result<Self> {
        Ok(Self {
            inner: TokenizerImpl::Byte(Box::new(ByteTokenizer::from_symbols(symbols)?)),
        })
    }

    pub(crate) fn from_checkpoint_hf_json(
        json: String,
        bos_id: usize,
        eos_id: Option<usize>,
        boundary_mode: BoundaryMode,
    ) -> Result<Self> {
        Ok(Self {
            inner: TokenizerImpl::HuggingFace(HfTokenizer::from_json(
                json,
                bos_id,
                eos_id,
                boundary_mode,
            )?),
        })
    }

    pub fn kind_name(&self) -> &'static str {
        match &self.inner {
            TokenizerImpl::Byte(_) => "byte",
            TokenizerImpl::HuggingFace(_) => "huggingface",
        }
    }

    pub fn vocab_size(&self) -> usize {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.vocab_size(),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.vocab_size(),
        }
    }

    pub fn bos_id(&self) -> usize {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.bos_id(),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.bos_id(),
        }
    }

    pub fn eos_id(&self) -> Option<usize> {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.eos_id(),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.eos_id(),
        }
    }

    pub fn boundary_mode(&self) -> BoundaryMode {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.boundary_mode(),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.boundary_mode(),
        }
    }

    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.encode_text(text),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.encode_text(text),
        }
    }

    pub fn encode_with_boundaries(&self, text: &str) -> Result<Vec<usize>> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(self.bos_id());
        tokens.extend(self.encode_text(text));
        tokens.push(self.eos_id().unwrap_or(self.bos_id()));
        Ok(tokens)
    }

    pub fn decode(&self, token_ids: &[usize], skip_special: bool) -> Result<String> {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.decode(token_ids, skip_special),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.decode(token_ids, skip_special),
        }
    }

    pub fn decode_streaming(&self, token_ids: &[usize]) -> Result<String> {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.decode_streaming(token_ids),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.decode(token_ids, true),
        }
    }

    pub fn token_symbol(&self, token_id: usize) -> Result<TokenSymbol> {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => tokenizer.token_symbol(token_id),
            TokenizerImpl::HuggingFace(tokenizer) => tokenizer.token_symbol(token_id),
        }
    }

    pub fn token_label(&self, token_id: usize) -> Result<String> {
        Ok(self.token_symbol(token_id)?.to_string())
    }

    pub fn is_end_token(&self, token_id: usize) -> bool {
        token_id == self.eos_id().unwrap_or(self.bos_id())
    }

    pub(crate) fn byte(&self) -> Option<&ByteTokenizer> {
        match &self.inner {
            TokenizerImpl::Byte(tokenizer) => Some(tokenizer),
            TokenizerImpl::HuggingFace(_) => None,
        }
    }

    pub(crate) fn hf(&self) -> Option<&HfTokenizer> {
        match &self.inner {
            TokenizerImpl::Byte(_) => None,
            TokenizerImpl::HuggingFace(tokenizer) => Some(tokenizer),
        }
    }
}

fn display_byte(byte: u8) -> String {
    match byte {
        b'\n' => "\\n".to_string(),
        b'\r' => "\\r".to_string(),
        b'\t' => "\\t".to_string(),
        b' ' => "<space>".to_string(),
        0x21..=0x7e => format!("{:?}", byte as char),
        _ => format!("0x{byte:02X}"),
    }
}

pub(crate) fn invalid_token_id(token_id: usize) -> RustGptError {
    RustGptError::Tokenizer(format!("unknown token id {token_id}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokenizers::Tokenizer as RawTokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use crate::core::config::BoundaryMode;

    use super::{TokenSymbol, Tokenizer};

    #[test]
    fn shared_bos_mode_reuses_boundary_token() {
        let docs = vec!["emma".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let tokens = tokenizer.encode_with_boundaries("em").unwrap();
        assert_eq!(tokens[0], tokenizer.bos_id());
        assert_eq!(tokens[3], tokenizer.bos_id());
        assert_eq!(tokenizer.vocab_size(), 257);
    }

    #[test]
    fn separate_eos_adds_extra_special_token() {
        let docs = vec!["ab".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SeparateBosEos).unwrap();
        assert_eq!(tokenizer.vocab_size(), 258);
        assert_eq!(tokenizer.token_symbol(256).unwrap(), TokenSymbol::Bos);
        assert_eq!(tokenizer.token_symbol(257).unwrap(), TokenSymbol::Eos);
    }

    #[test]
    fn decode_can_skip_special_tokens() {
        let docs = vec!["luke".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SeparateBosEos).unwrap();
        let tokens = tokenizer.encode_with_boundaries("luke").unwrap();
        assert_eq!(tokenizer.decode(&tokens, true).unwrap(), "luke");
    }

    #[test]
    fn tokenizer_encodes_any_utf8_input_as_bytes() {
        let docs = vec!["hei".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let text = "Hei 👋 世界";
        let token_ids = tokenizer.encode_text(text);
        assert_eq!(tokenizer.decode(&token_ids, true).unwrap(), text);
    }

    #[test]
    fn streaming_decode_holds_back_incomplete_utf8() {
        let docs = vec!["hei".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let bytes = "👋".as_bytes();
        let partial = bytes[..2]
            .iter()
            .map(|byte| *byte as usize)
            .collect::<Vec<_>>();
        let full = bytes.iter().map(|byte| *byte as usize).collect::<Vec<_>>();
        assert_eq!(tokenizer.decode_streaming(&partial).unwrap(), "");
        assert_eq!(tokenizer.decode_streaming(&full).unwrap(), "👋");
    }

    #[test]
    fn loads_external_tokenizer_json() {
        let mut vocab = HashMap::new();
        vocab.insert("<s>".to_string(), 0);
        vocab.insert("</s>".to_string(), 1);
        vocab.insert("<unk>".to_string(), 2);
        vocab.insert("hello".to_string(), 3);

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut raw = RawTokenizer::new(model);
        raw.with_pre_tokenizer(Some(Whitespace));

        let path = unique_temp_path("external_tokenizer");
        raw.save(&path, false).unwrap();

        let tokenizer = Tokenizer::from_tokenizer_file(&path, Some("<s>"), Some("</s>")).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(tokenizer.kind_name(), "huggingface");
        assert_eq!(tokenizer.bos_id(), 0);
        assert_eq!(tokenizer.eos_id(), Some(1));
        assert_eq!(tokenizer.encode_text("hello"), vec![3]);
        assert_eq!(tokenizer.token_label(3).unwrap(), "hello");
    }

    fn unique_temp_path(stem: &str) -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rustgpt_{stem}_{suffix}_{}.json",
            std::process::id()
        ))
    }
}
