use std::fs;
use std::path::Path;
use std::sync::Arc;

use tokenizers::Tokenizer as HfTokenizerImpl;

use crate::core::config::BoundaryMode;
use crate::core::error::{Result, RustGptError};

use super::{TokenSymbol, invalid_token_id};

const BOS_CANDIDATES: &[&str] = &[
    "<s>",
    "<bos>",
    "<BOS>",
    "<|bos|>",
    "<|begin_of_text|>",
    "<|endoftext|>",
];
const EOS_CANDIDATES: &[&str] = &[
    "</s>",
    "<eos>",
    "<EOS>",
    "<|eos|>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|endoftext|>",
];

#[derive(Clone, Debug)]
pub(crate) struct HfTokenizer {
    tokenizer_json: String,
    tokenizer: Arc<HfTokenizerImpl>,
    bos_id: usize,
    eos_id: Option<usize>,
    boundary_mode: BoundaryMode,
}

impl PartialEq for HfTokenizer {
    fn eq(&self, other: &Self) -> bool {
        self.tokenizer_json == other.tokenizer_json
            && self.bos_id == other.bos_id
            && self.eos_id == other.eos_id
            && self.boundary_mode == other.boundary_mode
    }
}

impl Eq for HfTokenizer {}

impl HfTokenizer {
    pub(crate) fn from_file(
        path: impl AsRef<Path>,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer_json =
            fs::read_to_string(path).map_err(|source| RustGptError::io_with_path(path, source))?;
        let tokenizer = HfTokenizerImpl::from_file(path).map_err(|source| {
            RustGptError::Tokenizer(format!(
                "failed to load tokenizer asset {}: {source}",
                path.display()
            ))
        })?;
        Self::build(tokenizer_json, tokenizer, bos_token, eos_token)
    }

    pub(crate) fn from_json(
        tokenizer_json: String,
        bos_id: usize,
        eos_id: Option<usize>,
        boundary_mode: BoundaryMode,
    ) -> Result<Self> {
        let tokenizer =
            HfTokenizerImpl::from_bytes(tokenizer_json.as_bytes()).map_err(|source| {
                RustGptError::Tokenizer(format!(
                    "failed to parse tokenizer JSON from checkpoint: {source}"
                ))
            })?;
        Ok(Self {
            tokenizer_json,
            tokenizer: Arc::new(tokenizer),
            bos_id,
            eos_id,
            boundary_mode,
        })
    }

    fn build(
        tokenizer_json: String,
        tokenizer: HfTokenizerImpl,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<Self> {
        let bos_id = resolve_special_id(&tokenizer, bos_token, BOS_CANDIDATES, "BOS")?;
        let eos_id = resolve_optional_special_id(&tokenizer, eos_token, EOS_CANDIDATES);
        let boundary_mode = if eos_id.is_some() {
            BoundaryMode::SeparateBosEos
        } else {
            BoundaryMode::SharedBos
        };
        Ok(Self {
            tokenizer_json,
            tokenizer: Arc::new(tokenizer),
            bos_id,
            eos_id,
            boundary_mode,
        })
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    pub(crate) fn bos_id(&self) -> usize {
        self.bos_id
    }

    pub(crate) fn eos_id(&self) -> Option<usize> {
        self.eos_id
    }

    pub(crate) fn boundary_mode(&self) -> BoundaryMode {
        self.boundary_mode
    }

    pub(crate) fn tokenizer_json(&self) -> &str {
        &self.tokenizer_json
    }

    pub(crate) fn encode_text(&self, text: &str) -> Vec<usize> {
        self.tokenizer
            .encode(text, false)
            .expect("validated tokenizer must encode text")
            .get_ids()
            .iter()
            .map(|token_id| *token_id as usize)
            .collect()
    }

    pub(crate) fn decode(&self, token_ids: &[usize], skip_special: bool) -> Result<String> {
        let ids = token_ids
            .iter()
            .map(|token_id| *token_id as u32)
            .collect::<Vec<_>>();
        self.tokenizer
            .decode(&ids, skip_special)
            .map_err(|source| RustGptError::Tokenizer(format!("failed to decode tokens: {source}")))
    }

    pub(crate) fn token_symbol(&self, token_id: usize) -> Result<TokenSymbol> {
        let token = self
            .tokenizer
            .id_to_token(token_id as u32)
            .ok_or_else(|| invalid_token_id(token_id))?;
        if token_id == self.bos_id {
            Ok(TokenSymbol::Bos)
        } else if self.eos_id == Some(token_id) {
            Ok(TokenSymbol::Eos)
        } else if token.starts_with('<') && token.ends_with('>') || token.starts_with("<|") {
            Ok(TokenSymbol::Special(token))
        } else {
            Ok(TokenSymbol::Piece(token))
        }
    }
}

fn resolve_special_id(
    tokenizer: &HfTokenizerImpl,
    override_token: Option<&str>,
    candidates: &[&str],
    label: &str,
) -> Result<usize> {
    if let Some(token) = override_token {
        return tokenizer
            .token_to_id(token)
            .map(|token_id| token_id as usize)
            .ok_or_else(|| {
                RustGptError::Tokenizer(format!(
                    "{label} token {token:?} was not found in tokenizer vocabulary"
                ))
            });
    }

    for token in candidates {
        if let Some(token_id) = tokenizer.token_to_id(token) {
            return Ok(token_id as usize);
        }
    }

    Err(RustGptError::Tokenizer(format!(
        "could not infer a {label} token from tokenizer.json; pass --bos-token/--eos-token explicitly"
    )))
}

fn resolve_optional_special_id(
    tokenizer: &HfTokenizerImpl,
    override_token: Option<&str>,
    candidates: &[&str],
) -> Option<usize> {
    if let Some(token) = override_token {
        return tokenizer
            .token_to_id(token)
            .map(|token_id| token_id as usize);
    }
    candidates.iter().find_map(|token| {
        tokenizer
            .token_to_id(token)
            .map(|token_id| token_id as usize)
    })
}
