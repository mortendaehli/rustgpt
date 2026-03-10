//! Byte-level tokenizer.
//! The model always sees raw UTF-8 bytes plus BOS/EOS, which keeps the code compact
//! and guarantees that any UTF-8 input can be encoded without an unknown-token path.

use std::fmt::{Display, Formatter};

use crate::core::config::BoundaryMode;
use crate::core::error::{Result, RustGptError};

const BYTE_VOCAB_SIZE: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenSymbol {
    Byte(u8),
    Bos,
    Eos,
}

impl Display for TokenSymbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Byte(byte) => write!(f, "{}", display_byte(*byte)),
            Self::Bos => write!(f, "<BOS>"),
            Self::Eos => write!(f, "<EOS>"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tokenizer {
    byte_to_id: [usize; BYTE_VOCAB_SIZE],
    id_to_symbol: Vec<TokenSymbol>,
    bos_id: usize,
    eos_id: Option<usize>,
    boundary_mode: BoundaryMode,
}

impl Tokenizer {
    pub fn from_docs(docs: &[String], boundary_mode: BoundaryMode) -> Result<Self> {
        if docs.is_empty() {
            return Err(RustGptError::Tokenizer(
                "cannot build tokenizer from an empty dataset".to_string(),
            ));
        }
        if docs.iter().all(|doc| doc.is_empty()) {
            return Err(RustGptError::Tokenizer(
                "dataset does not contain any tokenizable bytes".to_string(),
            ));
        }

        let mut id_to_symbol = (0_u16..=255)
            .map(|byte| TokenSymbol::Byte(byte as u8))
            .collect::<Vec<_>>();
        let mut byte_to_id = [usize::MAX; BYTE_VOCAB_SIZE];
        for byte in 0..BYTE_VOCAB_SIZE {
            byte_to_id[byte] = byte;
        }

        let bos_id = id_to_symbol.len();
        id_to_symbol.push(TokenSymbol::Bos);
        let eos_id = if matches!(boundary_mode, BoundaryMode::SeparateBosEos) {
            let id = id_to_symbol.len();
            id_to_symbol.push(TokenSymbol::Eos);
            Some(id)
        } else {
            None
        };

        Ok(Self {
            byte_to_id,
            id_to_symbol,
            bos_id,
            eos_id,
            boundary_mode,
        })
    }

    pub fn from_symbols(symbols: Vec<TokenSymbol>) -> Result<Self> {
        if symbols.is_empty() {
            return Err(RustGptError::Tokenizer(
                "cannot build tokenizer from empty symbols".to_string(),
            ));
        }

        let mut byte_to_id = [usize::MAX; BYTE_VOCAB_SIZE];
        let mut bos_id = None;
        let mut eos_id = None;

        for (idx, symbol) in symbols.iter().enumerate() {
            match symbol {
                TokenSymbol::Byte(byte) => {
                    let slot = &mut byte_to_id[*byte as usize];
                    if *slot != usize::MAX {
                        return Err(RustGptError::Tokenizer(format!(
                            "duplicate byte token 0x{byte:02X} in checkpoint symbols"
                        )));
                    }
                    *slot = idx;
                }
                TokenSymbol::Bos => {
                    if bos_id.replace(idx).is_some() {
                        return Err(RustGptError::Tokenizer(
                            "checkpoint tokenizer contains multiple BOS tokens".to_string(),
                        ));
                    }
                }
                TokenSymbol::Eos => {
                    if eos_id.replace(idx).is_some() {
                        return Err(RustGptError::Tokenizer(
                            "checkpoint tokenizer contains multiple EOS tokens".to_string(),
                        ));
                    }
                }
            }
        }

        if let Some(missing) = byte_to_id.iter().position(|id| *id == usize::MAX) {
            return Err(RustGptError::Tokenizer(format!(
                "checkpoint tokenizer is missing byte token 0x{missing:02X}"
            )));
        }

        let bos_id = bos_id.ok_or_else(|| {
            RustGptError::Tokenizer("checkpoint tokenizer is missing BOS".to_string())
        })?;
        let boundary_mode = if eos_id.is_some() {
            BoundaryMode::SeparateBosEos
        } else {
            BoundaryMode::SharedBos
        };

        Ok(Self {
            byte_to_id,
            id_to_symbol: symbols,
            bos_id,
            eos_id,
            boundary_mode,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_symbol.len()
    }

    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<usize> {
        self.eos_id
    }

    pub fn boundary_mode(&self) -> BoundaryMode {
        self.boundary_mode
    }

    pub fn symbols(&self) -> &[TokenSymbol] {
        &self.id_to_symbol
    }

    pub fn symbol(&self, token_id: usize) -> Result<&TokenSymbol> {
        self.id_to_symbol
            .get(token_id)
            .ok_or_else(|| RustGptError::Tokenizer(format!("unknown token id {token_id}")))
    }

    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        text.as_bytes()
            .iter()
            .map(|byte| self.byte_to_id[*byte as usize])
            .collect()
    }

    pub fn encode_with_boundaries(&self, text: &str) -> Result<Vec<usize>> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(self.bos_id);
        tokens.extend(self.encode_text(text));
        tokens.push(self.eos_id.unwrap_or(self.bos_id));
        Ok(tokens)
    }

    pub fn decode(&self, token_ids: &[usize], skip_special: bool) -> Result<String> {
        let mut decoded = String::new();
        let mut pending_bytes = Vec::new();

        for token_id in token_ids {
            match self
                .id_to_symbol
                .get(*token_id)
                .ok_or_else(|| RustGptError::Tokenizer(format!("unknown token id {token_id}")))?
            {
                TokenSymbol::Byte(byte) => pending_bytes.push(*byte),
                TokenSymbol::Bos | TokenSymbol::Eos if skip_special => {}
                TokenSymbol::Bos => {
                    flush_bytes(&mut decoded, &mut pending_bytes);
                    decoded.push_str("<BOS>");
                }
                TokenSymbol::Eos => {
                    flush_bytes(&mut decoded, &mut pending_bytes);
                    decoded.push_str("<EOS>");
                }
            }
        }

        flush_bytes(&mut decoded, &mut pending_bytes);
        Ok(decoded)
    }
}

fn flush_bytes(output: &mut String, pending_bytes: &mut Vec<u8>) {
    if pending_bytes.is_empty() {
        return;
    }
    output.push_str(&String::from_utf8_lossy(pending_bytes));
    pending_bytes.clear();
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

#[cfg(test)]
mod tests {
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
        assert_eq!(tokenizer.symbols()[256], TokenSymbol::Bos);
        assert_eq!(tokenizer.symbols()[257], TokenSymbol::Eos);
    }

    #[test]
    fn decode_can_skip_special_tokens() {
        let docs = vec!["luke".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SeparateBosEos).unwrap();
        let tokens = tokenizer.encode_with_boundaries("luke").unwrap();
        assert_eq!(tokenizer.decode(&tokens, true).unwrap(), "luke");
    }

    #[test]
    fn tokenizer_roundtrips_from_symbols() {
        let docs = vec!["ab".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SeparateBosEos).unwrap();
        let rebuilt = Tokenizer::from_symbols(tokenizer.symbols().to_vec()).unwrap();
        assert_eq!(rebuilt, tokenizer);
    }

    #[test]
    fn tokenizer_encodes_any_utf8_input_as_bytes() {
        let docs = vec!["hi".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SharedBos).unwrap();
        let text = "Hej 👋 Grüß dich";
        let token_ids = tokenizer.encode_text(text);
        assert_eq!(token_ids.len(), text.len());
        assert_eq!(tokenizer.decode(&token_ids, true).unwrap(), text);
    }
}
