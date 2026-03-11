use crate::core::config::BoundaryMode;
use crate::core::error::{Result, RustGptError};

use super::{TokenSymbol, invalid_token_id};

const BYTE_VOCAB_SIZE: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ByteTokenizer {
    byte_to_id: [usize; BYTE_VOCAB_SIZE],
    id_to_symbol: Vec<TokenSymbol>,
    bos_id: usize,
    eos_id: Option<usize>,
    boundary_mode: BoundaryMode,
}

impl ByteTokenizer {
    pub(crate) fn from_docs(docs: &[String], boundary_mode: BoundaryMode) -> Result<Self> {
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

    pub(crate) fn from_symbols(symbols: Vec<TokenSymbol>) -> Result<Self> {
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
                TokenSymbol::Piece(piece) | TokenSymbol::Special(piece) => {
                    return Err(RustGptError::Tokenizer(format!(
                        "byte tokenizer checkpoint contained non-byte symbol {piece:?}"
                    )));
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

    pub(crate) fn vocab_size(&self) -> usize {
        self.id_to_symbol.len()
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

    pub(crate) fn symbols(&self) -> &[TokenSymbol] {
        &self.id_to_symbol
    }

    pub(crate) fn encode_text(&self, text: &str) -> Vec<usize> {
        text.as_bytes()
            .iter()
            .map(|byte| self.byte_to_id[*byte as usize])
            .collect()
    }

    pub(crate) fn decode(&self, token_ids: &[usize], skip_special: bool) -> Result<String> {
        let mut decoded = String::new();
        let mut pending_bytes = Vec::new();

        for token_id in token_ids {
            match self
                .id_to_symbol
                .get(*token_id)
                .ok_or_else(|| invalid_token_id(*token_id))?
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
                TokenSymbol::Piece(piece) | TokenSymbol::Special(piece) => {
                    flush_bytes(&mut decoded, &mut pending_bytes);
                    decoded.push_str(piece);
                }
            }
        }

        flush_bytes(&mut decoded, &mut pending_bytes);
        Ok(decoded)
    }

    pub(crate) fn decode_streaming(&self, token_ids: &[usize]) -> Result<String> {
        let mut decoded = String::new();
        let mut pending_bytes = Vec::new();

        for token_id in token_ids {
            match self
                .id_to_symbol
                .get(*token_id)
                .ok_or_else(|| invalid_token_id(*token_id))?
            {
                TokenSymbol::Byte(byte) => pending_bytes.push(*byte),
                TokenSymbol::Bos | TokenSymbol::Eos => {}
                TokenSymbol::Piece(piece) | TokenSymbol::Special(piece) => {
                    flush_bytes_stable(&mut decoded, &mut pending_bytes);
                    decoded.push_str(piece);
                }
            }
        }

        flush_bytes_stable(&mut decoded, &mut pending_bytes);
        Ok(decoded)
    }

    pub(crate) fn token_symbol(&self, token_id: usize) -> Result<TokenSymbol> {
        self.id_to_symbol
            .get(token_id)
            .cloned()
            .ok_or_else(|| invalid_token_id(token_id))
    }
}

fn flush_bytes(output: &mut String, pending_bytes: &mut Vec<u8>) {
    if pending_bytes.is_empty() {
        return;
    }
    output.push_str(&String::from_utf8_lossy(pending_bytes));
    pending_bytes.clear();
}

fn flush_bytes_stable(output: &mut String, pending_bytes: &mut Vec<u8>) {
    if pending_bytes.is_empty() {
        return;
    }
    match std::str::from_utf8(pending_bytes) {
        Ok(valid) => {
            output.push_str(valid);
            pending_bytes.clear();
        }
        Err(error) => {
            let valid_up_to = error.valid_up_to();
            if valid_up_to > 0 {
                output.push_str(std::str::from_utf8(&pending_bytes[..valid_up_to]).unwrap());
                pending_bytes.drain(..valid_up_to);
            }
        }
    }
}
