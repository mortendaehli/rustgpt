use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::core::config::{BoundaryMode, ModelConfig};
use crate::core::error::{Result, RustGptError};
use crate::core::tensor::Matrix;
use crate::data::tokenizer::{TokenSymbol, Tokenizer};
use crate::model::{Model, TransformerLayer};

const MAGIC: &[u8; 8] = b"RGPTCKP1";
const VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub trained_steps: usize,
    pub seed: u64,
}

pub fn save_checkpoint(
    path: impl AsRef<Path>,
    model: &Model,
    tokenizer: &Tokenizer,
    trained_steps: usize,
    seed: u64,
) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).map_err(|source| RustGptError::io_with_path(path, source))?;
    let mut writer = BufWriter::new(file);

    writer
        .write_all(MAGIC)
        .map_err(|source| RustGptError::io_with_path(path, source))?;
    write_u32(&mut writer, VERSION, path)?;
    write_u64(&mut writer, trained_steps as u64, path)?;
    write_u64(&mut writer, seed, path)?;

    write_model_config(&mut writer, &model.cfg, path)?;
    write_tokenizer(&mut writer, tokenizer, path)?;

    write_matrix(&mut writer, &model.wte, path)?;
    write_matrix(&mut writer, &model.wpe, path)?;
    write_matrix(&mut writer, &model.lm_head, path)?;
    for layer in &model.layers {
        write_matrix(&mut writer, &layer.attn_wq, path)?;
        write_matrix(&mut writer, &layer.attn_wk, path)?;
        write_matrix(&mut writer, &layer.attn_wv, path)?;
        write_matrix(&mut writer, &layer.attn_wo, path)?;
        write_matrix(&mut writer, &layer.mlp_fc1, path)?;
        write_matrix(&mut writer, &layer.mlp_fc2, path)?;
    }

    writer
        .flush()
        .map_err(|source| RustGptError::io_with_path(path, source))?;
    Ok(())
}

pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<Checkpoint> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| RustGptError::io_with_path(path, source))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|source| RustGptError::io_with_path(path, source))?;
    if &magic != MAGIC {
        return Err(RustGptError::Checkpoint(format!(
            "invalid checkpoint magic in {}",
            path.display()
        )));
    }

    let version = read_u32(&mut reader, path)?;
    if version == 1 {
        return Err(RustGptError::Checkpoint(
            "checkpoint version 1 uses the legacy char tokenizer and is no longer supported; retrain to create a version 2 byte-token checkpoint".to_string(),
        ));
    }
    if version != VERSION {
        return Err(RustGptError::Checkpoint(format!(
            "unsupported checkpoint version {version} in {}",
            path.display()
        )));
    }

    let trained_steps = read_u64(&mut reader, path)? as usize;
    let seed = read_u64(&mut reader, path)?;
    let cfg = read_model_config(&mut reader, path)?;
    let tokenizer = read_tokenizer(&mut reader, path)?;

    let wte = read_matrix(&mut reader, path)?;
    let wpe = read_matrix(&mut reader, path)?;
    let lm_head = read_matrix(&mut reader, path)?;
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for _ in 0..cfg.n_layer {
        layers.push(TransformerLayer {
            attn_wq: read_matrix(&mut reader, path)?,
            attn_wk: read_matrix(&mut reader, path)?,
            attn_wv: read_matrix(&mut reader, path)?,
            attn_wo: read_matrix(&mut reader, path)?,
            mlp_fc1: read_matrix(&mut reader, path)?,
            mlp_fc2: read_matrix(&mut reader, path)?,
        });
    }

    let model = Model::from_parts(cfg, wte, wpe, lm_head, layers)?;
    Ok(Checkpoint {
        model,
        tokenizer,
        trained_steps,
        seed,
    })
}

fn write_model_config(writer: &mut BufWriter<File>, cfg: &ModelConfig, path: &Path) -> Result<()> {
    write_u64(writer, cfg.vocab_size as u64, path)?;
    write_u64(writer, cfg.block_size as u64, path)?;
    write_u64(writer, cfg.n_layer as u64, path)?;
    write_u64(writer, cfg.n_embd as u64, path)?;
    write_u64(writer, cfg.n_head as u64, path)?;
    write_u8(
        writer,
        match cfg.boundary_mode {
            BoundaryMode::SharedBos => 0,
            BoundaryMode::SeparateBosEos => 1,
        },
        path,
    )
}

fn read_model_config(reader: &mut BufReader<File>, path: &Path) -> Result<ModelConfig> {
    let vocab_size = read_u64(reader, path)? as usize;
    let block_size = read_u64(reader, path)? as usize;
    let n_layer = read_u64(reader, path)? as usize;
    let n_embd = read_u64(reader, path)? as usize;
    let n_head = read_u64(reader, path)? as usize;
    let boundary_mode = match read_u8(reader, path)? {
        0 => BoundaryMode::SharedBos,
        1 => BoundaryMode::SeparateBosEos,
        other => {
            return Err(RustGptError::Checkpoint(format!(
                "invalid boundary mode tag {other} in {}",
                path.display()
            )));
        }
    };

    Ok(ModelConfig {
        vocab_size,
        block_size,
        n_layer,
        n_embd,
        n_head,
        boundary_mode,
    })
}

fn write_tokenizer(writer: &mut BufWriter<File>, tokenizer: &Tokenizer, path: &Path) -> Result<()> {
    write_u64(writer, tokenizer.vocab_size() as u64, path)?;
    for symbol in tokenizer.symbols() {
        match symbol {
            TokenSymbol::Byte(byte) => {
                write_u8(writer, 0, path)?;
                write_u8(writer, *byte, path)?;
            }
            TokenSymbol::Bos => write_u8(writer, 1, path)?,
            TokenSymbol::Eos => write_u8(writer, 2, path)?,
        }
    }
    Ok(())
}

fn read_tokenizer(reader: &mut BufReader<File>, path: &Path) -> Result<Tokenizer> {
    let symbol_count = read_u64(reader, path)? as usize;
    let mut symbols = Vec::with_capacity(symbol_count);
    for _ in 0..symbol_count {
        let tag = read_u8(reader, path)?;
        let symbol = match tag {
            0 => TokenSymbol::Byte(read_u8(reader, path)?),
            1 => TokenSymbol::Bos,
            2 => TokenSymbol::Eos,
            other => {
                return Err(RustGptError::Checkpoint(format!(
                    "invalid tokenizer symbol tag {other} in {}",
                    path.display()
                )));
            }
        };
        symbols.push(symbol);
    }
    Tokenizer::from_symbols(symbols)
}

fn write_matrix(writer: &mut BufWriter<File>, matrix: &Matrix, path: &Path) -> Result<()> {
    write_u64(writer, matrix.rows as u64, path)?;
    write_u64(writer, matrix.cols as u64, path)?;
    write_f32_vec(writer, &matrix.data, path)?;
    write_f32_vec(writer, &matrix.m, path)?;
    write_f32_vec(writer, &matrix.v, path)?;
    Ok(())
}

fn read_matrix(reader: &mut BufReader<File>, path: &Path) -> Result<Matrix> {
    let rows = read_u64(reader, path)? as usize;
    let cols = read_u64(reader, path)? as usize;
    let data = read_f32_vec(reader, path)?;
    let m = read_f32_vec(reader, path)?;
    let v = read_f32_vec(reader, path)?;
    Matrix::from_parts(rows, cols, data, m, v)
}

fn write_f32_vec(writer: &mut BufWriter<File>, values: &[f32], path: &Path) -> Result<()> {
    write_u64(writer, values.len() as u64, path)?;
    for value in values {
        write_f32(writer, *value, path)?;
    }
    Ok(())
}

fn read_f32_vec(reader: &mut BufReader<File>, path: &Path) -> Result<Vec<f32>> {
    let len = read_u64(reader, path)? as usize;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_f32(reader, path)?);
    }
    Ok(out)
}

fn write_u8(writer: &mut BufWriter<File>, value: u8, path: &Path) -> Result<()> {
    writer
        .write_all(&[value])
        .map_err(|source| RustGptError::io_with_path(path, source))
}

fn read_u8(reader: &mut BufReader<File>, path: &Path) -> Result<u8> {
    let mut buf = [0_u8; 1];
    reader
        .read_exact(&mut buf)
        .map_err(|source| RustGptError::io_with_path(path, source))?;
    Ok(buf[0])
}

fn write_u32(writer: &mut BufWriter<File>, value: u32, path: &Path) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|source| RustGptError::io_with_path(path, source))
}

fn read_u32(reader: &mut BufReader<File>, path: &Path) -> Result<u32> {
    let mut buf = [0_u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|source| RustGptError::io_with_path(path, source))?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u64(writer: &mut BufWriter<File>, value: u64, path: &Path) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|source| RustGptError::io_with_path(path, source))
}

fn read_u64(reader: &mut BufReader<File>, path: &Path) -> Result<u64> {
    let mut buf = [0_u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|source| RustGptError::io_with_path(path, source))?;
    Ok(u64::from_le_bytes(buf))
}

fn write_f32(writer: &mut BufWriter<File>, value: f32, path: &Path) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|source| RustGptError::io_with_path(path, source))
}

fn read_f32(reader: &mut BufReader<File>, path: &Path) -> Result<f32> {
    let mut buf = [0_u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|source| RustGptError::io_with_path(path, source))?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::config::{BoundaryMode, ModelConfig};
    use crate::core::rng::Rng;

    use super::{load_checkpoint, save_checkpoint};
    use crate::data::tokenizer::Tokenizer;
    use crate::model::Model;

    #[test]
    fn checkpoint_roundtrip_restores_model_and_tokenizer() {
        let docs = vec!["emma".to_string()];
        let tokenizer = Tokenizer::from_docs(&docs, BoundaryMode::SeparateBosEos).unwrap();
        let cfg = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            block_size: 16,
            n_layer: 1,
            n_embd: 16,
            n_head: 4,
            boundary_mode: BoundaryMode::SeparateBosEos,
        };
        let mut rng = Rng::from_seed(42);
        let model = Model::new(cfg, &mut rng).unwrap();

        let path = unique_temp_path("checkpoint_roundtrip");
        save_checkpoint(&path, &model, &tokenizer, 12, 42).unwrap();
        let loaded = load_checkpoint(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.trained_steps, 12);
        assert_eq!(loaded.seed, 42);
        assert_eq!(loaded.tokenizer, tokenizer);
        assert_eq!(loaded.model, model);
    }

    fn unique_temp_path(stem: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rustgpt_{stem}_{suffix}_{}.ckpt",
            std::process::id()
        ))
    }
}
