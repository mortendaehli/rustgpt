use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::module::{AutodiffModule, Module};
use burn::optim::Optimizer;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder, RecorderError};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use crate::core::config::{BoundaryMode, ChatTemplateKind, ModelConfig};
use crate::core::error::{Result, RustGptError};
use crate::data::tokenizer::{TokenSymbol, Tokenizer};
use crate::engine::model::{LanguageModel, LanguageModelOptimizer, build_optimizer};

const CHECKPOINT_VERSION: u32 = 2;

#[derive(Clone, Debug)]
pub struct CheckpointMetadata {
    pub model_config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub trained_steps: usize,
    pub seed: u64,
    pub chat_template: ChatTemplateKind,
}

#[derive(Debug)]
pub struct LoadedCheckpoint<B: Backend> {
    pub model: LanguageModel<B>,
    pub tokenizer: Tokenizer,
    pub trained_steps: usize,
    pub seed: u64,
    pub chat_template: ChatTemplateKind,
}

pub struct LoadedTrainingCheckpoint<AD: AutodiffBackend> {
    pub model: LanguageModel<AD>,
    pub optimizer: LanguageModelOptimizer<AD>,
    pub tokenizer: Tokenizer,
    pub trained_steps: usize,
    pub seed: u64,
    pub chat_template: ChatTemplateKind,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CheckpointManifest {
    version: u32,
    model_config: ModelConfig,
    tokenizer: TokenizerAsset,
    trained_steps: usize,
    seed: u64,
    chat_template: ChatTemplateKind,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
enum TokenizerAsset {
    Byte {
        symbols: Vec<TokenSymbol>,
    },
    HuggingFace {
        json: String,
        bos_id: usize,
        eos_id: Option<usize>,
        boundary_mode: BoundaryMode,
    },
}

pub fn read_checkpoint_metadata(path: impl AsRef<Path>) -> Result<CheckpointMetadata> {
    let path = path.as_ref();
    let manifest = read_manifest(path)?;
    checkpoint_metadata_from_manifest(manifest)
}

pub fn load_inference_checkpoint<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedCheckpoint<B>> {
    let path = path.as_ref();
    let metadata = read_checkpoint_metadata(path)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model = LanguageModel::<B>::new(metadata.model_config.clone(), device)?
        .load_file(model_record_base(path), &recorder, device)
        .map_err(|err| checkpoint_error(path, err))?;

    Ok(LoadedCheckpoint {
        model,
        tokenizer: metadata.tokenizer,
        trained_steps: metadata.trained_steps,
        seed: metadata.seed,
        chat_template: metadata.chat_template,
    })
}

pub fn load_training_checkpoint<AD: AutodiffBackend>(
    path: impl AsRef<Path>,
    device: &AD::Device,
    train_config: &crate::core::config::TrainConfig,
) -> Result<LoadedTrainingCheckpoint<AD>> {
    let path = path.as_ref();
    let metadata = read_checkpoint_metadata(path)?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model = LanguageModel::<AD>::new(metadata.model_config.clone(), device)?
        .load_file(model_record_base(path), &recorder, device)
        .map_err(|err| checkpoint_error(path, err))?;

    let mut optimizer = build_optimizer::<AD>(train_config);
    match recorder.load(optimizer_record_base(path), device) {
        Ok(record) => {
            optimizer = optimizer.load_record(record);
        }
        Err(RecorderError::FileNotFound(_)) => {}
        Err(err) => return Err(checkpoint_error(path, err)),
    }

    Ok(LoadedTrainingCheckpoint {
        model,
        optimizer,
        tokenizer: metadata.tokenizer,
        trained_steps: metadata.trained_steps,
        seed: metadata.seed,
        chat_template: metadata.chat_template,
    })
}

pub fn save_training_checkpoint<AD: AutodiffBackend>(
    path: impl AsRef<Path>,
    model: &LanguageModel<AD>,
    optimizer: Option<&LanguageModelOptimizer<AD>>,
    tokenizer: &Tokenizer,
    chat_template: ChatTemplateKind,
    trained_steps: usize,
    seed: u64,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| RustGptError::io_with_path(parent, source))?;
    }

    let manifest = CheckpointManifest {
        version: CHECKPOINT_VERSION,
        model_config: model.config().clone(),
        tokenizer: TokenizerAsset::from_tokenizer(tokenizer)?,
        trained_steps,
        seed,
        chat_template,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|err| {
        RustGptError::Checkpoint(format!(
            "failed to serialize checkpoint manifest {}: {err}",
            path.display()
        ))
    })?;

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    save_record_atomically(
        &model_record_base(path),
        |record_path| model.valid().save_file(record_path, &recorder),
        path,
    )?;
    if let Some(optimizer) = optimizer {
        save_record_atomically(
            &optimizer_record_base(path),
            |record_path| recorder.record(optimizer.to_record(), record_path),
            path,
        )?;
    } else {
        remove_record_file_if_exists(&optimizer_record_base(path))?;
    }
    write_bytes_atomically(path, &manifest_bytes)?;

    Ok(())
}

impl TokenizerAsset {
    fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        if let Some(tokenizer) = tokenizer.byte() {
            return Ok(Self::Byte {
                symbols: tokenizer.symbols().to_vec(),
            });
        }
        if let Some(tokenizer) = tokenizer.hf() {
            return Ok(Self::HuggingFace {
                json: tokenizer.tokenizer_json().to_string(),
                bos_id: tokenizer.bos_id(),
                eos_id: tokenizer.eos_id(),
                boundary_mode: tokenizer.boundary_mode(),
            });
        }

        Err(RustGptError::Checkpoint(
            "unsupported tokenizer asset in checkpoint".to_string(),
        ))
    }

    fn into_tokenizer(self) -> Result<Tokenizer> {
        match self {
            Self::Byte { symbols } => Tokenizer::from_checkpoint_symbols(symbols),
            Self::HuggingFace {
                json,
                bos_id,
                eos_id,
                boundary_mode,
            } => Tokenizer::from_checkpoint_hf_json(json, bos_id, eos_id, boundary_mode),
        }
    }
}

fn read_manifest(path: &Path) -> Result<CheckpointManifest> {
    let bytes = fs::read(path).map_err(|source| RustGptError::io_with_path(path, source))?;
    if bytes.starts_with(b"RGPTCKP1") {
        return Err(RustGptError::Checkpoint(
            "legacy checkpoints are no longer supported by the Burn runtime; retrain or export into the new manifest format".to_string(),
        ));
    }
    let manifest: CheckpointManifest = serde_json::from_slice(&bytes).map_err(|err| {
        RustGptError::Checkpoint(format!(
            "failed to parse checkpoint manifest {}: {err}",
            path.display()
        ))
    })?;
    if manifest.version != CHECKPOINT_VERSION {
        return Err(RustGptError::Checkpoint(format!(
            "unsupported Burn checkpoint manifest version {} for {}; expected {}",
            manifest.version,
            path.display(),
            CHECKPOINT_VERSION
        )));
    }

    Ok(manifest)
}

fn model_record_base(path: &Path) -> PathBuf {
    sibling_path(path, "model")
}

fn optimizer_record_base(path: &Path) -> PathBuf {
    sibling_path(path, "optimizer")
}

fn sibling_path(path: &Path, suffix: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    path.with_file_name(format!("{}-{suffix}", sanitize_record_stem(stem)))
}

fn checkpoint_metadata_from_manifest(manifest: CheckpointManifest) -> Result<CheckpointMetadata> {
    Ok(CheckpointMetadata {
        model_config: manifest.model_config,
        tokenizer: manifest.tokenizer.into_tokenizer()?,
        trained_steps: manifest.trained_steps,
        seed: manifest.seed,
        chat_template: manifest.chat_template,
    })
}

fn save_record_atomically<F>(record_base: &Path, save: F, checkpoint_path: &Path) -> Result<()>
where
    F: FnOnce(PathBuf) -> std::result::Result<(), RecorderError>,
{
    let temp_base = temporary_record_base(record_base);
    save(temp_base.clone()).map_err(|err| checkpoint_error(checkpoint_path, err))?;
    rename_record_file(&temp_base, record_base)?;
    Ok(())
}

fn rename_record_file(from_base: &Path, to_base: &Path) -> Result<()> {
    let from = record_file_path(from_base);
    let to = record_file_path(to_base);
    fs::rename(&from, &to).map_err(|source| RustGptError::io_with_path(&to, source))
}

fn remove_record_file_if_exists(record_base: &Path) -> Result<()> {
    let record_path = record_file_path(record_base);
    if record_path.exists() {
        fs::remove_file(&record_path)
            .map_err(|source| RustGptError::io_with_path(&record_path, source))?;
    }

    Ok(())
}

fn record_file_path(record_base: &Path) -> PathBuf {
    let mut path = record_base.to_path_buf();
    path.set_extension("bin");
    path
}

fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_path = temporary_path(path);
    fs::write(&temp_path, bytes)
        .map_err(|source| RustGptError::io_with_path(&temp_path, source))?;
    fs::rename(&temp_path, path).map_err(|source| RustGptError::io_with_path(path, source))?;
    Ok(())
}

fn temporary_record_base(record_base: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = record_base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    record_base.with_file_name(format!("{file_name}-tmp-{nonce}"))
}

fn temporary_path(path: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    path.with_file_name(format!(".{file_name}.tmp-{nonce}"))
}

fn checkpoint_error(path: &Path, err: RecorderError) -> RustGptError {
    RustGptError::Checkpoint(format!(
        "checkpoint sidecar error for {}: {err}",
        path.display()
    ))
}

fn sanitize_record_stem(stem: &str) -> String {
    let mut sanitized = String::with_capacity(stem.len());
    for ch in stem.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' => sanitized.push(ch),
            _ => {
                sanitized.push('_');
                let _ = write!(&mut sanitized, "{:x}", ch as u32);
            }
        }
    }
    sanitized
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};

    use crate::core::config::{BoundaryMode, ChatTemplateKind, ModelConfig, TrainConfig};
    use crate::data::tokenizer::Tokenizer;
    use crate::engine::model::{LanguageModel, build_optimizer};

    use super::{
        CHECKPOINT_VERSION, load_training_checkpoint, model_record_base, optimizer_record_base,
        read_checkpoint_metadata, record_file_path, save_training_checkpoint,
    };

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<TestBackend>;

    fn checkpoint_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("rustgpt-{name}-{unique}.ckpt"))
    }

    fn test_config(vocab_size: usize) -> ModelConfig {
        ModelConfig {
            vocab_size,
            block_size: 8,
            n_layer: 1,
            n_embd: 8,
            n_head: 2,
            n_kv_head: 2,
            tie_embeddings: false,
            activation: crate::core::config::ActivationKind::Relu,
            position_encoding: crate::core::config::PositionEncodingKind::LearnedAbsolute,
            boundary_mode: BoundaryMode::SharedBos,
        }
    }

    #[test]
    fn record_paths_are_distinct_after_burn_extension_is_applied() {
        let checkpoint = PathBuf::from("run.ckpt");
        let model_path = record_file_path(&model_record_base(&checkpoint));
        let optimizer_path = record_file_path(&optimizer_record_base(&checkpoint));

        assert_ne!(model_path, optimizer_path);
        assert!(model_path.ends_with("run-model.bin"));
        assert!(optimizer_path.ends_with("run-optimizer.bin"));
    }

    #[test]
    fn record_paths_preserve_dotted_checkpoint_stems() {
        let checkpoint = PathBuf::from("run.best.ckpt");
        let model_path = record_file_path(&model_record_base(&checkpoint));
        let optimizer_path = record_file_path(&optimizer_record_base(&checkpoint));

        assert!(model_path.ends_with("run_2ebest-model.bin"));
        assert!(optimizer_path.ends_with("run_2ebest-optimizer.bin"));
    }

    #[test]
    fn training_checkpoint_round_trips_model_and_optimizer_sidecars() {
        let checkpoint = checkpoint_path("roundtrip");
        let device = NdArrayDevice::Cpu;
        let tokenizer =
            Tokenizer::from_docs(&["hello world".to_string()], BoundaryMode::SharedBos).unwrap();
        let model_config = test_config(tokenizer.vocab_size());
        let model =
            LanguageModel::<TestAutodiffBackend>::new(model_config.clone(), &device).unwrap();
        let optimizer = build_optimizer::<TestAutodiffBackend>(&TrainConfig::default());

        save_training_checkpoint(
            &checkpoint,
            &model,
            Some(&optimizer),
            &tokenizer,
            ChatTemplateKind::SimpleTranscript,
            12,
            7,
        )
        .unwrap();

        let metadata = read_checkpoint_metadata(&checkpoint).unwrap();
        assert_eq!(metadata.model_config, model_config);
        assert_eq!(metadata.trained_steps, 12);
        assert_eq!(metadata.seed, 7);

        let loaded = load_training_checkpoint::<TestAutodiffBackend>(
            &checkpoint,
            &device,
            &TrainConfig::default(),
        )
        .unwrap();
        assert_eq!(loaded.trained_steps, 12);
        assert_eq!(loaded.seed, 7);
        assert_eq!(loaded.chat_template, ChatTemplateKind::SimpleTranscript);

        fs::remove_file(&checkpoint).ok();
        fs::remove_file(record_file_path(&model_record_base(&checkpoint))).ok();
        fs::remove_file(record_file_path(&optimizer_record_base(&checkpoint))).ok();
    }

    #[test]
    fn manifest_version_matches_current_checkpoint_format() {
        assert_eq!(CHECKPOINT_VERSION, 2);
    }
}
