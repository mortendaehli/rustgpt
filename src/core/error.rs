use std::fmt::{Display, Formatter};
use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, RustGptError>;

#[derive(Debug)]
pub enum RustGptError {
    Io {
        path: Option<PathBuf>,
        source: std::io::Error,
    },
    Cli(String),
    Config(String),
    Data(String),
    Checkpoint(String),
    Tokenizer(String),
    Tensor(String),
    Gpu(String),
}

impl RustGptError {
    pub fn io(source: std::io::Error) -> Self {
        Self::Io { path: None, source }
    }

    pub fn io_with_path(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: Some(path.into()),
            source,
        }
    }
}

impl Display for RustGptError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io {
                path: Some(path),
                source,
            } => {
                write!(f, "I/O error for {}: {}", path.display(), source)
            }
            Self::Io { path: None, source } => write!(f, "I/O error: {source}"),
            Self::Cli(message) => write!(f, "{message}"),
            Self::Config(message) => write!(f, "{message}"),
            Self::Data(message) => write!(f, "{message}"),
            Self::Checkpoint(message) => write!(f, "{message}"),
            Self::Tokenizer(message) => write!(f, "{message}"),
            Self::Tensor(message) => write!(f, "{message}"),
            Self::Gpu(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for RustGptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RustGptError {
    fn from(source: std::io::Error) -> Self {
        Self::io(source)
    }
}
