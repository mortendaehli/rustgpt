use std::fs;

use crate::core::config::{DataConfig, DataFormat};
use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dataset {
    format: DataFormat,
    docs: Vec<String>,
}

impl Dataset {
    pub fn from_path(config: &DataConfig) -> Result<Self> {
        let raw = fs::read_to_string(&config.data_path)
            .map_err(|source| RustGptError::io_with_path(&config.data_path, source))?;
        Self::from_text(&raw, config.format, config.lowercase)
    }

    pub fn from_text(raw: &str, format: DataFormat, lowercase: bool) -> Result<Self> {
        let normalized = if lowercase {
            raw.to_lowercase()
        } else {
            raw.to_string()
        };

        let docs = match format {
            DataFormat::Lines => normalized
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            DataFormat::PlainText => {
                let text = normalized.trim();
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![text.to_string()]
                }
            }
        };

        if docs.is_empty() {
            return Err(RustGptError::Data(
                "dataset is empty after preprocessing".to_string(),
            ));
        }

        Ok(Self { format, docs })
    }

    pub fn docs(&self) -> &[String] {
        &self.docs
    }

    pub fn format(&self) -> DataFormat {
        self.format
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.docs.iter().map(|doc| doc.len()).sum()
    }

    pub fn shuffled(&self, rng: &mut Rng) -> Self {
        let mut docs = self.docs.clone();
        rng.shuffle(&mut docs);
        Self {
            format: self.format,
            docs,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::DataFormat;

    use super::Dataset;

    #[test]
    fn line_format_drops_blank_lines() {
        let dataset = Dataset::from_text("alice\n\n bob \n", DataFormat::Lines, false).unwrap();
        assert_eq!(dataset.docs(), &["alice".to_string(), "bob".to_string()]);
    }

    #[test]
    fn plain_text_format_keeps_single_document() {
        let dataset =
            Dataset::from_text("hello world\nagain", DataFormat::PlainText, false).unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.docs()[0], "hello world\nagain");
    }
}
