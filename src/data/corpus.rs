use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};

use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::Value;

use crate::core::config::{ChatTemplateKind, DataConfig, DataFormat};
use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;
use crate::data::schema::{DatasetRecord, parse_chat_value, parse_text_value};

#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    format: DataFormat,
    records: Vec<DatasetRecord>,
    rendered_docs: Vec<String>,
    total_bytes: usize,
}

impl Dataset {
    pub fn from_path(config: &DataConfig) -> Result<Self> {
        match config.format {
            DataFormat::Lines | DataFormat::PlainText => {
                let raw = fs::read_to_string(&config.data_path)
                    .map_err(|source| RustGptError::io_with_path(&config.data_path, source))?;
                Self::from_text(&raw, config.format, config.lowercase)
            }
            DataFormat::JsonlText | DataFormat::JsonlChat => {
                Self::from_jsonl_path(config.format, &config.data_path, config.lowercase)
            }
            DataFormat::ParquetText | DataFormat::ParquetChat => {
                Self::from_parquet_path(config.format, &config.data_path, config.lowercase)
            }
        }
    }

    pub fn from_text(raw: &str, format: DataFormat, lowercase: bool) -> Result<Self> {
        let normalized = if lowercase {
            raw.to_lowercase()
        } else {
            raw.to_string()
        };

        let records = match format {
            DataFormat::Lines => normalized
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(|line| {
                    DatasetRecord::Text(crate::data::schema::TextRecord {
                        text: line.to_string(),
                        source: None,
                        meta: None,
                    })
                })
                .collect::<Vec<_>>(),
            DataFormat::PlainText => {
                let text = normalized.trim();
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![DatasetRecord::Text(crate::data::schema::TextRecord {
                        text: text.to_string(),
                        source: None,
                        meta: None,
                    })]
                }
            }
            DataFormat::JsonlText
            | DataFormat::JsonlChat
            | DataFormat::ParquetText
            | DataFormat::ParquetChat => {
                return Err(RustGptError::Data(format!(
                    "format {format} is not supported by Dataset::from_text"
                )));
            }
        };

        Self::from_records(format, records)
    }

    pub fn records(&self) -> &[DatasetRecord] {
        &self.records
    }

    pub fn docs(&self) -> &[String] {
        &self.rendered_docs
    }

    pub fn docs_with_template(&self, template: ChatTemplateKind) -> Vec<String> {
        self.records
            .iter()
            .map(|record| record.rendered_text_with_template(template))
            .collect()
    }

    pub fn format(&self) -> DataFormat {
        self.format
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn total_bytes_with_template(&self, template: ChatTemplateKind) -> usize {
        self.records
            .iter()
            .map(|record| record.rendered_text_with_template(template).len())
            .sum()
    }

    pub fn shuffled(&self, rng: &mut Rng) -> Self {
        let mut indices = (0..self.records.len()).collect::<Vec<_>>();
        rng.shuffle(&mut indices);
        let records = indices
            .iter()
            .map(|idx| self.records[*idx].clone())
            .collect::<Vec<_>>();
        Self::from_records(self.format, records).expect("shuffling preserves non-empty records")
    }

    fn from_jsonl_path(
        format: DataFormat,
        path: &std::path::Path,
        lowercase: bool,
    ) -> Result<Self> {
        let file = File::open(path).map_err(|source| RustGptError::io_with_path(path, source))?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.map_err(|source| RustGptError::io_with_path(path, source))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed).map_err(|source| {
                RustGptError::Data(format!(
                    "failed to parse JSONL line {} in {}: {source}",
                    line_idx + 1,
                    path.display()
                ))
            })?;
            let record = parse_record_value(format, value)?;
            records.push(if lowercase {
                record.lowercase()
            } else {
                record
            });
        }

        Self::from_records(format, records)
    }

    fn from_parquet_path(
        format: DataFormat,
        path: &std::path::Path,
        lowercase: bool,
    ) -> Result<Self> {
        let file = File::open(path).map_err(|source| RustGptError::io_with_path(path, source))?;
        let reader = SerializedFileReader::new(file).map_err(|source| {
            RustGptError::Data(format!(
                "failed to open parquet {}: {source}",
                path.display()
            ))
        })?;
        let row_iter = reader.get_row_iter(None).map_err(|source| {
            RustGptError::Data(format!(
                "failed to iterate parquet {}: {source}",
                path.display()
            ))
        })?;

        let mut records = Vec::new();
        for row in row_iter {
            let row = row.map_err(|source| {
                RustGptError::Data(format!(
                    "failed reading parquet row in {}: {source}",
                    path.display()
                ))
            })?;
            let value = row.to_json_value();
            let record = parse_record_value(format, value)?;
            records.push(if lowercase {
                record.lowercase()
            } else {
                record
            });
        }

        Self::from_records(format, records)
    }

    fn from_records(format: DataFormat, records: Vec<DatasetRecord>) -> Result<Self> {
        if records.is_empty() {
            return Err(RustGptError::Data(
                "dataset is empty after preprocessing".to_string(),
            ));
        }

        let rendered_docs = records
            .iter()
            .map(DatasetRecord::rendered_text)
            .collect::<Vec<_>>();
        let total_bytes = rendered_docs.iter().map(|doc| doc.len()).sum();
        if total_bytes == 0 {
            return Err(RustGptError::Data(
                "dataset does not contain any tokenizable text".to_string(),
            ));
        }

        Ok(Self {
            format,
            records,
            rendered_docs,
            total_bytes,
        })
    }
}

fn parse_record_value(format: DataFormat, value: Value) -> Result<DatasetRecord> {
    match format {
        DataFormat::Lines | DataFormat::PlainText => Err(RustGptError::Data(format!(
            "format {format} expects raw text input, not structured records"
        ))),
        DataFormat::JsonlText | DataFormat::ParquetText => {
            Ok(DatasetRecord::Text(parse_text_value(value)?))
        }
        DataFormat::JsonlChat | DataFormat::ParquetChat => {
            Ok(DatasetRecord::Chat(parse_chat_value(value)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::DataFormat;
    use crate::data::schema::DatasetRecord;

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

    #[test]
    fn chat_records_render_transcript_docs() {
        let dataset = Dataset::from_records(
            DataFormat::JsonlChat,
            vec![DatasetRecord::Chat(crate::data::schema::ChatRecord {
                messages: vec![
                    crate::data::schema::Message {
                        role: crate::data::schema::MessageRole::User,
                        content: "hello".to_string(),
                    },
                    crate::data::schema::Message {
                        role: crate::data::schema::MessageRole::Assistant,
                        content: "hi".to_string(),
                    },
                ],
                source: None,
                meta: None,
            })],
        )
        .unwrap();
        assert_eq!(dataset.docs()[0], "User: hello\nAssistant: hi\n");
    }
}
