//! Dataset export helpers.
//! This keeps parser-specific IO inside the data layer so the trainer can stay focused on
//! normalized text/chat records.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder, StructBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::ArrowWriter;
use serde_json::json;

use crate::core::config::{ChatTemplateKind, DataFormat};
use crate::core::error::{Result, RustGptError};
use crate::data::corpus::Dataset;
use crate::data::schema::{ChatRecord, DatasetRecord, TextRecord};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDataSummary {
    pub output_path: PathBuf,
    pub output_format: DataFormat,
    pub records: usize,
}

pub fn write_dataset(
    dataset: &Dataset,
    template: ChatTemplateKind,
    output_path: impl AsRef<Path>,
    output_format: DataFormat,
    pretty: bool,
) -> Result<PreparedDataSummary> {
    let output_path = output_path.as_ref();
    match output_format {
        DataFormat::Lines => write_rendered_lines(dataset, template, output_path)?,
        DataFormat::PlainText => write_rendered_text(dataset, template, output_path)?,
        DataFormat::JsonlText => write_jsonl_text(dataset, template, output_path, pretty)?,
        DataFormat::JsonlChat => write_jsonl_chat(dataset, output_path, pretty)?,
        DataFormat::ParquetText => write_parquet_text(dataset, template, output_path)?,
        DataFormat::ParquetChat => write_parquet_chat(dataset, output_path)?,
    }

    Ok(PreparedDataSummary {
        output_path: output_path.to_path_buf(),
        output_format,
        records: dataset.len(),
    })
}

fn write_rendered_lines(
    dataset: &Dataset,
    template: ChatTemplateKind,
    output_path: &Path,
) -> Result<()> {
    let file = File::create(output_path)
        .map_err(|source| RustGptError::io_with_path(output_path, source))?;
    let mut writer = BufWriter::new(file);
    for doc in dataset.docs_with_template(template) {
        writer
            .write_all(doc.as_bytes())
            .map_err(|source| RustGptError::io_with_path(output_path, source))?;
        writer
            .write_all(b"\n")
            .map_err(|source| RustGptError::io_with_path(output_path, source))?;
    }
    writer
        .flush()
        .map_err(|source| RustGptError::io_with_path(output_path, source))
}

fn write_rendered_text(
    dataset: &Dataset,
    template: ChatTemplateKind,
    output_path: &Path,
) -> Result<()> {
    let joined = dataset.docs_with_template(template).join("\n\n");
    std::fs::write(output_path, joined)
        .map_err(|source| RustGptError::io_with_path(output_path, source))
}

fn write_jsonl_text(
    dataset: &Dataset,
    template: ChatTemplateKind,
    output_path: &Path,
    pretty: bool,
) -> Result<()> {
    let file = File::create(output_path)
        .map_err(|source| RustGptError::io_with_path(output_path, source))?;
    let mut writer = BufWriter::new(file);
    for record in text_records(dataset, template) {
        write_json_line(&mut writer, &record, pretty, output_path)?;
    }
    writer
        .flush()
        .map_err(|source| RustGptError::io_with_path(output_path, source))
}

fn write_jsonl_chat(dataset: &Dataset, output_path: &Path, pretty: bool) -> Result<()> {
    let file = File::create(output_path)
        .map_err(|source| RustGptError::io_with_path(output_path, source))?;
    let mut writer = BufWriter::new(file);
    for record in chat_records(dataset)? {
        write_json_line(&mut writer, &record, pretty, output_path)?;
    }
    writer
        .flush()
        .map_err(|source| RustGptError::io_with_path(output_path, source))
}

fn write_json_line<T: serde::Serialize>(
    writer: &mut BufWriter<File>,
    value: &T,
    pretty: bool,
    output_path: &Path,
) -> Result<()> {
    let encoded = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|source| {
        RustGptError::Data(format!(
            "failed to serialize JSON for {}: {source}",
            output_path.display()
        ))
    })?;
    writer
        .write_all(encoded.as_bytes())
        .and_then(|_| writer.write_all(b"\n"))
        .map_err(|source| RustGptError::io_with_path(output_path, source))
}

fn write_parquet_text(
    dataset: &Dataset,
    template: ChatTemplateKind,
    output_path: &Path,
) -> Result<()> {
    let records = text_records(dataset, template);
    let mut text_builder = StringBuilder::new();
    let mut source_builder = StringBuilder::new();
    let mut meta_builder = StringBuilder::new();

    for record in &records {
        text_builder.append_value(&record.text);
        source_builder.append_option(record.source.as_deref());
        let meta_json = record.meta.as_ref().map(|meta| meta.to_string());
        meta_builder.append_option(meta_json.as_deref());
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, true),
        Field::new("meta_json", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(text_builder.finish()) as ArrayRef,
            Arc::new(source_builder.finish()) as ArrayRef,
            Arc::new(meta_builder.finish()) as ArrayRef,
        ],
    )
    .map_err(|source| {
        RustGptError::Data(format!("failed to build text parquet batch: {source}"))
    })?;
    write_parquet_batch(output_path, schema, &batch)
}

fn write_parquet_chat(dataset: &Dataset, output_path: &Path) -> Result<()> {
    let records = chat_records(dataset)?;
    let message_fields = Fields::from(vec![
        Field::new("role", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
    ]);
    let struct_builder = StructBuilder::new(
        message_fields.clone(),
        vec![
            Box::new(StringBuilder::new()),
            Box::new(StringBuilder::new()),
        ],
    );
    let mut messages_builder = ListBuilder::new(struct_builder);
    let mut source_builder = StringBuilder::new();
    let mut meta_builder = StringBuilder::new();

    for record in &records {
        {
            let values = messages_builder.values();
            for message in &record.messages {
                values
                    .field_builder::<StringBuilder>(0)
                    .expect("role builder")
                    .append_value(message.role.as_str());
                values
                    .field_builder::<StringBuilder>(1)
                    .expect("content builder")
                    .append_value(&message.content);
                values.append(true);
            }
        }
        messages_builder.append(true);
        source_builder.append_option(record.source.as_deref());
        let meta_json = record.meta.as_ref().map(|meta| meta.to_string());
        meta_builder.append_option(meta_json.as_deref());
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "messages",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(message_fields),
                true,
            ))),
            false,
        ),
        Field::new("source", DataType::Utf8, true),
        Field::new("meta_json", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(messages_builder.finish()) as ArrayRef,
            Arc::new(source_builder.finish()) as ArrayRef,
            Arc::new(meta_builder.finish()) as ArrayRef,
        ],
    )
    .map_err(|source| {
        RustGptError::Data(format!("failed to build chat parquet batch: {source}"))
    })?;
    write_parquet_batch(output_path, schema, &batch)
}

fn write_parquet_batch(output_path: &Path, schema: Arc<Schema>, batch: &RecordBatch) -> Result<()> {
    let file = File::create(output_path)
        .map_err(|source| RustGptError::io_with_path(output_path, source))?;
    let mut writer = ArrowWriter::try_new(file, schema, None).map_err(|source| {
        RustGptError::Data(format!(
            "failed to open parquet writer {}: {source}",
            output_path.display()
        ))
    })?;
    writer.write(batch).map_err(|source| {
        RustGptError::Data(format!(
            "failed writing parquet {}: {source}",
            output_path.display()
        ))
    })?;
    writer.close().map_err(|source| {
        RustGptError::Data(format!(
            "failed closing parquet {}: {source}",
            output_path.display()
        ))
    })?;
    Ok(())
}

fn text_records(dataset: &Dataset, template: ChatTemplateKind) -> Vec<TextRecord> {
    dataset
        .records()
        .iter()
        .map(|record| match record {
            DatasetRecord::Text(record) => record.clone(),
            DatasetRecord::Chat(record) => TextRecord {
                text: crate::data::schema::render_messages(&record.messages, template),
                source: record.source.clone(),
                meta: Some(json!({
                    "prepared_from": "chat",
                    "meta": record.meta.clone()
                })),
            },
        })
        .collect()
}

fn chat_records(dataset: &Dataset) -> Result<Vec<ChatRecord>> {
    dataset
        .records()
        .iter()
        .map(|record| match record {
            DatasetRecord::Chat(record) => Ok(record.clone()),
            DatasetRecord::Text(_) => Err(RustGptError::Data(
                "chat export requires structured chat input; convert to jsonl-chat or parquet-chat from a chat dataset".to_string(),
            )),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::core::config::{ChatTemplateKind, DataFormat};
    use crate::data::corpus::Dataset;
    use crate::data::schema::DatasetRecord;

    use super::write_dataset;

    #[test]
    fn writes_jsonl_text_from_plain_text() {
        let dataset = Dataset::from_text("hello\nworld", DataFormat::PlainText, false).unwrap();
        let path = temp_path("prepare_jsonl_text.jsonl");
        write_dataset(
            &dataset,
            ChatTemplateKind::SimpleTranscript,
            &path,
            DataFormat::JsonlText,
            false,
        )
        .unwrap();

        let written = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(written.contains("\"text\":\"hello\\nworld\""));
    }

    #[test]
    fn parquet_text_roundtrips_through_dataset_loader() {
        let dataset = Dataset::from_text("hello", DataFormat::PlainText, false).unwrap();
        let path = temp_path("prepare_parquet_text.parquet");
        write_dataset(
            &dataset,
            ChatTemplateKind::SimpleTranscript,
            &path,
            DataFormat::ParquetText,
            false,
        )
        .unwrap();

        let loaded = Dataset::from_path(&crate::core::config::DataConfig {
            data_path: path.clone(),
            format: DataFormat::ParquetText,
            ..Default::default()
        })
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.docs(), &["hello".to_string()]);
    }

    #[test]
    fn parquet_chat_roundtrips_through_dataset_loader() {
        let input_path = temp_chat_jsonl_path();
        let dataset = Dataset::from_path(&crate::core::config::DataConfig {
            data_path: input_path.clone(),
            format: DataFormat::JsonlChat,
            ..Default::default()
        })
        .unwrap();
        let path = temp_path("prepare_parquet_chat.parquet");
        write_dataset(
            &dataset,
            ChatTemplateKind::ChatMl,
            &path,
            DataFormat::ParquetChat,
            false,
        )
        .unwrap();

        let loaded = Dataset::from_path(&crate::core::config::DataConfig {
            data_path: path.clone(),
            format: DataFormat::ParquetChat,
            ..Default::default()
        })
        .unwrap();
        std::fs::remove_file(&input_path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(matches!(
            &loaded.records()[0],
            DatasetRecord::Chat(record) if record.messages.len() == 2
        ));
    }

    fn temp_chat_jsonl_path() -> std::path::PathBuf {
        let path = temp_path("prepare_chat_input.jsonl");
        std::fs::write(
            &path,
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"},{\"role\":\"assistant\",\"content\":\"hi\"}]}\n",
        )
        .unwrap();
        path
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rustgpt_{label}_{nanos}"))
    }
}
