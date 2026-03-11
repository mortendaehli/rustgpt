//! Pre-encoded training examples.
//! Text pretraining examples use full next-token loss; chat/SFT examples can mask
//! non-assistant targets while preserving the same model core.

use crate::core::config::{ChatTemplateKind, DataFormat, TrainMode};
use crate::core::error::{Result, RustGptError};
use crate::data::corpus::Dataset;
use crate::data::schema::{ChatRecord, DatasetRecord, MessageRole};
use crate::data::tokenizer::Tokenizer;

#[derive(Clone, Debug, PartialEq)]
pub struct SequenceExample {
    pub input_ids: Vec<usize>,
    pub target_ids: Vec<usize>,
    pub loss_mask: Vec<f32>,
}

impl SequenceExample {
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn tokens_with_boundaries(&self) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(self.input_ids.len() + 1);
        tokens.extend_from_slice(&self.input_ids);
        if let Some(last) = self.target_ids.last().copied() {
            tokens.push(last);
        }
        tokens
    }

    pub fn has_uniform_loss_mask(&self) -> bool {
        self.loss_mask
            .iter()
            .all(|weight| (*weight - 1.0).abs() <= f32::EPSILON)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TrainingData {
    Examples { examples: Vec<SequenceExample> },
    TokenStream { tokens: Vec<usize> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingWindowSummary {
    pub source_sequences: usize,
    pub total_windows: usize,
    pub multi_window_sequences: usize,
    pub max_windows_per_sequence: usize,
}

impl TrainingData {
    pub fn from_dataset(
        dataset: &Dataset,
        tokenizer: &Tokenizer,
        mode: TrainMode,
        chat_template: ChatTemplateKind,
    ) -> Result<Self> {
        match dataset.format() {
            DataFormat::PlainText => Ok(Self::TokenStream {
                tokens: tokenizer.encode_with_boundaries(&dataset.docs()[0])?,
            }),
            DataFormat::Lines | DataFormat::JsonlText | DataFormat::ParquetText => {
                let mut examples = Vec::with_capacity(dataset.records().len());
                for record in dataset.records() {
                    let DatasetRecord::Text(record) = record else {
                        return Err(RustGptError::Data(
                            "text dataset contained a non-text record".to_string(),
                        ));
                    };
                    examples.push(build_text_example(tokenizer, &record.text)?);
                }
                Ok(Self::Examples { examples })
            }
            DataFormat::JsonlChat | DataFormat::ParquetChat => {
                let mut examples = Vec::with_capacity(dataset.records().len());
                for record in dataset.records() {
                    let DatasetRecord::Chat(record) = record else {
                        return Err(RustGptError::Data(
                            "chat dataset contained a non-chat record".to_string(),
                        ));
                    };
                    examples.push(build_chat_example(
                        tokenizer,
                        record,
                        matches!(mode, TrainMode::Auto | TrainMode::Sft),
                        chat_template,
                    )?);
                }
                Ok(Self::Examples { examples })
            }
        }
    }

    pub fn build_batch(
        &self,
        start_example_idx: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Vec<SequenceExample> {
        match self {
            Self::Examples { examples } => {
                let total_windows = total_example_windows(examples, block_size);
                (0..batch_size)
                    .map(|batch_idx| {
                        let window_idx = (start_example_idx + batch_idx) % total_windows;
                        select_example_window(examples, window_idx, block_size)
                    })
                    .collect()
            }
            Self::TokenStream { tokens } => {
                let window_len = usize::min(tokens.len(), block_size + 1);
                if tokens.len() <= window_len {
                    let example = build_window_example(tokens);
                    return (0..batch_size).map(|_| example.clone()).collect();
                }

                let window_count = tokens.len() - window_len + 1;
                (0..batch_size)
                    .map(|batch_idx| {
                        let start = (start_example_idx + batch_idx) % window_count;
                        build_window_example(&tokens[start..start + window_len])
                    })
                    .collect()
            }
        }
    }

    pub fn example_count(&self, block_size: usize) -> usize {
        match self {
            Self::Examples { examples } => total_example_windows(examples, block_size),
            Self::TokenStream { tokens } => {
                let window_len = usize::min(tokens.len(), block_size + 1);
                if tokens.len() <= window_len {
                    1
                } else {
                    tokens.len() - window_len + 1
                }
            }
        }
    }

    pub fn window_summary(&self, block_size: usize) -> TrainingWindowSummary {
        match self {
            Self::Examples { examples } => {
                let window_counts = examples
                    .iter()
                    .map(|example| window_starts(example, block_size).len())
                    .collect::<Vec<_>>();
                TrainingWindowSummary {
                    source_sequences: examples.len(),
                    total_windows: window_counts.iter().sum(),
                    multi_window_sequences: window_counts
                        .iter()
                        .filter(|&&count| count > 1)
                        .count(),
                    max_windows_per_sequence: window_counts.into_iter().max().unwrap_or(0),
                }
            }
            Self::TokenStream { .. } => {
                let total_windows = self.example_count(block_size);
                TrainingWindowSummary {
                    source_sequences: 1,
                    total_windows,
                    multi_window_sequences: usize::from(total_windows > 1),
                    max_windows_per_sequence: total_windows,
                }
            }
        }
    }

    pub fn split_validation(self, validation_ratio: f32) -> Result<(Self, Self)> {
        if !(0.0..1.0).contains(&validation_ratio) {
            return Err(RustGptError::Config(format!(
                "validation ratio must be in [0, 1), got {validation_ratio}"
            )));
        }
        match self {
            Self::Examples { mut examples } => {
                if examples.len() < 2 {
                    return Err(RustGptError::Data(
                        "validation split requires at least two examples".to_string(),
                    ));
                }
                let validation_count = usize::max(
                    1,
                    (examples.len() as f32 * validation_ratio).round() as usize,
                )
                .min(examples.len() - 1);
                let split_at = examples.len() - validation_count;
                let validation_examples = examples.split_off(split_at);
                Ok((
                    Self::Examples { examples },
                    Self::Examples {
                        examples: validation_examples,
                    },
                ))
            }
            Self::TokenStream { tokens } => {
                if tokens.len() < 4 {
                    return Err(RustGptError::Data(
                        "validation split requires at least four boundary-aware tokens".to_string(),
                    ));
                }
                let validation_len =
                    usize::max(2, (tokens.len() as f32 * validation_ratio).round() as usize)
                        .min(tokens.len() - 2);
                let split_at = tokens.len() - validation_len;
                Ok((
                    Self::TokenStream {
                        tokens: tokens[..split_at].to_vec(),
                    },
                    Self::TokenStream {
                        tokens: tokens[split_at..].to_vec(),
                    },
                ))
            }
        }
    }
}

fn build_text_example(tokenizer: &Tokenizer, text: &str) -> Result<SequenceExample> {
    let tokens = tokenizer.encode_with_boundaries(text)?;
    Ok(build_window_example(&tokens))
}

fn build_chat_example(
    tokenizer: &Tokenizer,
    chat: &ChatRecord,
    assistant_only_loss: bool,
    template: ChatTemplateKind,
) -> Result<SequenceExample> {
    let mut tokens = Vec::new();
    let mut target_token_mask = Vec::new();
    let mut previous_role = None;

    tokens.push(tokenizer.bos_id());
    target_token_mask.push(0.0);

    for message in &chat.messages {
        let prefix_tokens = tokenizer.encode_text(message.role.prefix(template));
        let prefix_weight = if assistant_only_loss {
            usize::from(matches!(previous_role, Some(MessageRole::Assistant))) as f32
        } else {
            1.0
        };
        target_token_mask.extend(std::iter::repeat_n(prefix_weight, prefix_tokens.len()));
        tokens.extend(prefix_tokens);

        let content_tokens = tokenizer.encode_text(&message.content);
        let content_weight =
            if !assistant_only_loss || matches!(message.role, MessageRole::Assistant) {
                1.0
            } else {
                0.0
            };
        target_token_mask.extend(std::iter::repeat_n(content_weight, content_tokens.len()));
        tokens.extend(content_tokens);

        let newline_tokens = tokenizer.encode_text("\n");
        let newline_weight =
            if !assistant_only_loss || matches!(message.role, MessageRole::Assistant) {
                1.0
            } else {
                0.0
            };
        target_token_mask.extend(std::iter::repeat_n(newline_weight, newline_tokens.len()));
        tokens.extend(newline_tokens);
        previous_role = Some(message.role);
    }

    tokens.push(tokenizer.eos_id().unwrap_or(tokenizer.bos_id()));
    let eos_weight =
        if !assistant_only_loss || matches!(previous_role, Some(MessageRole::Assistant)) {
            1.0
        } else {
            0.0
        };
    target_token_mask.push(eos_weight);

    if tokens.len() < 2 {
        return Err(RustGptError::Data(
            "chat record is too short to produce training targets".to_string(),
        ));
    }

    let input_ids = tokens[..tokens.len() - 1].to_vec();
    let target_ids = tokens[1..].to_vec();
    let loss_mask = target_token_mask[1..].to_vec();
    if loss_mask.iter().all(|weight| *weight <= 0.0) {
        return Err(RustGptError::Data(
            "chat record does not contain any assistant tokens for supervised loss".to_string(),
        ));
    }

    Ok(SequenceExample {
        input_ids,
        target_ids,
        loss_mask,
    })
}

fn build_window_example(tokens: &[usize]) -> SequenceExample {
    SequenceExample {
        input_ids: tokens[..tokens.len() - 1].to_vec(),
        target_ids: tokens[1..].to_vec(),
        loss_mask: vec![1.0; tokens.len() - 1],
    }
}

#[cfg(test)]
fn truncate_example(example: &SequenceExample, block_size: usize) -> SequenceExample {
    let keep = usize::min(example.len(), block_size);
    if keep == example.len() {
        return example.clone();
    }
    let start = best_window_start(&example.loss_mask, keep);
    window_from_start(example, start, keep)
}

fn window_from_start(example: &SequenceExample, start: usize, keep: usize) -> SequenceExample {
    SequenceExample {
        input_ids: example.input_ids[start..start + keep].to_vec(),
        target_ids: example.target_ids[start..start + keep].to_vec(),
        loss_mask: example.loss_mask[start..start + keep].to_vec(),
    }
}

fn best_window_start(loss_mask: &[f32], window_len: usize) -> usize {
    if loss_mask.len() <= window_len {
        return 0;
    }

    let mut current_sum = loss_mask[..window_len].iter().sum::<f32>();
    let mut best_sum = current_sum;
    let mut best_start = 0;

    for start in 1..=loss_mask.len() - window_len {
        current_sum += loss_mask[start + window_len - 1] - loss_mask[start - 1];
        if current_sum > best_sum + f32::EPSILON {
            best_sum = current_sum;
            best_start = start;
        }
    }

    best_start
}

fn total_example_windows(examples: &[SequenceExample], block_size: usize) -> usize {
    examples
        .iter()
        .map(|example| window_starts(example, block_size).len())
        .sum()
}

fn select_example_window(
    examples: &[SequenceExample],
    mut window_idx: usize,
    block_size: usize,
) -> SequenceExample {
    for example in examples {
        let starts = window_starts(example, block_size);
        if window_idx < starts.len() {
            let keep = usize::min(example.len(), block_size);
            return window_from_start(example, starts[window_idx], keep);
        }
        window_idx -= starts.len();
    }
    unreachable!("window index must resolve against total example windows");
}

fn window_starts(example: &SequenceExample, block_size: usize) -> Vec<usize> {
    let keep = usize::min(example.len(), block_size);
    if keep >= example.len() {
        return vec![0];
    }
    if example.has_uniform_loss_mask() {
        return (0..=example.len() - keep).collect();
    }

    let mut starts = Vec::new();
    let mut current_sum = example.loss_mask[..keep].iter().sum::<f32>();
    if current_sum > 0.0 {
        starts.push(0);
    }
    for start in 1..=example.loss_mask.len() - keep {
        current_sum += example.loss_mask[start + keep - 1] - example.loss_mask[start - 1];
        if current_sum > 0.0 {
            starts.push(start);
        }
    }
    if starts.is_empty() {
        starts.push(best_window_start(&example.loss_mask, keep));
    }
    starts
}

#[cfg(test)]
mod tests {
    use crate::core::config::{BoundaryMode, ChatTemplateKind, DataFormat, TrainMode};
    use crate::data::corpus::Dataset;
    use crate::data::schema::{ChatRecord, Message, MessageRole};
    use crate::data::tokenizer::Tokenizer;

    use super::{TrainingData, build_chat_example};

    #[test]
    fn line_datasets_cycle_encoded_documents() {
        let dataset = Dataset::from_text("emma\nolivia\n", DataFormat::Lines, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training = TrainingData::from_dataset(
            &dataset,
            &tokenizer,
            TrainMode::Auto,
            ChatTemplateKind::SimpleTranscript,
        )
        .unwrap();
        let batch = training.build_batch(1, 2, 8);
        assert_eq!(
            batch[0].tokens_with_boundaries(),
            tokenizer.encode_with_boundaries("olivia").unwrap()
        );
        assert_eq!(
            batch[1].tokens_with_boundaries(),
            tokenizer.encode_with_boundaries("emma").unwrap()
        );
    }

    #[test]
    fn plain_text_datasets_build_sliding_windows() {
        let dataset = Dataset::from_text("abcdef", DataFormat::PlainText, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training = TrainingData::from_dataset(
            &dataset,
            &tokenizer,
            TrainMode::Auto,
            ChatTemplateKind::SimpleTranscript,
        )
        .unwrap();
        let batch = training.build_batch(1, 2, 3);

        let encoded = tokenizer.encode_with_boundaries("abcdef").unwrap();
        assert_eq!(batch[0].tokens_with_boundaries(), encoded[1..5].to_vec());
        assert_eq!(batch[1].tokens_with_boundaries(), encoded[2..6].to_vec());
    }

    #[test]
    fn chat_examples_mask_only_assistant_targets() {
        let dataset = Dataset::from_text("placeholder", DataFormat::PlainText, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let chat = ChatRecord {
            messages: vec![
                Message {
                    role: MessageRole::User,
                    content: "hello".to_string(),
                },
                Message {
                    role: MessageRole::Assistant,
                    content: "hi".to_string(),
                },
            ],
            source: None,
            meta: None,
        };
        let example =
            build_chat_example(&tokenizer, &chat, true, ChatTemplateKind::SimpleTranscript)
                .unwrap();
        assert!(example.loss_mask.iter().any(|weight| *weight > 0.0));
        assert!(example.loss_mask.iter().any(|weight| *weight == 0.0));
    }

    #[test]
    fn chat_examples_supervise_assistant_end_markers() {
        let dataset = Dataset::from_text("placeholder", DataFormat::PlainText, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let chat = ChatRecord {
            messages: vec![
                Message {
                    role: MessageRole::User,
                    content: "hello".to_string(),
                },
                Message {
                    role: MessageRole::Assistant,
                    content: "hi".to_string(),
                },
                Message {
                    role: MessageRole::User,
                    content: "again".to_string(),
                },
            ],
            source: None,
            meta: None,
        };
        let example =
            build_chat_example(&tokenizer, &chat, true, ChatTemplateKind::SimpleTranscript)
                .unwrap();
        let tokens = example.tokens_with_boundaries();

        let user_prefix = tokenizer.encode_text("User: ");
        let trailing_user_prefix_start = tokens
            .windows(user_prefix.len())
            .rposition(|window| window == user_prefix.as_slice())
            .unwrap();
        let newline = tokenizer.encode_text("\n")[0];
        assert_eq!(tokens[trailing_user_prefix_start - 1], newline);
        assert_eq!(example.loss_mask[trailing_user_prefix_start - 2], 1.0);
        assert_eq!(example.loss_mask[trailing_user_prefix_start - 1], 1.0);
    }

    #[test]
    fn truncation_keeps_window_with_supervised_tokens() {
        let example = super::SequenceExample {
            input_ids: vec![0, 1, 2, 3, 4, 5],
            target_ids: vec![1, 2, 3, 4, 5, 6],
            loss_mask: vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        };
        let truncated = super::truncate_example(&example, 3);
        assert_eq!(truncated.loss_mask, vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn long_examples_produce_multiple_windows() {
        let dataset = Dataset::from_text("abcdefghij", DataFormat::Lines, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training = TrainingData::from_dataset(
            &dataset,
            &tokenizer,
            TrainMode::Auto,
            ChatTemplateKind::SimpleTranscript,
        )
        .unwrap();

        assert_eq!(training.example_count(4), 8);
        assert_eq!(
            training.window_summary(4),
            super::TrainingWindowSummary {
                source_sequences: 1,
                total_windows: 8,
                multi_window_sequences: 1,
                max_windows_per_sequence: 8,
            }
        );
        let batch = training.build_batch(2, 2, 4);
        let encoded = tokenizer.encode_with_boundaries("abcdefghij").unwrap();
        assert_eq!(batch[0].tokens_with_boundaries(), encoded[2..7].to_vec());
        assert_eq!(batch[1].tokens_with_boundaries(), encoded[3..8].to_vec());
    }

    #[test]
    fn validation_split_preserves_token_stream_shape() {
        let training = TrainingData::TokenStream {
            tokens: vec![0, 1, 2, 3, 4, 5],
        };
        let (train, valid) = training.split_validation(0.34).unwrap();
        match (train, valid) {
            (
                TrainingData::TokenStream { tokens: train },
                TrainingData::TokenStream { tokens: valid },
            ) => {
                assert!(train.len() >= 2);
                assert!(valid.len() >= 2);
            }
            _ => panic!("expected token-stream split"),
        }
    }
}
