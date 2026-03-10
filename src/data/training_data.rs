//! Pre-encoded training examples.
//! Line datasets are stored as one sequence per document; plain-text datasets become
//! one long token stream that is sampled with sliding windows.

use crate::core::config::DataFormat;
use crate::core::error::Result;
use crate::data::corpus::Dataset;
use crate::data::tokenizer::Tokenizer;

#[derive(Clone, Debug, PartialEq)]
pub enum TrainingData {
    Documents { encoded_docs: Vec<Vec<usize>> },
    TokenStream { tokens: Vec<usize> },
}

impl TrainingData {
    pub fn from_dataset(dataset: &Dataset, tokenizer: &Tokenizer) -> Result<Self> {
        match dataset.format() {
            DataFormat::Lines => Ok(Self::Documents {
                encoded_docs: dataset
                    .docs()
                    .iter()
                    .map(|doc| tokenizer.encode_with_boundaries(doc))
                    .collect::<Result<Vec<_>>>()?,
            }),
            DataFormat::PlainText => Ok(Self::TokenStream {
                tokens: tokenizer.encode_with_boundaries(&dataset.docs()[0])?,
            }),
        }
    }

    pub fn build_batch(
        &self,
        start_example_idx: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Vec<Vec<usize>> {
        match self {
            Self::Documents { encoded_docs } => (0..batch_size)
                .map(|batch_idx| {
                    let example_idx = start_example_idx + batch_idx;
                    encoded_docs[example_idx % encoded_docs.len()].clone()
                })
                .collect(),
            Self::TokenStream { tokens } => {
                let window_len = usize::min(tokens.len(), block_size + 1);
                if tokens.len() <= window_len {
                    return (0..batch_size).map(|_| tokens.clone()).collect();
                }

                let window_count = tokens.len() - window_len + 1;
                (0..batch_size)
                    .map(|batch_idx| {
                        let start = (start_example_idx + batch_idx) % window_count;
                        tokens[start..start + window_len].to_vec()
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::{BoundaryMode, DataFormat};
    use crate::data::corpus::Dataset;
    use crate::data::tokenizer::Tokenizer;

    use super::TrainingData;

    #[test]
    fn line_datasets_cycle_encoded_documents() {
        let dataset = Dataset::from_text("emma\nolivia\n", DataFormat::Lines, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training = TrainingData::from_dataset(&dataset, &tokenizer).unwrap();
        let batch = training.build_batch(1, 3, 8);
        assert_eq!(batch.len(), 3);
        assert_eq!(
            batch[0],
            tokenizer.encode_with_boundaries("olivia").unwrap()
        );
        assert_eq!(batch[1], tokenizer.encode_with_boundaries("emma").unwrap());
    }

    #[test]
    fn plain_text_datasets_build_sliding_windows() {
        let dataset = Dataset::from_text("abcdef", DataFormat::PlainText, false).unwrap();
        let tokenizer = Tokenizer::from_docs(dataset.docs(), BoundaryMode::SharedBos).unwrap();
        let training = TrainingData::from_dataset(&dataset, &tokenizer).unwrap();
        let batch = training.build_batch(1, 2, 3);

        let encoded = tokenizer.encode_with_boundaries("abcdef").unwrap();
        assert_eq!(batch[0], encoded[1..5].to_vec());
        assert_eq!(batch[1], encoded[2..6].to_vec());
    }
}
