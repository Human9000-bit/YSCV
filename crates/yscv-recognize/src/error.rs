use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use yscv_tensor::TensorError;

#[derive(Debug, Clone, Error)]
pub enum RecognizeError {
    #[error("invalid embedding shape: expected rank-1, got {got:?}")]
    InvalidEmbeddingShape { got: Vec<usize> },
    #[error("embedding must contain at least one element")]
    EmptyEmbedding,
    #[error("embedding contains non-finite value at index {index}")]
    NonFiniteEmbeddingValue { index: usize },
    #[error("invalid recognition threshold: {value}; expected finite threshold in [-1, 1]")]
    InvalidThreshold { value: f32 },
    #[error("identity already enrolled: {id}")]
    DuplicateIdentity { id: String },
    #[error("embedding dim mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch { expected: usize, got: usize },
    #[error("recognizer serialization error: {source}")]
    Serialization {
        #[source]
        source: Arc<serde_json::Error>,
    },
    #[error("recognizer IO error for {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

impl From<serde_json::Error> for RecognizeError {
    fn from(source: serde_json::Error) -> Self {
        Self::Serialization {
            source: Arc::new(source),
        }
    }
}
