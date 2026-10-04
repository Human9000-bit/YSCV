use thiserror::Error;

/// Errors returned by tensor construction and math operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorError {
    #[error("tensor size overflow for shape {shape:?}")]
    SizeOverflow { shape: Vec<usize> },
    #[error(
        "tensor shape {shape:?} expects {} elements, got {data_len}",
        shape_element_count(shape).unwrap_or(0)
    )]
    SizeMismatch { shape: Vec<usize>, data_len: usize },
    #[error("shape mismatch: left={left:?}, right={right:?}")]
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    #[error("broadcast mismatch: left={left:?}, right={right:?}")]
    BroadcastIncompatible { left: Vec<usize>, right: Vec<usize> },
    #[error("cannot reshape from {from:?} to {to:?} due to size mismatch")]
    ReshapeSizeMismatch { from: Vec<usize>, to: Vec<usize> },
    #[error("axis {axis} is out of range for rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },
    #[error("index rank mismatch: expected {expected}, got {got}")]
    InvalidIndexRank { expected: usize, got: usize },
    #[error("index out of bounds at axis {axis}: index={index}, dim={dim}")]
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        dim: usize,
    },
    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },
    #[error("{op} requires a rank-{expected} tensor, got rank {got}")]
    InvalidRank {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("matrix is singular")]
    SingularMatrix,
    #[error("matrix is not positive definite")]
    NotPositiveDefinite,
    #[error("unsupported einsum equation: {equation}")]
    UnsupportedEinsum { equation: String },
    #[error("einsum equation has {inputs} inputs but {tensors} tensors provided")]
    EinsumOperandCount { inputs: usize, tensors: usize },
    #[error("{op}: {arg} must be > 0")]
    ZeroArgument { op: &'static str, arg: &'static str },
    #[error("randint requires high > low, got low={low}, high={high}")]
    EmptyRandintRange { low: i64, high: i64 },
}

/// Element data type for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 half-precision float (stored as u16 bit pattern).
    F16,
    /// 16-bit Brain float (stored as u16 bit pattern).
    BF16,
}

fn shape_element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
}
