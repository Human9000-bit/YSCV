use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use yscv_kernels::KernelError;
use yscv_tensor::TensorError;

use crate::ir::IrError;

/// Errors returned by ONNX model loading and conversion.
#[derive(Debug, Clone, Error)]
pub enum OnnxError {
    #[error("failed to decode ONNX protobuf: {source}")]
    Decode {
        #[from]
        source: prost::DecodeError,
    },
    #[error("failed to encode ONNX protobuf: {source}")]
    Encode {
        #[from]
        source: prost::EncodeError,
    },
    #[error("I/O error on {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error(transparent)]
    Ir(#[from] IrError),
    #[error("node {node} ({op_type}): {source}")]
    AtNode {
        node: String,
        op_type: String,
        #[source]
        source: Box<OnnxError>,
    },
    #[error("ONNX model has no graph")]
    MissingGraph,
    #[error("unsupported ONNX data type: {data_type}")]
    UnsupportedDataType { data_type: i32 },
    #[error(
        "tensor shape mismatch in initializer '{name}': expected {expected} elements, got {got}"
    )]
    InitializerShapeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("unsupported ONNX op type: {op_type}")]
    UnsupportedOpType { op_type: String },
    #[error("missing input '{input}' for node '{node}'")]
    MissingInput { node: String, input: String },
    #[error("value '{name}' is defined more than once")]
    DuplicateDefinition { name: String },
    #[error("node '{node}' has no output")]
    MissingOutput { node: String },
    #[error("missing required attribute '{attr}' on node '{node}'")]
    MissingAttribute { node: String, attr: &'static str },
    #[error("invalid attribute '{attr}' on node '{node}'")]
    InvalidAttribute { node: String, attr: &'static str },
    #[error("{op} expects a rank-{expected} input, got shape {shape:?}")]
    InvalidRank {
        op: &'static str,
        expected: usize,
        shape: Vec<usize>,
    },
    #[error("{op}: shapes {left:?} and {right:?} are incompatible")]
    IncompatibleShapes {
        op: &'static str,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    #[error("{op}: {what} {dims:?} not divisible by {divisor}")]
    NotDivisible {
        op: &'static str,
        what: &'static str,
        dims: Vec<usize>,
        divisor: usize,
    },
    #[error("{op} axis {axis} out of range for a rank-{rank} input")]
    AxisOutOfRange {
        op: &'static str,
        axis: i64,
        rank: usize,
    },
    #[error("{op} index {index} out of bounds for axis {axis} (dim {dim})")]
    IndexOutOfRange {
        op: &'static str,
        index: i64,
        axis: usize,
        dim: usize,
    },
    #[error("node '{node}': scale length {len} does not match axis {axis} of shape {shape:?}")]
    ScaleLenMismatch {
        node: String,
        len: usize,
        axis: usize,
        shape: Vec<usize>,
    },
    #[error("cannot build {what} tensor for '{name}': {source}")]
    BuildTensor {
        name: String,
        what: &'static str,
        #[source]
        source: TensorError,
    },
    #[error("bias scale for '{name}' is not usable: {scale}")]
    UnusableBiasScale { name: String, scale: f32 },
    #[error("int4 group size must be even and > 0, got {group_size}")]
    InvalidGroupSize { group_size: usize },
    #[error("KV-cache layer {layer} out of range ({num_layers} layers)")]
    KvCacheLayerOutOfRange { layer: usize, num_layers: usize },
    #[error("KV-cache overflow: {seq_len} + {new_tokens} > {max_seq_len}")]
    KvCacheOverflow {
        seq_len: usize,
        new_tokens: usize,
        max_seq_len: usize,
    },
    #[error("{op} '{node}': {weight} weight not prepacked (loader gate broken)")]
    NotPrepacked {
        op: &'static str,
        node: String,
        weight: &'static str,
    },
    #[error("{op} '{node}': prepacked weight is {got:?}, weight needs {expected:?}")]
    PrepackedShapeMismatch {
        op: &'static str,
        node: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("{op} '{node}': unexpected {what} shape {shape:?}")]
    UnexpectedShape {
        op: &'static str,
        node: String,
        what: &'static str,
        shape: Vec<usize>,
    },
    #[error("{op} '{node}': expected {expected} channels, got {got}")]
    ChannelMismatch {
        op: &'static str,
        node: String,
        expected: usize,
        got: usize,
    },
    #[error("plan expects {expected} inputs, got {got}")]
    InputCountMismatch { expected: usize, got: usize },
    #[error("input '{input}' expects {expected} elements, got {got}")]
    InputLenMismatch {
        input: String,
        expected: usize,
        got: usize,
    },
    #[error("no Metal device available")]
    MetalUnavailable,
    #[error("MPSGraph: {reason}")]
    MpsGraph { reason: &'static str },
    #[error("failed to create thread pool: {source}")]
    ThreadPool {
        #[source]
        source: Arc<rayon::ThreadPoolBuildError>,
    },
    #[error("{lock} lock poisoned")]
    LockPoisoned { lock: &'static str },
}

impl From<rayon::ThreadPoolBuildError> for OnnxError {
    fn from(source: rayon::ThreadPoolBuildError) -> Self {
        Self::ThreadPool {
            source: Arc::new(source),
        }
    }
}
