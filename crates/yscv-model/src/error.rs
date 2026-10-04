use std::num::ParseFloatError;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::str::Utf8Error;
use std::sync::Arc;

use thiserror::Error;
use yscv_autograd::AutogradError;
use yscv_imgproc::ImgProcError;
use yscv_kernels::KernelError;
use yscv_onnx::OnnxError;
use yscv_optim::OptimError;
use yscv_tensor::TensorError;

/// Errors returned by model-layer assembly, checkpoints, and training helpers.
#[derive(Debug, Clone, Error)]
pub enum ModelError {
    #[error("invalid shape for {parameter}: expected {expected:?}, got {got:?}")]
    InvalidParameterShape {
        parameter: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error(
        "invalid linear input shape: expected rank-2 with last dim {expected_features}, got {got:?}"
    )]
    InvalidInputShape {
        expected_features: usize,
        got: Vec<usize>,
    },
    #[error("invalid leaky-relu negative slope: {negative_slope}; expected finite value >= 0")]
    InvalidLeakyReluSlope { negative_slope: f32 },
    #[error("invalid dropout rate: {rate}; expected finite value in [0, 1)")]
    InvalidDropoutRate { rate: f32 },
    #[error("prediction/target shape mismatch: prediction={prediction:?}, target={target:?}")]
    PredictionTargetShapeMismatch {
        prediction: Vec<usize>,
        target: Vec<usize>,
    },
    #[error("cannot compute mean loss for empty tensor")]
    EmptyLossTensor,
    #[error("invalid huber delta: {delta}; expected finite value > 0")]
    InvalidHuberDelta { delta: f32 },
    #[error("invalid hinge margin: {margin}; expected finite value > 0")]
    InvalidHingeMargin { margin: f32 },
    #[error(
        "dataset tensors must have rank >= 1, got inputs_rank={inputs_rank}, targets_rank={targets_rank}"
    )]
    InvalidDatasetRank {
        inputs_rank: usize,
        targets_rank: usize,
    },
    #[error("dataset sample mismatch: inputs={inputs:?}, targets={targets:?}")]
    DatasetShapeMismatch {
        inputs: Vec<usize>,
        targets: Vec<usize>,
    },
    #[error("dataset is empty")]
    EmptyDataset,
    #[error("invalid batch size: {batch_size}; expected batch_size > 0")]
    InvalidBatchSize { batch_size: usize },
    #[error("invalid epoch count: {epochs}; expected epochs > 0")]
    InvalidEpochCount { epochs: usize },
    #[error(
        "invalid split ratios: train_ratio={train_ratio}, validation_ratio={validation_ratio}; expected finite values in [0, 1] with train+validation <= 1"
    )]
    InvalidSplitRatios {
        train_ratio: f32,
        validation_ratio: f32,
    },
    #[error(
        "invalid split counts: train_count={train_count}, validation_count={validation_count}, dataset_len={dataset_len}"
    )]
    InvalidSplitCounts {
        train_count: usize,
        validation_count: usize,
        dataset_len: usize,
    },
    #[error("invalid sampling weights length: expected {expected} weights, got {got}")]
    InvalidSamplingWeightsLength { expected: usize, got: usize },
    #[error("invalid sampling weight at index {index}: {value}; expected finite value >= 0")]
    InvalidSamplingWeight { index: usize, value: f32 },
    #[error("invalid sampling distribution: at least one weight must be > 0")]
    InvalidSamplingDistribution,
    #[error(
        "invalid class-balanced sampling target shape: expected scalar class labels ([N,1]) or one-hot labels ([N,C]), got {got:?}"
    )]
    InvalidClassSamplingTargetShape { got: Vec<usize> },
    #[error("invalid class-balanced sampling target at sample {index}: {value}; {reason}")]
    InvalidClassSamplingTargetValue {
        index: usize,
        value: f32,
        reason: &'static str,
    },
    #[error(
        "invalid augmentation probability for {operation}: {value}; expected finite value in [0, 1]"
    )]
    InvalidAugmentationProbability { operation: &'static str, value: f32 },
    #[error("invalid {op} argument {param}={value}; expected {expected}")]
    InvalidArgument {
        op: &'static str,
        param: &'static str,
        value: f32,
        expected: &'static str,
    },
    #[error("invalid {op} {param} range: min {min} > max {max}")]
    InvalidRange {
        op: &'static str,
        param: &'static str,
        min: f32,
        max: f32,
    },
    #[error("invalid channel_normalize {param}[{channel}]={value}; expected {expected}")]
    InvalidChannelStat {
        param: &'static str,
        channel: usize,
        value: f32,
        expected: &'static str,
    },
    #[error("{what} length mismatch: expected {expected}, got {got}")]
    LengthMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{op} expects a rank-{expected} input, got shape {shape:?}")]
    InvalidRank {
        op: &'static str,
        expected: usize,
        shape: Vec<usize>,
    },
    #[error(
        "augmentation produced inconsistent sample shapes: first={first:?}, sample[{index}]={got:?}"
    )]
    InconsistentSampleShapes {
        first: Vec<usize>,
        index: usize,
        got: Vec<usize>,
    },
    #[error("crop size ({crop_height}x{crop_width}) exceeds input size ({height}x{width})")]
    CropExceedsInput {
        crop_height: usize,
        crop_width: usize,
        height: usize,
        width: usize,
    },
    #[error("invalid augmentation input shape: expected rank-4 NHWC, got {got:?}")]
    InvalidAugmentationInputShape { got: Vec<usize> },
    #[error("invalid cutmix input shape: expected rank-4 NHWC, got {got:?}")]
    InvalidCutMixInputShape { got: Vec<usize> },
    #[error("invalid dataset-adapter shape for {field}: {shape:?}; every dimension must be > 0")]
    InvalidDatasetAdapterShape {
        field: &'static str,
        shape: Vec<usize>,
    },
    #[error("dataset-adapter {field} overflows usize: {shape:?}")]
    DatasetSizeOverflow {
        field: &'static str,
        shape: Vec<usize>,
    },
    #[error("failed to infer class name from directory path {path:?}")]
    ClassNameFromPath { path: PathBuf },
    #[error("invalid one-hot class configuration: class_id={class_id}, class_count={class_count}")]
    InvalidClassId { class_id: usize, class_count: usize },
    #[error("invalid image-folder extension configuration for {extension}: {reason}")]
    InvalidImageFolderExtension {
        extension: String,
        reason: &'static str,
    },
    #[error("invalid CSV delimiter: {delimiter:?}; expected a non-control character")]
    InvalidCsvDelimiter { delimiter: char },
    #[error("invalid CSV dataset column count at line {line}: expected {expected}, got {got}")]
    InvalidDatasetRecordColumns {
        line: usize,
        expected: usize,
        got: usize,
    },
    #[error("invalid dataset record path at line {line}: image path is empty")]
    InvalidDatasetRecordPath { line: usize },
    #[error("failed to parse CSV dataset value at line {line}, column {column}: {source}")]
    DatasetCsvParse {
        line: usize,
        column: usize,
        #[source]
        source: ParseFloatError,
    },
    #[error(
        "invalid JSONL dataset record length at line {line} for {field}: expected {expected}, got {got}"
    )]
    InvalidDatasetRecordLength {
        line: usize,
        field: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("invalid JSONL dataset record value at line {line} for {field}[{index}]: {reason}")]
    InvalidDatasetRecordValue {
        line: usize,
        field: &'static str,
        index: usize,
        reason: &'static str,
    },
    #[error("failed to parse JSONL dataset record at line {line}: {source}")]
    DatasetJsonlParse {
        line: usize,
        #[source]
        source: Arc<serde_json::Error>,
    },
    #[error("failed to decode dataset image {path:?}: {source}")]
    DatasetImageDecode {
        path: PathBuf,
        #[source]
        source: Arc<image::ImageError>,
    },
    #[error("I/O error on {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[error("JSON error in {what}: {source}")]
    Json {
        what: &'static str,
        #[source]
        source: Arc<serde_json::Error>,
    },
    #[error("invalid conv2d stride: stride_h={stride_h}, stride_w={stride_w}; both must be > 0")]
    InvalidConv2dStride { stride_h: usize, stride_w: usize },
    #[error("invalid batch-norm epsilon: {epsilon}; expected finite value > 0")]
    InvalidBatchNormEpsilon { epsilon: f32 },
    #[error("invalid pool kernel: kernel_h={kernel_h}, kernel_w={kernel_w}; both must be > 0")]
    InvalidPoolKernel { kernel_h: usize, kernel_w: usize },
    #[error("invalid pool stride: stride_h={stride_h}, stride_w={stride_w}; both must be > 0")]
    InvalidPoolStride { stride_h: usize, stride_w: usize },
    #[error("invalid flatten input shape: expected rank >= 2, got {got:?}")]
    InvalidFlattenShape { got: Vec<usize> },
    #[error("layer is inference-only and cannot be used in autograd graph forward pass")]
    InferenceOnlyLayer,
    #[error("layer {layer} parameters not registered in graph; call register_params first")]
    ParamsNotRegistered { layer: &'static str },
    #[error("layer is graph-only and cannot be used in direct tensor inference forward pass")]
    GraphOnlyLayer,
    #[error("invalid accumulation steps: {steps}; expected steps > 0")]
    InvalidAccumulationSteps { steps: usize },
    #[error("invalid layer index {index}: model has {count} layers")]
    InvalidLayerIndex { index: usize, count: usize },
    #[error("missing weight tensor: {name}")]
    WeightNotFound { name: String },
    #[error("{what} truncated: need {needed} bytes, have {len}")]
    Truncated {
        what: &'static str,
        needed: usize,
        len: usize,
    },
    #[error("weight file header is not valid UTF-8: {source}")]
    HeaderNotUtf8 {
        #[from]
        source: Utf8Error,
    },
    #[error("safetensors header entry '{tensor}': missing or invalid {field}")]
    InvalidTensorHeader { tensor: String, field: &'static str },
    #[error("unsupported safetensors dtype: {dtype}")]
    UnsupportedDType { dtype: String },
    #[error("tensor '{name}' data [{start}, {end}) out of bounds (data len {len})")]
    TensorDataOutOfBounds {
        name: String,
        start: usize,
        end: usize,
        len: usize,
    },
    #[error("tensor '{name}' expected {expected} bytes, got {got}")]
    TensorByteLenMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("model '{name}' is not in the hub registry")]
    UnknownHubModel { name: String },
    #[error("download failed for {url}: curl exited with {status}: {stderr}")]
    DownloadFailed {
        url: String,
        status: ExitStatus,
        stderr: String,
    },
    #[error("file size mismatch for {path:?}: expected {expected} bytes, got {actual} bytes")]
    FileSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("failed to {op} {addr}: {source}")]
    TransportAddress {
        op: &'static str,
        addr: String,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[error("transport {op} failed: {source}")]
    TransportIo {
        op: &'static str,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[error(
        "invalid worker rank {rank}: workers use ranks 1..world_size, rank 0 is the coordinator"
    )]
    InvalidWorkerRank { rank: usize },
    #[error("peer index {peer} out of range (have {peers} peers)")]
    PeerOutOfRange { peer: usize, peers: usize },
    #[error("in-process transport {op} with rank {peer} failed: channel closed")]
    ChannelClosed { op: &'static str, peer: usize },
    #[error("{lock} lock poisoned")]
    LockPoisoned { lock: &'static str },
    #[error("{thread} thread panicked")]
    ThreadPanicked { thread: &'static str },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    ImgProc(#[from] ImgProcError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error(transparent)]
    Autograd(#[from] AutogradError),
    #[error(transparent)]
    Optim(#[from] OptimError),
    #[error(transparent)]
    Onnx(#[from] OnnxError),
}
