use std::sync::Arc;

use thiserror::Error;
use yscv_tensor::{Layout, TensorError};

/// Errors returned by kernel backends.
#[derive(Debug, Clone, Error)]
pub enum KernelError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(
        "matmul_2d requires rank-2 tensors, got left_rank={left_rank}, right_rank={right_rank}"
    )]
    InvalidMatMulRank { left_rank: usize, right_rank: usize },
    #[error(
        "matmul_2d shape mismatch: left={left:?}, right={right:?}; expected left[1] == right[0]"
    )]
    MatMulShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    #[error("failed to build cpu thread pool: {source}")]
    ThreadPoolBuild {
        #[source]
        source: Arc<rayon::ThreadPoolBuildError>,
    },
    #[error("pool2d_nhwc requires rank-4 tensor [N,H,W,C], got rank={got_rank}")]
    InvalidPoolRank { got_rank: usize },
    #[error(
        "invalid pool2d parameters: kernel=({kernel_h},{kernel_w}), stride=({stride_h},{stride_w}); all values must be > 0"
    )]
    InvalidPoolParameters {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
    #[error(
        "pool2d kernel larger than input: input_hw=({input_h},{input_w}), kernel_hw=({kernel_h},{kernel_w})"
    )]
    PoolKernelLargerThanInput {
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
    },
    #[error(
        "conv2d_nhwc requires rank-4 input/kernel tensors [N,H,W,C] and [KH,KW,C,OC], got input_rank={input_rank}, kernel_rank={kernel_rank}"
    )]
    InvalidConvRank {
        input_rank: usize,
        kernel_rank: usize,
    },
    #[error(
        "invalid conv2d parameters: kernel=({kernel_h},{kernel_w}), stride=({stride_h},{stride_w}); all values must be > 0"
    )]
    InvalidConvParameters {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
    #[error(
        "conv2d input/kernel channel mismatch: input_channels={input_channels}, kernel_in_channels={kernel_in_channels}"
    )]
    ConvChannelMismatch {
        input_channels: usize,
        kernel_in_channels: usize,
    },
    #[error(
        "conv2d kernel larger than input: input_hw=({input_h},{input_w}), kernel_hw=({kernel_h},{kernel_w})"
    )]
    ConvKernelLargerThanInput {
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
    },
    #[error("conv2d bias shape mismatch: bias_shape={bias_shape:?}, expected [{out_channels}]")]
    ConvBiasShapeMismatch {
        bias_shape: Vec<usize>,
        out_channels: usize,
    },
    #[error(
        "depthwise_conv2d_nhwc requires rank-4 input/kernel tensors [N,H,W,C] and [KH,KW,C,depth_multiplier], got input_rank={input_rank}, kernel_rank={kernel_rank}"
    )]
    InvalidDepthwiseConvRank {
        input_rank: usize,
        kernel_rank: usize,
    },
    #[error(
        "invalid depthwise_conv2d parameters: kernel=({kernel_h},{kernel_w}), stride=({stride_h},{stride_w}); all values must be > 0"
    )]
    InvalidDepthwiseConvParameters {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    },
    #[error(
        "depthwise_conv2d input/kernel channel mismatch: input_channels={input_channels}, kernel_channels={kernel_channels}"
    )]
    DepthwiseConvChannelMismatch {
        input_channels: usize,
        kernel_channels: usize,
    },
    #[error(
        "depthwise_conv2d kernel larger than input: input_hw=({input_h},{input_w}), kernel_hw=({kernel_h},{kernel_w})"
    )]
    DepthwiseConvKernelLargerThanInput {
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
    },
    #[error(
        "depthwise_conv2d bias shape mismatch: bias_shape={bias_shape:?}, expected [{out_channels}]"
    )]
    DepthwiseConvBiasShapeMismatch {
        bias_shape: Vec<usize>,
        out_channels: usize,
    },
    #[error(
        "separable_conv2d_nhwc pointwise kernel must have shape [1,1,C_in,C_out], got pointwise_shape={pointwise_shape:?}"
    )]
    InvalidSeparablePointwiseKernelShape { pointwise_shape: Vec<usize> },
    #[error("batch_norm2d_nhwc requires rank-4 input tensor [N,H,W,C], got rank={got_rank}")]
    InvalidBatchNormRank { got_rank: usize },
    #[error(
        "batch_norm2d_nhwc parameter `{parameter}` shape mismatch: got {shape:?}, expected [{expected_channels}]"
    )]
    BatchNormParameterShapeMismatch {
        parameter: &'static str,
        shape: Vec<usize>,
        expected_channels: usize,
    },
    #[error("batch_norm2d_nhwc requires epsilon to be finite and > 0")]
    InvalidBatchNormEpsilon,
    #[error(
        "batch_norm2d_nhwc requires variance[channel] + epsilon > 0, invalid at channel={channel}"
    )]
    InvalidBatchNormVariance { channel: usize },
    #[error("softmax_last_dim requires rank >= 1 tensor, got rank={got_rank}")]
    InvalidSoftmaxRank { got_rank: usize },
    #[error("log_softmax_last_dim requires rank >= 1 tensor, got rank={got_rank}")]
    InvalidLogSoftmaxRank { got_rank: usize },
    #[error("logsumexp_last_dim requires rank >= 1 tensor, got rank={got_rank}")]
    InvalidLogSumExpRank { got_rank: usize },
    #[error("layer_norm_last_dim requires rank >= 1 tensor, got rank={got_rank}")]
    InvalidLayerNormRank { got_rank: usize },
    #[error(
        "layer_norm_last_dim parameter `{parameter}` shape mismatch: got {shape:?}, expected [{expected_features}]"
    )]
    LayerNormParameterShapeMismatch {
        parameter: &'static str,
        shape: Vec<usize>,
        expected_features: usize,
    },
    #[error("layer_norm_last_dim requires epsilon to be finite and > 0")]
    InvalidLayerNormEpsilon,
    #[error(
        "scaled_dot_product_attention requires rank-2 tensors, got query_rank={query_rank}, key_rank={key_rank}, value_rank={value_rank}"
    )]
    InvalidAttentionRank {
        query_rank: usize,
        key_rank: usize,
        value_rank: usize,
    },
    #[error("scaled_dot_product_attention d_k mismatch: query d_k={query_dk}, key d_k={key_dk}")]
    AttentionDkMismatch { query_dk: usize, key_dk: usize },
    #[error(
        "scaled_dot_product_attention seq_k mismatch: key seq={key_seq}, value seq={value_seq}"
    )]
    AttentionSeqKMismatch { key_seq: usize, value_seq: usize },
    #[error("scaled_dot_product_attention mask shape mismatch: expected {expected:?}, got {got:?}")]
    AttentionMaskShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("group_norm_nhwc requires rank-4 input tensor [N,H,W,C], got rank={got_rank}")]
    InvalidGroupNormRank { got_rank: usize },
    #[error(
        "group_norm_nhwc: channels ({channels}) must be divisible by num_groups ({num_groups})"
    )]
    GroupNormChannelGroupMismatch { channels: usize, num_groups: usize },
    #[error(
        "group_norm_nhwc parameter `{parameter}` shape mismatch: got {shape:?}, expected [{expected_channels}]"
    )]
    GroupNormParameterShapeMismatch {
        parameter: &'static str,
        shape: Vec<usize>,
        expected_channels: usize,
    },
    #[error("group_norm_nhwc requires epsilon to be finite and > 0")]
    InvalidGroupNormEpsilon,
    #[error("group_norm_nhwc requires num_groups > 0")]
    InvalidGroupNormNumGroups,
    #[error("rms_norm_last_dim requires rank >= 1 tensor, got rank={got_rank}")]
    InvalidRmsNormRank { got_rank: usize },
    #[error(
        "rms_norm_last_dim parameter `{parameter}` shape mismatch: got {shape:?}, expected [{expected_features}]"
    )]
    RmsNormParameterShapeMismatch {
        parameter: &'static str,
        shape: Vec<usize>,
        expected_features: usize,
    },
    #[error("rms_norm_last_dim requires epsilon to be finite and > 0")]
    InvalidRmsNormEpsilon,
    #[error(
        "embedding_lookup requires rank-2 weight tensor [vocab_size, embed_dim], got rank={got_rank}"
    )]
    InvalidEmbeddingWeightRank { got_rank: usize },
    #[error("embedding_lookup index {index} out of bounds for vocab_size={vocab_size}")]
    EmbeddingIndexOutOfBounds { index: usize, vocab_size: usize },
    #[error("deformable_conv2d offset shape mismatch: expected {expected:?}, got {got:?}")]
    DeformableConvOffsetShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("no GPU adapter found")]
    GpuNoAdapter,
    #[cfg(feature = "gpu")]
    #[error("GPU device request failed: {source}")]
    GpuDeviceRequest {
        #[from]
        source: wgpu::RequestDeviceError,
    },
    #[error("GPU channel closed during read-back")]
    GpuReadbackChannelClosed,
    #[error("GPU buffer map failed during read-back")]
    GpuBufferMapFailed,
    #[error("{pipeline} pipeline not available on this device")]
    GpuPipelineUnavailable { pipeline: &'static str },
    #[error("GPU op got empty {what}")]
    GpuEmpty { what: &'static str },
    #[error("GPU device index {index} out of range (available: {available})")]
    GpuDeviceIndexOutOfRange { index: usize, available: usize },
    #[error("no GPU devices available")]
    NoGpuDevices,
    #[error("{class} class not available")]
    ObjcClassUnavailable { class: &'static str },
    #[error("kan layer {name} has {got} values, expected {expected}")]
    KanShapeMismatch {
        name: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "kan grid needs grid_size > 0 and lo < hi, got grid_size={grid_size}, range=({lo}, {hi})"
    )]
    InvalidKanGrid { grid_size: usize, lo: f32, hi: f32 },
    #[error("{op}: expected {expected} shape, got {got:?}")]
    UnexpectedShape {
        op: &'static str,
        expected: &'static str,
        got: Vec<usize>,
    },
    #[error("{op}: shape mismatch, expected {expected:?}, got {got:?}")]
    NchwcShapeMismatch {
        op: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("expected NCHWc-tagged tensor, got {layout:?}")]
    NotNchwc { layout: Layout },
    #[error("invalid NCHWc block size {block}, expected a power of two in 1..=255")]
    InvalidNchwcBlock { block: usize },
    #[error("NCHWc block size mismatch: {left} vs {right}")]
    NchwcBlockMismatch { left: usize, right: usize },
    #[error("NCHWc channel count {channels} outside 1..={capacity}")]
    InvalidNchwcChannels { channels: usize, capacity: usize },
    #[error(
        "{channels} channels not divisible by NCHWc block {block} (caller must dispatch to NHWC path)"
    )]
    NchwcChannelsNotBlockAligned { channels: usize, block: usize },
    #[error("{op}: {what} has {got} channels, expected {expected}")]
    NchwcChannelMismatch {
        op: &'static str,
        what: &'static str,
        got: usize,
        expected: usize,
    },
    #[cfg(feature = "rknn")]
    #[error("{func} failed: {} ({status})", crate::rknn::ffi::rknn_error_name(*status))]
    RknnCall { func: &'static str, status: i32 },
    #[cfg(feature = "rknn")]
    #[error(
        "rknn_wait failed: {} ({status}) — frame {frame_id} after {elapsed:?}",
        crate::rknn::ffi::rknn_error_name(*status)
    )]
    RknnWait {
        status: i32,
        frame_id: u64,
        elapsed: std::time::Duration,
    },
    #[cfg(feature = "rknn")]
    #[error("RKNN symbol `{symbol}` not available")]
    RknnSymbolMissing { symbol: &'static str },
    #[cfg(feature = "rknn")]
    #[error("failed to load {library}")]
    RknnLibraryUnavailable { library: &'static str },
    #[cfg(feature = "rknn")]
    #[error("RKNN compiler only available on Linux")]
    RknnCompilerUnsupportedPlatform,
    #[cfg(feature = "rknn")]
    #[error("{func} returned null")]
    RknnNullPointer { func: &'static str },
    #[cfg(feature = "rknn")]
    #[error("RKNN {lock} lock poisoned (slot {slot:?})")]
    RknnLockPoisoned {
        lock: &'static str,
        slot: Option<usize>,
    },
    #[cfg(feature = "rknn")]
    #[error("RKNN {what} index {index} out of range (have {len})")]
    RknnIndexOutOfRange {
        what: &'static str,
        index: usize,
        len: usize,
    },
    #[cfg(feature = "rknn")]
    #[error("no RKNN {what} named `{name}`")]
    RknnUnknownTensor { what: &'static str, name: String },
    #[cfg(feature = "rknn")]
    #[error("missing RKNN input `{name}`")]
    RknnMissingInput { name: String },
    #[cfg(feature = "rknn")]
    #[error("expected {expected} {what}, got {got}")]
    RknnCountMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[cfg(feature = "rknn")]
    #[error(
        "input `{name}`: {got} bytes but the model expects exactly {expected} \
         (check your pre-processing matches the .rknn's input shape/dtype)"
    )]
    RknnInputSize {
        name: String,
        expected: usize,
        got: usize,
    },
    #[cfg(feature = "rknn")]
    #[error("{op}: {reason}")]
    RknnInvalidCall {
        op: &'static str,
        reason: &'static str,
    },
    #[cfg(feature = "rknn")]
    #[error(
        "slot {slot} has no pending frame — handle already waited on or slot was reused by a later submit"
    )]
    RknnNoPendingFrame { slot: usize },
    #[cfg(feature = "rknn")]
    #[error("{what} {path:?}: {source}")]
    RknnIo {
        what: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[cfg(feature = "rknn")]
    #[error("path {path:?} contains an interior NUL byte")]
    RknnPathContainsNul { path: std::path::PathBuf },
    #[cfg(feature = "rknn")]
    #[error("calibration dataset {path:?} is {reason}")]
    RknnCalibrationDataset {
        path: std::path::PathBuf,
        reason: &'static str,
    },
}
