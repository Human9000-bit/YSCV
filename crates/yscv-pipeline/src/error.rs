use std::path::PathBuf;
use thiserror::Error;
use yscv_kernels::KernelError;
use yscv_onnx::OnnxError;
use yscv_tensor::TensorError;

/// Errors raised when loading or validating a [`crate::PipelineConfig`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// File could not be read or doesn't exist.
    #[error("failed to read config file {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// TOML parse failure.
    #[error("invalid TOML in {path:?}: {source}")]
    Toml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    /// User asked for an accelerator that isn't compiled in or whose
    /// runtime library is missing on this host.
    #[error(
        "task '{task}' requested {accelerator} but it is not available — \
         either compile with the matching feature flag (`--features {feature_hint}`) \
         or run on a host where the runtime library is present"
    )]
    AcceleratorUnavailable {
        task: String,
        accelerator: String,
        feature_hint: &'static str,
    },

    /// Model file referenced by a task does not exist.
    #[error("task '{task}': model file not found at {path:?}")]
    ModelNotFound { task: String, path: PathBuf },

    /// Model file exists but could not be read.
    #[error("task '{task}': failed to read model at {path:?}: {source}")]
    ModelRead {
        task: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Model file is too short to hold its format's magic bytes
    /// (`len == 0` for an empty file).
    #[error("task '{task}': model at {path:?} is only {len} bytes — truncated?")]
    ModelTruncated {
        task: String,
        path: PathBuf,
        len: usize,
    },

    /// `.rknn` model does not start with the `RKNN` / `RKNF` magic.
    #[error(
        "task '{task}': model at {path:?} has magic {magic:02x?}, expected RKNN / RKNF — \
         is this really a .rknn file? (export via rknn-toolkit2)"
    )]
    BadRknnMagic {
        task: String,
        path: PathBuf,
        magic: [u8; 4],
    },

    /// ONNX model does not start with the protobuf `ir_version` tag `0x08`.
    #[error(
        "task '{task}': model at {path:?} starts with {first_byte:#04x}, expected ONNX \
         protobuf (first byte 0x08 for ir_version) — is this really an .onnx file?"
    )]
    BadOnnxMagic {
        task: String,
        path: PathBuf,
        first_byte: u8,
    },

    /// The RKNN SDK refused a model during the `rknn-validate` dry-run load.
    #[cfg(feature = "rknn-validate")]
    #[error("task '{task}': rknn SDK refused the model at {path:?}: {source}")]
    RknnRejected {
        task: String,
        path: PathBuf,
        #[source]
        source: Box<KernelError>,
    },

    /// Task graph contains a cycle (A depends on B which depends on A).
    #[error("pipeline graph has a cycle through task '{task}'")]
    CyclicDependency { task: String },

    /// Task references an input that no other task or source produces.
    #[error("task '{task}' references unknown input source '{src_name}'")]
    UnknownSource { task: String, src_name: String },

    /// A numeric field that must be positive is zero.
    #[error("{field} must be > 0")]
    ZeroField { field: &'static str },

    /// Two tasks share the same name.
    #[error("duplicate task name '{task}'")]
    DuplicateTask { task: String },
}

/// Top-level error type returned from pipeline runtime entry points.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Config(#[from] ConfigError),

    /// Errors from the underlying yscv-kernels backend.
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// Errors from the yscv-onnx CPU, wgpu or MPSGraph runtime.
    #[error(transparent)]
    Onnx(#[from] OnnxError),

    /// I/O failure during pipeline execution.
    #[error("pipeline I/O: {0}")]
    Io(#[from] std::io::Error),

    /// The task's accelerator is not compiled into this build.
    #[error(
        "task '{task}' needs Accelerator::{accelerator}, which requires building \
         yscv-pipeline with {requires}"
    )]
    AcceleratorNotCompiled {
        task: String,
        accelerator: &'static str,
        requires: &'static str,
    },

    /// Loading a task's ONNX model failed.
    #[error("task '{task}': load ONNX {path:?} failed — {source}")]
    LoadModel {
        task: String,
        path: PathBuf,
        #[source]
        source: Box<OnnxError>,
    },

    /// Reading a task's model file failed.
    #[cfg(feature = "rknn")]
    #[error("task '{task}': read {path:?} failed — {source}")]
    ModelRead {
        task: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// On-device ONNX → RKNN compilation failed.
    #[cfg(feature = "rknn")]
    #[error("task '{task}': compile {path:?} → .rknn failed — {source}")]
    RknnCompile {
        task: String,
        path: PathBuf,
        #[source]
        source: Box<KernelError>,
    },

    /// `Accelerator::Rknn` model path is neither `.rknn` nor `.onnx`.
    #[cfg(feature = "rknn")]
    #[error(
        "task '{task}': unsupported model extension of {path:?} for Accelerator::Rknn \
         (expected `.rknn` or `.onnx`)"
    )]
    UnsupportedModelExtension { task: String, path: PathBuf },

    /// `Accelerator::RknnMatmul` has a zero dimension.
    #[cfg(feature = "rknn")]
    #[error("task '{task}': matmul dims must be > 0 (got {m}×{k}×{n})")]
    ZeroMatmulDim {
        task: String,
        m: u32,
        k: u32,
        n: u32,
    },

    /// Matmul dispatcher got an input other than `a` or `b`.
    #[cfg(feature = "rknn")]
    #[error("matmul dispatcher: unknown input '{name}' (expected 'a' or 'b')")]
    UnknownMatmulInput { name: String },

    /// Matmul dispatcher was not given input `a` or `b`.
    #[cfg(feature = "rknn")]
    #[error("matmul dispatcher: missing input '{name}'")]
    MissingMatmulInput { name: &'static str },

    /// Matmul input byte count differs from the SDK-declared size.
    #[cfg(feature = "rknn")]
    #[error(
        "matmul input '{name}': {got} bytes but the model expects exactly {expected} \
         (with the configured dtype)"
    )]
    MatmulInputSize {
        name: &'static str,
        expected: usize,
        got: usize,
    },

    /// A dispatcher mutex was poisoned by a panic on another thread.
    #[cfg(any(feature = "rknn", all(feature = "metal-backend", target_os = "macos")))]
    #[error("{lock} lock poisoned")]
    LockPoisoned { lock: &'static str },

    /// Input byte count is not a multiple of 4 (inputs are f32 LE).
    #[error("input '{input}': {len} bytes is not a whole number of f32s")]
    InputNotF32 { input: String, len: usize },

    /// Building the f32 tensor for an input failed.
    #[error("input '{input}': tensor build failed — {source}")]
    BuildInput {
        input: String,
        #[source]
        source: TensorError,
    },

    /// Scheduler order names a task with no registered metadata or
    /// dispatcher (internal invariant).
    #[error("scheduler has no {what} for task '{task}'")]
    MissingTaskState { task: String, what: &'static str },

    /// A task input's source is produced by nothing that ran before it.
    #[error("task '{task}': input '{input}' source '{src_name}' has no producer")]
    NoProducer {
        task: String,
        input: String,
        src_name: String,
    },

    /// A dispatcher's `recover` hook failed.
    #[error("recover dispatcher '{dispatcher}' failed: {source}")]
    Recover {
        dispatcher: String,
        #[source]
        source: Box<Error>,
    },
}
