use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use yscv_tensor::TensorError;

use crate::VideoCodec;
use crate::hw_decode::HwBackend;

#[derive(Debug, Clone, Error)]
pub enum VideoError {
    #[error("invalid frame tensor shape: expected [H, W, C], got {got:?}")]
    InvalidFrameShape { got: Vec<usize> },
    #[error("unsupported channel count: {channels}; expected 1 or 3")]
    UnsupportedChannelCount { channels: usize },
    #[error(
        "invalid camera resolution: width={width}, height={height}; expected width > 0 and height > 0"
    )]
    InvalidCameraResolution { width: u32, height: u32 },
    #[error("invalid camera fps: {fps}; expected fps > 0")]
    InvalidCameraFps { fps: u32 },
    #[error("invalid camera device query `{query}`; expected non-empty value")]
    InvalidCameraDeviceQuery { query: String },
    #[error("no camera device matched query `{query}`; run device listing to inspect names")]
    CameraDeviceNotFound { query: String },
    #[error(
        "camera query `{query}` matched multiple devices: {}; refine query",
        matches.join(", ")
    )]
    CameraDeviceAmbiguous { query: String, matches: Vec<String> },
    #[error("invalid raw frame buffer size: expected {expected} bytes, got {got}")]
    RawFrameSizeMismatch { expected: usize, got: usize },
    #[error("invalid normalized output buffer size: expected {expected} f32 values, got {got}")]
    NormalizedBufferSizeMismatch { expected: usize, got: usize },
    #[error("native camera backend is disabled; enable `yscv-video` feature `native-camera`")]
    CameraBackendDisabled,
    #[error("truncated {what}")]
    Truncated { what: &'static str },
    #[error("{what} at offset {offset} spans {len} bytes, past the {available} available")]
    OutOfBounds {
        what: &'static str,
        offset: usize,
        len: usize,
        available: usize,
    },
    #[error("missing {what}")]
    Missing { what: &'static str },
    #[error("malformed {what}")]
    Malformed { what: &'static str },
    #[error("{field} out of range: {value}")]
    FieldOutOfRange { field: &'static str, value: i64 },
    #[error("invalid {what} dimensions {width}x{height}")]
    InvalidDimensions {
        what: &'static str,
        width: usize,
        height: usize,
    },
    #[error("{what} too small: expected {expected}, got {got}")]
    BufferTooSmall {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "{} is {size} bytes, over the {limit}-byte limit for in-memory parsing",
        path.display()
    )]
    FileTooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    #[error("{what} unavailable")]
    Unavailable { what: &'static str },
    #[error("{codec:?} decoder panicked")]
    DecodePanicked { codec: VideoCodec },
    #[error("{backend} cannot decode {codec:?}")]
    UnsupportedCodec {
        backend: HwBackend,
        codec: VideoCodec,
    },
    #[error("{backend} {op} failed with status {status}")]
    HwDecode {
        backend: HwBackend,
        op: &'static str,
        status: i32,
    },
    #[error(
        "MAVLink CRC mismatch for msgid {msgid}: expected {expected:#06X}, computed {computed:#06X}"
    )]
    MavlinkCrcMismatch {
        msgid: u32,
        expected: u16,
        computed: u16,
    },
    #[error(
        "{op} failed{}: {source}",
        path.as_ref().map(|p| format!(" for {}", p.display())).unwrap_or_default()
    )]
    Io {
        op: &'static str,
        path: Option<PathBuf>,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[cfg(feature = "native-camera")]
    #[error("camera {op} failed: {source}")]
    Camera {
        op: &'static str,
        #[source]
        source: Box<nokhwa::NokhwaError>,
    },
    #[cfg(feature = "native-camera")]
    #[error(
        "camera {op} failed with backend {backend:?}: {primary}; auto fallback failed: {source}"
    )]
    CameraFallback {
        op: &'static str,
        backend: nokhwa::utils::ApiBackend,
        primary: Box<nokhwa::NokhwaError>,
        #[source]
        source: Box<nokhwa::NokhwaError>,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
}
