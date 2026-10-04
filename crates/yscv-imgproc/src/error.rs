use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use yscv_tensor::TensorError;

#[derive(Debug, Clone, Error)]
pub enum ImgProcError {
    #[error("invalid image shape: expected rank {expected_rank}, got {got:?}")]
    InvalidImageShape {
        expected_rank: usize,
        got: Vec<usize>,
    },
    #[error("invalid channel count: expected {expected}, got {got}")]
    InvalidChannelCount { expected: usize, got: usize },
    #[error("invalid output size: height={height}, width={width}; both must be > 0")]
    InvalidSize { height: usize, width: usize },
    #[error(
        "invalid normalization params for {expected_channels} channels: mean_len={mean_len}, std_len={std_len}"
    )]
    InvalidNormalizationParams {
        expected_channels: usize,
        mean_len: usize,
        std_len: usize,
    },
    #[error("std must be non-zero at channel {channel}")]
    ZeroStdAtChannel { channel: usize },
    #[error("invalid block size: {block_size}; must be odd and > 0")]
    InvalidBlockSize { block_size: usize },
    #[error("invalid output dimensions: height={out_h}, width={out_w}; both must be > 0")]
    InvalidOutputDimensions { out_h: usize, out_w: usize },
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("failed to open image {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[error("unsupported image format: {path:?}")]
    UnsupportedFormat { path: PathBuf },
    #[error("image decode error for {path:?}: {source}")]
    ImageDecode {
        path: PathBuf,
        #[source]
        source: Arc<image::ImageError>,
    },
    #[error("image encode error for {path:?}: {source}")]
    ImageEncode {
        path: PathBuf,
        #[source]
        source: Arc<image::ImageError>,
    },
    #[error("{height}x{width}x{channels} image does not fit an image buffer")]
    ImageBufferSize {
        height: usize,
        width: usize,
        channels: usize,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
}
