use std::num::{ParseFloatError, ParseIntError, TryFromIntError};
use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum EvalError {
    #[error("invalid IoU threshold: {value}; expected finite threshold in [0, 1]")]
    InvalidIouThreshold { value: f32 },
    #[error("invalid score threshold: {value}; expected finite threshold in [0, 1]")]
    InvalidScoreThreshold { value: f32 },
    #[error(
        "count sequence length mismatch: ground_truth={ground_truth}, predictions={predictions}"
    )]
    CountLengthMismatch {
        ground_truth: usize,
        predictions: usize,
    },
    #[error("duration series must contain at least one item")]
    EmptyDurationSeries,
    #[error("duration series length mismatch for `{series}`: expected {expected}, got {got}")]
    DurationSeriesLengthMismatch {
        expected: usize,
        got: usize,
        series: &'static str,
    },
    #[error("invalid threshold entry at line {line}: expected {expected}")]
    MalformedThresholdEntry { line: usize, expected: &'static str },
    #[error("invalid threshold entry at line {line}: invalid numeric value `{value}`: {source}")]
    InvalidThresholdNumber {
        line: usize,
        value: String,
        #[source]
        source: ParseFloatError,
    },
    #[error(
        "invalid threshold entry at line {line}: threshold value must be finite and non-negative, got {value}"
    )]
    InvalidThresholdValue { line: usize, value: f64 },
    #[error("invalid threshold entry at line {line}: unsupported {key} `{name}`")]
    UnsupportedThresholdKey {
        line: usize,
        key: &'static str,
        name: String,
    },
    #[error("invalid dataset entry at line {line}: {source}")]
    InvalidDatasetEntry {
        line: usize,
        #[source]
        source: Arc<serde_json::Error>,
    },
    #[error("invalid dataset format `{format}`: {source}")]
    InvalidDatasetFormat {
        format: &'static str,
        #[source]
        source: DatasetFormatError,
    },
    #[error("invalid dataset format `{format}` in {path:?}: {source}")]
    InvalidDatasetFile {
        format: &'static str,
        path: PathBuf,
        #[source]
        source: DatasetFormatError,
    },
    #[error("dataset I/O error for {path:?}: {source}")]
    DatasetIo {
        path: PathBuf,
        #[source]
        source: Arc<std::io::Error>,
    },
    #[error("invalid diagnostics report: {source}")]
    InvalidDiagnosticsReport {
        #[source]
        source: Arc<serde_json::Error>,
    },
    #[error("diagnostics report I/O error for {path:?}: {source}")]
    DiagnosticsReportIo {
        path: PathBuf,
        #[source]
        source: Arc<std::io::Error>,
    },
}

/// What was wrong inside a dataset input, carried by
/// [`EvalError::InvalidDatasetFormat`] and [`EvalError::InvalidDatasetFile`].
///
/// `record` names the kind of record the problem was found in (for example
/// `"ground-truth line"`, `"prediction row"`, `"object"`, `"annotation"`) and
/// `index` is its position as the format counts it: 1-based for text lines
/// and CSV rows, 0-based for JSON array entries and XML objects.
#[derive(Debug, Clone, Error)]
pub enum DatasetFormatError {
    #[error("invalid JSON: {0}")]
    Json(#[source] Arc<serde_json::Error>),
    #[error("invalid XML: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("{record} {index}: unreadable CSV record: {source}")]
    Csv {
        record: &'static str,
        index: usize,
        #[source]
        source: Arc<csv::Error>,
    },
    #[error("missing required header `{header}`")]
    MissingHeader { header: &'static str },
    #[error("{record} {index}: expected {expected}")]
    FieldCount {
        record: &'static str,
        index: usize,
        expected: &'static str,
    },
    #[error("{record} {index}: missing `{field}`")]
    MissingField {
        record: &'static str,
        index: usize,
        field: &'static str,
    },
    #[error("{record} {index}: invalid `{field}` value `{value}`: {source}")]
    ParseFloat {
        record: &'static str,
        index: usize,
        field: &'static str,
        value: String,
        #[source]
        source: ParseFloatError,
    },
    #[error("{record} {index}: invalid `{field}` value `{value}`: {source}")]
    ParseInt {
        record: &'static str,
        index: usize,
        field: &'static str,
        value: String,
        #[source]
        source: ParseIntError,
    },
    #[error("{record} {index}: `{field}` value {value} exceeds usize::MAX on this platform")]
    IntOverflow {
        record: &'static str,
        index: usize,
        field: &'static str,
        value: u64,
        #[source]
        source: TryFromIntError,
    },
    #[error("{record} {index}: `{field}` must be {expected}")]
    OutOfRange {
        record: &'static str,
        index: usize,
        field: &'static str,
        expected: &'static str,
    },
    #[error("{record} {index}: invalid bbox: {reason}")]
    InvalidBox {
        record: &'static str,
        index: usize,
        reason: &'static str,
    },
    #[error("{record} {index}: duplicate image id `{id}`")]
    DuplicateImageId {
        record: &'static str,
        index: usize,
        id: String,
    },
    #[error("{record} {index}: unknown image id `{id}`")]
    UnknownImageId {
        record: &'static str,
        index: usize,
        id: String,
    },
    #[error("{record} {index}: missing box count for image `{image_id}`")]
    MissingBoxCount {
        record: &'static str,
        index: usize,
        image_id: String,
    },
    #[error(
        "{record} {index}: image `{image_id}` declares {expected} boxes but only {got} rows follow"
    )]
    MissingBoxRows {
        record: &'static str,
        index: usize,
        image_id: String,
        expected: usize,
        got: usize,
    },
}

impl From<serde_json::Error> for DatasetFormatError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(Arc::new(source))
    }
}
