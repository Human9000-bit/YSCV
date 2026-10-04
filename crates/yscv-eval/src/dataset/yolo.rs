use rustc_hash::FxHashSet;
use std::path::Path;

use yscv_detect::{BoundingBox, Detection};

use crate::{DatasetFormatError, DetectionDatasetFrame, EvalError, LabeledBox};

use super::helpers::read_optional_text_file;
use super::types::YoloManifestEntry;

pub(crate) fn load_yolo_label_dirs(
    manifest_text: &str,
    ground_truth_labels_dir: &Path,
    prediction_labels_dir: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let entries =
        parse_yolo_manifest(manifest_text).map_err(|source| EvalError::InvalidDatasetFormat {
            format: "yolo",
            source,
        })?;
    let mut frames = Vec::with_capacity(entries.len());

    for entry in entries {
        let gt_path = yolo_label_path(ground_truth_labels_dir, &entry.image_id);
        let pred_path = yolo_label_path(prediction_labels_dir, &entry.image_id);

        let ground_truth = match read_optional_text_file(&gt_path)? {
            Some(text) => parse_yolo_ground_truth_labels(&text, &entry).map_err(|source| {
                EvalError::InvalidDatasetFile {
                    format: "yolo",
                    path: gt_path,
                    source,
                }
            })?,
            None => Vec::new(),
        };
        let predictions = match read_optional_text_file(&pred_path)? {
            Some(text) => parse_yolo_predictions(&text, &entry).map_err(|source| {
                EvalError::InvalidDatasetFile {
                    format: "yolo",
                    path: pred_path,
                    source,
                }
            })?,
            None => Vec::new(),
        };

        frames.push(DetectionDatasetFrame {
            ground_truth,
            predictions,
        });
    }

    Ok(frames)
}

fn parse_yolo_manifest(text: &str) -> Result<Vec<YoloManifestEntry>, DatasetFormatError> {
    const RECORD: &str = "manifest line";
    let mut entries = Vec::new();
    let mut seen_ids = FxHashSet::default();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        let image_id = parts.next().unwrap_or_default();
        let width_raw = parts.next();
        let height_raw = parts.next();
        let extra = parts.next();

        if image_id.is_empty() || width_raw.is_none() || height_raw.is_none() || extra.is_some() {
            return Err(DatasetFormatError::FieldCount {
                record: RECORD,
                index: line_no,
                expected: "`<image_id> <width> <height>`",
            });
        }

        let width = parse_positive_usize(width_raw.unwrap_or_default(), "width", RECORD, line_no)?;
        let height =
            parse_positive_usize(height_raw.unwrap_or_default(), "height", RECORD, line_no)?;

        if !seen_ids.insert(image_id.to_string()) {
            return Err(DatasetFormatError::DuplicateImageId {
                record: RECORD,
                index: line_no,
                id: image_id.to_string(),
            });
        }

        entries.push(YoloManifestEntry {
            image_id: image_id.to_string(),
            width,
            height,
        });
    }
    Ok(entries)
}

fn yolo_label_path(labels_dir: &Path, image_id: &str) -> std::path::PathBuf {
    labels_dir.join(format!("{image_id}.txt"))
}

fn parse_yolo_ground_truth_labels(
    text: &str,
    entry: &YoloManifestEntry,
) -> Result<Vec<LabeledBox>, DatasetFormatError> {
    const RECORD: &str = "ground-truth line";
    let mut boxes = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 5 {
            return Err(DatasetFormatError::FieldCount {
                record: RECORD,
                index: line_no,
                expected: "5 fields `<class_id> <x_center> <y_center> <width> <height>`",
            });
        }
        let class_id = parse_usize_token(tokens[0], "class_id", RECORD, line_no)?;
        let bbox = parse_yolo_bbox_tokens(&tokens[1..5], entry, RECORD, line_no)?;
        boxes.push(LabeledBox { bbox, class_id });
    }
    Ok(boxes)
}

fn parse_yolo_predictions(
    text: &str,
    entry: &YoloManifestEntry,
) -> Result<Vec<Detection>, DatasetFormatError> {
    const RECORD: &str = "prediction line";
    let mut detections = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 6 {
            return Err(DatasetFormatError::FieldCount {
                record: RECORD,
                index: line_no,
                expected: "6 fields `<class_id> <x_center> <y_center> <width> <height> <score>`",
            });
        }

        let class_id = parse_usize_token(tokens[0], "class_id", RECORD, line_no)?;
        let bbox = parse_yolo_bbox_tokens(&tokens[1..5], entry, RECORD, line_no)?;
        let score = parse_unit_interval_f32(tokens[5], "score", RECORD, line_no)?;
        detections.push(Detection {
            bbox,
            score,
            class_id,
        });
    }
    Ok(detections)
}

fn parse_yolo_bbox_tokens(
    tokens: &[&str],
    entry: &YoloManifestEntry,
    record: &'static str,
    line_no: usize,
) -> Result<BoundingBox, DatasetFormatError> {
    let x_center = parse_unit_interval_f32(tokens[0], "x_center", record, line_no)?;
    let y_center = parse_unit_interval_f32(tokens[1], "y_center", record, line_no)?;
    let width = parse_unit_interval_f32(tokens[2], "width", record, line_no)?;
    let height = parse_unit_interval_f32(tokens[3], "height", record, line_no)?;
    if width == 0.0 || height == 0.0 {
        return Err(DatasetFormatError::InvalidBox {
            record,
            index: line_no,
            reason: "width and height must be > 0",
        });
    }

    let image_width = entry.width as f32;
    let image_height = entry.height as f32;
    let x1 = ((x_center - width * 0.5) * image_width).clamp(0.0, image_width);
    let y1 = ((y_center - height * 0.5) * image_height).clamp(0.0, image_height);
    let x2 = ((x_center + width * 0.5) * image_width).clamp(0.0, image_width);
    let y2 = ((y_center + height * 0.5) * image_height).clamp(0.0, image_height);
    if x2 <= x1 || y2 <= y1 {
        return Err(DatasetFormatError::InvalidBox {
            record,
            index: line_no,
            reason: "normalized bbox collapses after conversion",
        });
    }

    Ok(BoundingBox { x1, y1, x2, y2 })
}

fn parse_positive_usize(
    token: &str,
    field: &'static str,
    record: &'static str,
    line_no: usize,
) -> Result<usize, DatasetFormatError> {
    let value = parse_usize_token(token, field, record, line_no)?;
    if value == 0 {
        return Err(DatasetFormatError::OutOfRange {
            record,
            index: line_no,
            field,
            expected: "> 0",
        });
    }
    Ok(value)
}

fn parse_usize_token(
    token: &str,
    field: &'static str,
    record: &'static str,
    line_no: usize,
) -> Result<usize, DatasetFormatError> {
    token
        .parse::<usize>()
        .map_err(|source| DatasetFormatError::ParseInt {
            record,
            index: line_no,
            field,
            value: token.to_string(),
            source,
        })
}

fn parse_unit_interval_f32(
    token: &str,
    field: &'static str,
    record: &'static str,
    line_no: usize,
) -> Result<f32, DatasetFormatError> {
    let value = token
        .parse::<f32>()
        .map_err(|source| DatasetFormatError::ParseFloat {
            record,
            index: line_no,
            field,
            value: token.to_string(),
            source,
        })?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(DatasetFormatError::OutOfRange {
            record,
            index: line_no,
            field,
            expected: "finite and in [0, 1]",
        });
    }
    Ok(value)
}
