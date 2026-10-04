use rustc_hash::FxHashMap;
use std::path::Path;

use yscv_detect::{BoundingBox, Detection};

use crate::{DatasetFormatError, DetectionDatasetFrame, EvalError, LabeledBox};

use super::helpers::{parse_image_id_manifest, read_optional_text_file};

pub(crate) fn load_kitti_label_dirs(
    manifest_text: &str,
    ground_truth_labels_dir: &Path,
    prediction_labels_dir: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let image_ids = parse_image_id_manifest(manifest_text).map_err(|source| {
        EvalError::InvalidDatasetFormat {
            format: "kitti",
            source,
        }
    })?;
    let mut frames = Vec::with_capacity(image_ids.len());
    let mut class_map = FxHashMap::default();

    for image_id in image_ids {
        let gt_path = kitti_label_path(ground_truth_labels_dir, &image_id);
        let pred_path = kitti_label_path(prediction_labels_dir, &image_id);

        let ground_truth = match read_optional_text_file(&gt_path)? {
            Some(text) => {
                parse_kitti_ground_truth_labels(&text, &mut class_map).map_err(|source| {
                    EvalError::InvalidDatasetFile {
                        format: "kitti",
                        path: gt_path,
                        source,
                    }
                })?
            }
            None => Vec::new(),
        };
        let predictions = match read_optional_text_file(&pred_path)? {
            Some(text) => parse_kitti_predictions(&text, &mut class_map).map_err(|source| {
                EvalError::InvalidDatasetFile {
                    format: "kitti",
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

fn kitti_label_path(labels_dir: &Path, image_id: &str) -> std::path::PathBuf {
    labels_dir.join(format!("{image_id}.txt"))
}

fn parse_kitti_ground_truth_labels(
    text: &str,
    class_map: &mut FxHashMap<String, usize>,
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
        if tokens.len() < 8 {
            return Err(DatasetFormatError::FieldCount {
                record: RECORD,
                index: line_no,
                expected: "at least 8 fields",
            });
        }
        if is_kitti_dont_care(tokens[0]) {
            continue;
        }

        let class_id = resolve_kitti_class_id(tokens[0], class_map);
        let bbox = parse_kitti_bbox_tokens(&tokens[4..8], RECORD, line_no)?;
        boxes.push(LabeledBox { bbox, class_id });
    }
    Ok(boxes)
}

fn parse_kitti_predictions(
    text: &str,
    class_map: &mut FxHashMap<String, usize>,
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
        if tokens.len() < 8 {
            return Err(DatasetFormatError::FieldCount {
                record: RECORD,
                index: line_no,
                expected: "at least 8 fields",
            });
        }
        if is_kitti_dont_care(tokens[0]) {
            continue;
        }

        let class_id = resolve_kitti_class_id(tokens[0], class_map);
        let bbox = parse_kitti_bbox_tokens(&tokens[4..8], RECORD, line_no)?;
        let score = match tokens.get(15).copied() {
            Some(raw) => parse_kitti_score(raw, RECORD, line_no)?,
            None => 1.0,
        };
        detections.push(Detection {
            bbox,
            score,
            class_id,
        });
    }
    Ok(detections)
}

fn parse_kitti_bbox_tokens(
    tokens: &[&str],
    record: &'static str,
    line_no: usize,
) -> Result<BoundingBox, DatasetFormatError> {
    let x1 = parse_kitti_f32(tokens[0], "bbox_left", record, line_no)?;
    let y1 = parse_kitti_f32(tokens[1], "bbox_top", record, line_no)?;
    let x2 = parse_kitti_f32(tokens[2], "bbox_right", record, line_no)?;
    let y2 = parse_kitti_f32(tokens[3], "bbox_bottom", record, line_no)?;
    if x2 <= x1 || y2 <= y1 {
        return Err(DatasetFormatError::InvalidBox {
            record,
            index: line_no,
            reason: "expected bbox_right>bbox_left and bbox_bottom>bbox_top",
        });
    }
    Ok(BoundingBox { x1, y1, x2, y2 })
}

fn parse_kitti_f32(
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
    if !value.is_finite() {
        return Err(DatasetFormatError::OutOfRange {
            record,
            index: line_no,
            field,
            expected: "finite",
        });
    }
    Ok(value)
}

fn parse_kitti_score(
    token: &str,
    record: &'static str,
    line_no: usize,
) -> Result<f32, DatasetFormatError> {
    let score = parse_kitti_f32(token, "score", record, line_no)?;
    if score < 0.0 {
        return Err(DatasetFormatError::OutOfRange {
            record,
            index: line_no,
            field: "score",
            expected: ">= 0",
        });
    }
    Ok(score)
}

fn resolve_kitti_class_id(class_name: &str, class_map: &mut FxHashMap<String, usize>) -> usize {
    if let Some(&class_id) = class_map.get(class_name) {
        return class_id;
    }
    let class_id = class_map.len();
    class_map.insert(class_name.to_string(), class_id);
    class_id
}

const fn is_kitti_dont_care(class_name: &str) -> bool {
    class_name.eq_ignore_ascii_case("DontCare")
}
