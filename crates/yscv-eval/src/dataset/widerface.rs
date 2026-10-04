use rustc_hash::FxHashSet;
use rustc_hash::{FxBuildHasher, FxHashMap};

use yscv_detect::{BoundingBox, CLASS_ID_FACE, Detection};

use crate::{DatasetFormatError, DetectionDatasetFrame, LabeledBox};

#[derive(Debug, Clone, PartialEq)]
struct WiderFaceGroundTruthFrame {
    image_id: String,
    ground_truth: Vec<LabeledBox>,
}

#[derive(Debug, Clone, PartialEq)]
struct WiderFacePredictionFrame {
    image_id: String,
    line: usize,
    predictions: Vec<Detection>,
}

pub(crate) fn parse_and_build_widerface(
    ground_truth_text: &str,
    predictions_text: &str,
) -> Result<Vec<DetectionDatasetFrame>, DatasetFormatError> {
    let ground_truth_frames = parse_widerface_ground_truth(ground_truth_text)?;
    let prediction_frames = parse_widerface_predictions(predictions_text)?;

    // parse_widerface_predictions already rejects duplicate image ids.
    let mut predictions_by_image =
        FxHashMap::with_capacity_and_hasher(prediction_frames.len(), FxBuildHasher);
    for frame in prediction_frames {
        predictions_by_image.insert(frame.image_id, (frame.line, frame.predictions));
    }

    let mut frames = Vec::with_capacity(ground_truth_frames.len());
    for frame in ground_truth_frames {
        let predictions = predictions_by_image
            .remove(&frame.image_id)
            .map(|(_, predictions)| predictions)
            .unwrap_or_default();
        frames.push(DetectionDatasetFrame {
            ground_truth: frame.ground_truth,
            predictions,
        });
    }

    if let Some((image_id, (line, _))) = predictions_by_image.into_iter().next() {
        return Err(DatasetFormatError::UnknownImageId {
            record: "prediction line",
            index: line,
            id: image_id,
        });
    }

    Ok(frames)
}

fn parse_widerface_ground_truth(
    text: &str,
) -> Result<Vec<WiderFaceGroundTruthFrame>, DatasetFormatError> {
    const RECORD: &str = "ground-truth line";
    let lines = widerface_significant_lines(text);
    let mut cursor = 0usize;
    let mut frames = Vec::new();
    let mut seen_ids = FxHashSet::default();

    while cursor < lines.len() {
        let (image_line_no, image_id_raw) = lines[cursor];
        cursor += 1;
        let image_id = image_id_raw.to_string();
        if !seen_ids.insert(image_id.clone()) {
            return Err(DatasetFormatError::DuplicateImageId {
                record: RECORD,
                index: image_line_no,
                id: image_id,
            });
        }

        let Some((count_line_no, count_raw)) = lines.get(cursor).copied() else {
            return Err(DatasetFormatError::MissingBoxCount {
                record: RECORD,
                index: image_line_no,
                image_id,
            });
        };
        cursor += 1;
        let box_count = parse_widerface_box_count(count_raw, RECORD, count_line_no)?;
        let mut ground_truth = Vec::with_capacity(box_count);

        for box_idx in 0..box_count {
            let Some((line_no, row)) = lines.get(cursor).copied() else {
                return Err(DatasetFormatError::MissingBoxRows {
                    record: RECORD,
                    index: image_line_no,
                    image_id,
                    expected: box_count,
                    got: box_idx,
                });
            };
            cursor += 1;
            let tokens = row.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 4 {
                return Err(DatasetFormatError::FieldCount {
                    record: RECORD,
                    index: line_no,
                    expected: "at least 4 fields `<x> <y> <w> <h>`",
                });
            }

            let x1 = parse_widerface_f32(tokens[0], "x", RECORD, line_no)?;
            let y1 = parse_widerface_f32(tokens[1], "y", RECORD, line_no)?;
            let width = parse_widerface_f32(tokens[2], "w", RECORD, line_no)?;
            let height = parse_widerface_f32(tokens[3], "h", RECORD, line_no)?;
            if width <= 0.0 || height <= 0.0 {
                return Err(DatasetFormatError::InvalidBox {
                    record: RECORD,
                    index: line_no,
                    reason: "`w` and `h` must be > 0",
                });
            }
            if let Some(raw_invalid) = tokens.get(7).copied() {
                let invalid = parse_widerface_i32(raw_invalid, "invalid", RECORD, line_no)?;
                if invalid != 0 {
                    continue;
                }
            }

            let x2 = x1 + width;
            let y2 = y1 + height;
            if !x2.is_finite() || !y2.is_finite() {
                return Err(DatasetFormatError::InvalidBox {
                    record: RECORD,
                    index: line_no,
                    reason: "bbox corner overflow",
                });
            }
            ground_truth.push(LabeledBox {
                bbox: BoundingBox { x1, y1, x2, y2 },
                class_id: CLASS_ID_FACE,
            });
        }

        frames.push(WiderFaceGroundTruthFrame {
            image_id,
            ground_truth,
        });
    }

    Ok(frames)
}

fn parse_widerface_predictions(
    text: &str,
) -> Result<Vec<WiderFacePredictionFrame>, DatasetFormatError> {
    const RECORD: &str = "prediction line";
    let lines = widerface_significant_lines(text);
    let mut cursor = 0usize;
    let mut frames = Vec::new();
    let mut seen_ids = FxHashSet::default();

    while cursor < lines.len() {
        let (image_line_no, image_id_raw) = lines[cursor];
        cursor += 1;
        let image_id = image_id_raw.to_string();
        if !seen_ids.insert(image_id.clone()) {
            return Err(DatasetFormatError::DuplicateImageId {
                record: RECORD,
                index: image_line_no,
                id: image_id,
            });
        }

        let Some((count_line_no, count_raw)) = lines.get(cursor).copied() else {
            return Err(DatasetFormatError::MissingBoxCount {
                record: RECORD,
                index: image_line_no,
                image_id,
            });
        };
        cursor += 1;
        let box_count = parse_widerface_box_count(count_raw, RECORD, count_line_no)?;
        let mut predictions = Vec::with_capacity(box_count);

        for box_idx in 0..box_count {
            let Some((line_no, row)) = lines.get(cursor).copied() else {
                return Err(DatasetFormatError::MissingBoxRows {
                    record: RECORD,
                    index: image_line_no,
                    image_id,
                    expected: box_count,
                    got: box_idx,
                });
            };
            cursor += 1;
            let tokens = row.split_whitespace().collect::<Vec<_>>();
            if tokens.len() < 5 {
                return Err(DatasetFormatError::FieldCount {
                    record: RECORD,
                    index: line_no,
                    expected: "at least 5 fields `<x> <y> <w> <h> <score>`",
                });
            }

            let x1 = parse_widerface_f32(tokens[0], "x", RECORD, line_no)?;
            let y1 = parse_widerface_f32(tokens[1], "y", RECORD, line_no)?;
            let width = parse_widerface_f32(tokens[2], "w", RECORD, line_no)?;
            let height = parse_widerface_f32(tokens[3], "h", RECORD, line_no)?;
            if width <= 0.0 || height <= 0.0 {
                return Err(DatasetFormatError::InvalidBox {
                    record: RECORD,
                    index: line_no,
                    reason: "`w` and `h` must be > 0",
                });
            }
            let score = parse_widerface_non_negative_f32(tokens[4], "score", RECORD, line_no)?;

            let x2 = x1 + width;
            let y2 = y1 + height;
            if !x2.is_finite() || !y2.is_finite() {
                return Err(DatasetFormatError::InvalidBox {
                    record: RECORD,
                    index: line_no,
                    reason: "bbox corner overflow",
                });
            }
            predictions.push(Detection {
                bbox: BoundingBox { x1, y1, x2, y2 },
                score,
                class_id: CLASS_ID_FACE,
            });
        }

        frames.push(WiderFacePredictionFrame {
            image_id,
            line: image_line_no,
            predictions,
        });
    }

    Ok(frames)
}

fn widerface_significant_lines(text: &str) -> Vec<(usize, &str)> {
    text.lines()
        .enumerate()
        .filter_map(|(line_idx, raw_line)| {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                None
            } else {
                Some((line_idx + 1, line))
            }
        })
        .collect()
}

fn parse_widerface_box_count(
    token: &str,
    record: &'static str,
    line_no: usize,
) -> Result<usize, DatasetFormatError> {
    token
        .parse::<usize>()
        .map_err(|source| DatasetFormatError::ParseInt {
            record,
            index: line_no,
            field: "box count",
            value: token.to_string(),
            source,
        })
}

fn parse_widerface_f32(
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

fn parse_widerface_non_negative_f32(
    token: &str,
    field: &'static str,
    record: &'static str,
    line_no: usize,
) -> Result<f32, DatasetFormatError> {
    let value = parse_widerface_f32(token, field, record, line_no)?;
    if value < 0.0 {
        return Err(DatasetFormatError::OutOfRange {
            record,
            index: line_no,
            field,
            expected: ">= 0",
        });
    }
    Ok(value)
}

fn parse_widerface_i32(
    token: &str,
    field: &'static str,
    record: &'static str,
    line_no: usize,
) -> Result<i32, DatasetFormatError> {
    token
        .parse::<i32>()
        .map_err(|source| DatasetFormatError::ParseInt {
            record,
            index: line_no,
            field,
            value: token.to_string(),
            source,
        })
}
