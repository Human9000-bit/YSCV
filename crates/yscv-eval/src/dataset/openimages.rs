use std::sync::Arc;

use rustc_hash::FxHashMap;

use csv::StringRecord;
use yscv_detect::{BoundingBox, Detection};

use crate::{DatasetFormatError, DetectionDatasetFrame, LabeledBox};

use super::helpers::ensure_detection_frame_index;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpenImagesColumns {
    image_id: usize,
    label_name: usize,
    x_min: usize,
    x_max: usize,
    y_min: usize,
    y_max: usize,
}

impl OpenImagesColumns {
    fn from_headers(headers: &StringRecord) -> Result<Self, DatasetFormatError> {
        let required = |header: &'static str| {
            find_openimages_header(headers, header)
                .ok_or(DatasetFormatError::MissingHeader { header })
        };
        Ok(Self {
            image_id: required("ImageID")?,
            label_name: required("LabelName")?,
            x_min: required("XMin")?,
            x_max: required("XMax")?,
            y_min: required("YMin")?,
            y_max: required("YMax")?,
        })
    }
}

fn find_openimages_header(headers: &StringRecord, name: &str) -> Option<usize> {
    headers
        .iter()
        .position(|header| header.trim().eq_ignore_ascii_case(name))
}

#[derive(Debug, Clone, PartialEq)]
struct OpenImagesGroundTruthEntry {
    image_id: String,
    label_name: String,
    bbox: BoundingBox,
}

#[derive(Debug, Clone, PartialEq)]
struct OpenImagesPredictionEntry {
    image_id: String,
    label_name: String,
    bbox: BoundingBox,
    score: f32,
}

pub(crate) fn parse_and_build_openimages(
    ground_truth_csv: &str,
    predictions_csv: &str,
) -> Result<Vec<DetectionDatasetFrame>, DatasetFormatError> {
    let ground_truth = parse_openimages_ground_truth(ground_truth_csv)?;
    let predictions = parse_openimages_predictions(predictions_csv)?;

    let mut frames = Vec::new();
    let mut image_index_by_id = FxHashMap::default();
    let mut class_id_by_label = FxHashMap::default();

    for entry in ground_truth {
        let frame_idx = ensure_detection_frame_index(
            &mut frames,
            &mut image_index_by_id,
            entry.image_id.as_str(),
        );
        let class_id =
            resolve_openimages_class_id(entry.label_name.as_str(), &mut class_id_by_label);
        frames[frame_idx].ground_truth.push(LabeledBox {
            bbox: entry.bbox,
            class_id,
        });
    }

    for entry in predictions {
        let frame_idx = ensure_detection_frame_index(
            &mut frames,
            &mut image_index_by_id,
            entry.image_id.as_str(),
        );
        let class_id =
            resolve_openimages_class_id(entry.label_name.as_str(), &mut class_id_by_label);
        frames[frame_idx].predictions.push(Detection {
            bbox: entry.bbox,
            score: entry.score,
            class_id,
        });
    }

    Ok(frames)
}

fn resolve_openimages_class_id(
    label_name: &str,
    class_id_by_label: &mut FxHashMap<String, usize>,
) -> usize {
    if let Some(class_id) = class_id_by_label.get(label_name) {
        return *class_id;
    }
    let class_id = class_id_by_label.len();
    class_id_by_label.insert(label_name.to_string(), class_id);
    class_id
}

fn parse_openimages_ground_truth(
    csv_text: &str,
) -> Result<Vec<OpenImagesGroundTruthEntry>, DatasetFormatError> {
    const RECORD: &str = "ground-truth row";
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|err| DatasetFormatError::Csv {
            record: RECORD,
            index: 1,
            source: Arc::new(err),
        })?
        .clone();
    let columns = OpenImagesColumns::from_headers(&headers)?;

    let mut rows = Vec::new();
    for (row_idx, record_result) in reader.records().enumerate() {
        let row_no = row_idx + 2;
        let record = record_result.map_err(|err| DatasetFormatError::Csv {
            record: RECORD,
            index: row_no,
            source: Arc::new(err),
        })?;
        let image_id =
            parse_openimages_non_empty_field(&record, columns.image_id, "ImageID", RECORD, row_no)?;
        let label_name = parse_openimages_non_empty_field(
            &record,
            columns.label_name,
            "LabelName",
            RECORD,
            row_no,
        )?;
        let bbox = parse_openimages_bbox(&record, &columns, RECORD, row_no)?;
        rows.push(OpenImagesGroundTruthEntry {
            image_id: image_id.to_string(),
            label_name: label_name.to_string(),
            bbox,
        });
    }
    Ok(rows)
}

fn parse_openimages_predictions(
    csv_text: &str,
) -> Result<Vec<OpenImagesPredictionEntry>, DatasetFormatError> {
    const RECORD: &str = "prediction row";
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|err| DatasetFormatError::Csv {
            record: RECORD,
            index: 1,
            source: Arc::new(err),
        })?
        .clone();
    let columns = OpenImagesColumns::from_headers(&headers)?;
    let score_idx = find_openimages_header(&headers, "Score")
        .or_else(|| find_openimages_header(&headers, "Confidence"))
        .or_else(|| find_openimages_header(&headers, "Conf"))
        .ok_or(DatasetFormatError::MissingHeader {
            header: "Score/Confidence",
        })?;

    let mut rows = Vec::new();
    for (row_idx, record_result) in reader.records().enumerate() {
        let row_no = row_idx + 2;
        let record = record_result.map_err(|err| DatasetFormatError::Csv {
            record: RECORD,
            index: row_no,
            source: Arc::new(err),
        })?;
        let image_id =
            parse_openimages_non_empty_field(&record, columns.image_id, "ImageID", RECORD, row_no)?;
        let label_name = parse_openimages_non_empty_field(
            &record,
            columns.label_name,
            "LabelName",
            RECORD,
            row_no,
        )?;
        let bbox = parse_openimages_bbox(&record, &columns, RECORD, row_no)?;
        let score_raw = parse_openimages_non_empty_field(
            &record,
            score_idx,
            "Score/Confidence",
            RECORD,
            row_no,
        )?;
        let score = parse_openimages_unit_f32(score_raw, "Score/Confidence", RECORD, row_no)?;

        rows.push(OpenImagesPredictionEntry {
            image_id: image_id.to_string(),
            label_name: label_name.to_string(),
            bbox,
            score,
        });
    }
    Ok(rows)
}

fn parse_openimages_non_empty_field<'a>(
    record: &'a StringRecord,
    column: usize,
    field: &'static str,
    at: &'static str,
    row_no: usize,
) -> Result<&'a str, DatasetFormatError> {
    match record.get(column).map(str::trim) {
        Some(value) if !value.is_empty() => Ok(value),
        _ => Err(DatasetFormatError::MissingField {
            record: at,
            index: row_no,
            field,
        }),
    }
}

fn parse_openimages_bbox(
    record: &StringRecord,
    columns: &OpenImagesColumns,
    at: &'static str,
    row_no: usize,
) -> Result<BoundingBox, DatasetFormatError> {
    let x1_raw = parse_openimages_non_empty_field(record, columns.x_min, "XMin", at, row_no)?;
    let x2_raw = parse_openimages_non_empty_field(record, columns.x_max, "XMax", at, row_no)?;
    let y1_raw = parse_openimages_non_empty_field(record, columns.y_min, "YMin", at, row_no)?;
    let y2_raw = parse_openimages_non_empty_field(record, columns.y_max, "YMax", at, row_no)?;
    let x1 = parse_openimages_unit_f32(x1_raw, "XMin", at, row_no)?;
    let x2 = parse_openimages_unit_f32(x2_raw, "XMax", at, row_no)?;
    let y1 = parse_openimages_unit_f32(y1_raw, "YMin", at, row_no)?;
    let y2 = parse_openimages_unit_f32(y2_raw, "YMax", at, row_no)?;
    if x2 <= x1 || y2 <= y1 {
        return Err(DatasetFormatError::InvalidBox {
            record: at,
            index: row_no,
            reason: "expected XMax>XMin and YMax>YMin",
        });
    }
    Ok(BoundingBox { x1, y1, x2, y2 })
}

fn parse_openimages_unit_f32(
    token: &str,
    field: &'static str,
    record: &'static str,
    row_no: usize,
) -> Result<f32, DatasetFormatError> {
    let value = token
        .parse::<f32>()
        .map_err(|source| DatasetFormatError::ParseFloat {
            record,
            index: row_no,
            field,
            value: token.to_string(),
            source,
        })?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(DatasetFormatError::OutOfRange {
            record,
            index: row_no,
            field,
            expected: "finite and in [0, 1]",
        });
    }
    Ok(value)
}
