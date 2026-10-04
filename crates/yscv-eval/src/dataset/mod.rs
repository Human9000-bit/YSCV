mod coco;
mod helpers;
mod jsonl;
mod kitti;
mod mot;
mod openimages;
mod types;
mod voc;
mod widerface;
mod yolo;

use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::{DetectionDatasetFrame, EvalError, TrackingDatasetFrame};

use self::coco::build_detection_dataset_from_coco;
use self::jsonl::parse_dataset_jsonl;
use self::mot::parse_and_build_mot;
use self::openimages::parse_and_build_openimages;
use self::types::{
    CocoGroundTruthWire, CocoPredictionWire, DetectionDatasetFrameWire, TrackingDatasetFrameWire,
};
use self::widerface::parse_and_build_widerface;

pub fn load_detection_dataset_coco_files(
    ground_truth_path: &Path,
    predictions_path: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let ground_truth_text =
        fs::read_to_string(ground_truth_path).map_err(|err| EvalError::DatasetIo {
            path: ground_truth_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    let predictions_text =
        fs::read_to_string(predictions_path).map_err(|err| EvalError::DatasetIo {
            path: predictions_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    parse_detection_dataset_coco(&ground_truth_text, &predictions_text)
}

pub fn parse_detection_dataset_coco(
    ground_truth_json: &str,
    predictions_json: &str,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let ground_truth: CocoGroundTruthWire =
        serde_json::from_str(ground_truth_json).map_err(|err| EvalError::InvalidDatasetFormat {
            format: "coco-ground-truth",
            source: Arc::new(err).into(),
        })?;
    let predictions: Vec<CocoPredictionWire> =
        serde_json::from_str(predictions_json).map_err(|err| EvalError::InvalidDatasetFormat {
            format: "coco-predictions",
            source: Arc::new(err).into(),
        })?;
    build_detection_dataset_from_coco(ground_truth, predictions).map_err(|source| {
        EvalError::InvalidDatasetFormat {
            format: "coco",
            source,
        }
    })
}

pub fn load_detection_dataset_openimages_csv_files(
    ground_truth_path: &Path,
    predictions_path: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let ground_truth_text =
        fs::read_to_string(ground_truth_path).map_err(|err| EvalError::DatasetIo {
            path: ground_truth_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    let predictions_text =
        fs::read_to_string(predictions_path).map_err(|err| EvalError::DatasetIo {
            path: predictions_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    parse_detection_dataset_openimages_csv(&ground_truth_text, &predictions_text)
}

pub fn parse_detection_dataset_openimages_csv(
    ground_truth_csv: &str,
    predictions_csv: &str,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    parse_and_build_openimages(ground_truth_csv, predictions_csv).map_err(|source| {
        EvalError::InvalidDatasetFormat {
            format: "openimages",
            source,
        }
    })
}

pub fn load_detection_dataset_yolo_label_dirs(
    manifest_path: &Path,
    ground_truth_labels_dir: &Path,
    prediction_labels_dir: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let manifest_text = fs::read_to_string(manifest_path).map_err(|err| EvalError::DatasetIo {
        path: manifest_path.to_path_buf(),
        source: Arc::new(err),
    })?;
    yolo::load_yolo_label_dirs(
        &manifest_text,
        ground_truth_labels_dir,
        prediction_labels_dir,
    )
}

pub fn load_detection_dataset_voc_xml_dirs(
    manifest_path: &Path,
    ground_truth_xml_dir: &Path,
    prediction_xml_dir: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let manifest_text = fs::read_to_string(manifest_path).map_err(|err| EvalError::DatasetIo {
        path: manifest_path.to_path_buf(),
        source: Arc::new(err),
    })?;
    voc::load_voc_xml_dirs(&manifest_text, ground_truth_xml_dir, prediction_xml_dir)
}

pub fn load_detection_dataset_kitti_label_dirs(
    manifest_path: &Path,
    ground_truth_labels_dir: &Path,
    prediction_labels_dir: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let manifest_text = fs::read_to_string(manifest_path).map_err(|err| EvalError::DatasetIo {
        path: manifest_path.to_path_buf(),
        source: Arc::new(err),
    })?;
    kitti::load_kitti_label_dirs(
        &manifest_text,
        ground_truth_labels_dir,
        prediction_labels_dir,
    )
}

pub fn load_detection_dataset_widerface_files(
    ground_truth_path: &Path,
    predictions_path: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let ground_truth_text =
        fs::read_to_string(ground_truth_path).map_err(|err| EvalError::DatasetIo {
            path: ground_truth_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    let predictions_text =
        fs::read_to_string(predictions_path).map_err(|err| EvalError::DatasetIo {
            path: predictions_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    parse_detection_dataset_widerface(&ground_truth_text, &predictions_text)
}

pub fn parse_detection_dataset_widerface(
    ground_truth_text: &str,
    predictions_text: &str,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    parse_and_build_widerface(ground_truth_text, predictions_text).map_err(|source| {
        EvalError::InvalidDatasetFormat {
            format: "widerface",
            source,
        }
    })
}

pub fn load_detection_dataset_jsonl_file(
    path: &Path,
) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    let text = fs::read_to_string(path).map_err(|err| EvalError::DatasetIo {
        path: path.to_path_buf(),
        source: Arc::new(err),
    })?;
    parse_detection_dataset_jsonl(&text)
}

pub fn parse_detection_dataset_jsonl(text: &str) -> Result<Vec<DetectionDatasetFrame>, EvalError> {
    parse_dataset_jsonl(text, |line, line_no| {
        let wire = serde_json::from_str::<DetectionDatasetFrameWire>(line).map_err(|err| {
            EvalError::InvalidDatasetEntry {
                line: line_no,
                source: Arc::new(err),
            }
        })?;
        Ok(wire.into_runtime())
    })
}

pub fn load_tracking_dataset_jsonl_file(
    path: &Path,
) -> Result<Vec<TrackingDatasetFrame>, EvalError> {
    let text = fs::read_to_string(path).map_err(|err| EvalError::DatasetIo {
        path: path.to_path_buf(),
        source: Arc::new(err),
    })?;
    parse_tracking_dataset_jsonl(&text)
}

pub fn parse_tracking_dataset_jsonl(text: &str) -> Result<Vec<TrackingDatasetFrame>, EvalError> {
    parse_dataset_jsonl(text, |line, line_no| {
        let wire = serde_json::from_str::<TrackingDatasetFrameWire>(line).map_err(|err| {
            EvalError::InvalidDatasetEntry {
                line: line_no,
                source: Arc::new(err),
            }
        })?;
        Ok(wire.into_runtime())
    })
}

pub fn load_tracking_dataset_mot_txt_files(
    ground_truth_path: &Path,
    predictions_path: &Path,
) -> Result<Vec<TrackingDatasetFrame>, EvalError> {
    let ground_truth_text =
        fs::read_to_string(ground_truth_path).map_err(|err| EvalError::DatasetIo {
            path: ground_truth_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    let predictions_text =
        fs::read_to_string(predictions_path).map_err(|err| EvalError::DatasetIo {
            path: predictions_path.to_path_buf(),
            source: Arc::new(err),
        })?;
    parse_tracking_dataset_mot(&ground_truth_text, &predictions_text)
}

pub fn parse_tracking_dataset_mot(
    ground_truth_text: &str,
    predictions_text: &str,
) -> Result<Vec<TrackingDatasetFrame>, EvalError> {
    parse_and_build_mot(ground_truth_text, predictions_text).map_err(|source| {
        EvalError::InvalidDatasetFormat {
            format: "mot",
            source,
        }
    })
}
