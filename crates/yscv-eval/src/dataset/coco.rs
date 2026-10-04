use rustc_hash::{FxBuildHasher, FxHashMap};

use yscv_detect::{BoundingBox, Detection};

use crate::{DatasetFormatError, DetectionDatasetFrame, LabeledBox};

use super::types::{CocoGroundTruthWire, CocoPredictionWire};

pub(crate) fn build_detection_dataset_from_coco(
    ground_truth: CocoGroundTruthWire,
    predictions: Vec<CocoPredictionWire>,
) -> Result<Vec<DetectionDatasetFrame>, DatasetFormatError> {
    let mut frames = Vec::with_capacity(ground_truth.images.len());
    let mut image_index_by_id =
        FxHashMap::with_capacity_and_hasher(ground_truth.images.len(), FxBuildHasher);

    for (frame_idx, image) in ground_truth.images.into_iter().enumerate() {
        if image_index_by_id.insert(image.id, frame_idx).is_some() {
            return Err(DatasetFormatError::DuplicateImageId {
                record: "image",
                index: frame_idx,
                id: image.id.to_string(),
            });
        }
        frames.push(DetectionDatasetFrame {
            ground_truth: Vec::new(),
            predictions: Vec::new(),
        });
    }

    for (annotation_idx, annotation) in ground_truth.annotations.into_iter().enumerate() {
        let Some(&frame_idx) = image_index_by_id.get(&annotation.image_id) else {
            return Err(DatasetFormatError::UnknownImageId {
                record: "annotation",
                index: annotation_idx,
                id: annotation.image_id.to_string(),
            });
        };
        let class_id =
            coco_category_id_to_class_id(annotation.category_id, "annotation", annotation_idx)?;
        let bbox = coco_bbox_to_runtime(annotation.bbox, "annotation", annotation_idx)?;
        frames[frame_idx]
            .ground_truth
            .push(LabeledBox { bbox, class_id });
    }

    for (prediction_idx, prediction) in predictions.into_iter().enumerate() {
        let Some(&frame_idx) = image_index_by_id.get(&prediction.image_id) else {
            return Err(DatasetFormatError::UnknownImageId {
                record: "prediction",
                index: prediction_idx,
                id: prediction.image_id.to_string(),
            });
        };
        if !prediction.score.is_finite() {
            return Err(DatasetFormatError::OutOfRange {
                record: "prediction",
                index: prediction_idx,
                field: "score",
                expected: "finite",
            });
        }
        let class_id =
            coco_category_id_to_class_id(prediction.category_id, "prediction", prediction_idx)?;
        let bbox = coco_bbox_to_runtime(prediction.bbox, "prediction", prediction_idx)?;
        frames[frame_idx].predictions.push(Detection {
            bbox,
            score: prediction.score,
            class_id,
        });
    }

    Ok(frames)
}

fn coco_category_id_to_class_id(
    category_id: u64,
    record: &'static str,
    index: usize,
) -> Result<usize, DatasetFormatError> {
    usize::try_from(category_id).map_err(|source| DatasetFormatError::IntOverflow {
        record,
        index,
        field: "category_id",
        value: category_id,
        source,
    })
}

fn coco_bbox_to_runtime(
    bbox: [f32; 4],
    record: &'static str,
    index: usize,
) -> Result<BoundingBox, DatasetFormatError> {
    let [x, y, width, height] = bbox;
    if !x.is_finite() || !y.is_finite() || !width.is_finite() || !height.is_finite() {
        return Err(DatasetFormatError::OutOfRange {
            record,
            index,
            field: "bbox",
            expected: "finite",
        });
    }
    if width < 0.0 || height < 0.0 {
        return Err(DatasetFormatError::InvalidBox {
            record,
            index,
            reason: "negative bbox size",
        });
    }

    let x2 = x + width;
    let y2 = y + height;
    if !x2.is_finite() || !y2.is_finite() {
        return Err(DatasetFormatError::InvalidBox {
            record,
            index,
            reason: "bbox corner overflow",
        });
    }
    Ok(BoundingBox {
        x1: x,
        y1: y,
        x2,
        y2,
    })
}
