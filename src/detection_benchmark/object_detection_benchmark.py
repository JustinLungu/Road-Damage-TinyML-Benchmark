from __future__ import annotations

import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.detection_benchmark.benchmark_result import DetectionBenchmarkResult
from src.detection_benchmark.constants import OBJECT_DETECTION_TASK
from src.detection_benchmark.object_detection_inference_adapter import (
    ObjectDetectionInferenceAdapter,
    PredictedBox,
)
from src.detection_benchmark.utils import print_progress, validate_unit_interval


@dataclass(frozen=True)
class CocoImage:
    image_id: int
    image_path: Path


@dataclass(frozen=True)
class GroundTruthBox:
    category_id: int
    xyxy: tuple[float, float, float, float]
    is_crowd: bool


@dataclass(frozen=True)
class DetectionCounts:
    true_positive: int
    false_positive: int
    false_negative: int
    matched_ious: tuple[float, ...]


class ObjectDetectionBenchmark:
    """Evaluate Ultralytics COCO detectors with COCO AP and fixed-point metrics."""

    def __init__(
        self,
        model_name: str,
        model: Any,
        images_dir: Path,
        annotation_path: Path,
        device: torch.device,
        confidence_threshold: float,
        iou_threshold: float,
        num_images: int | None = None,
        dataset_name: str = "coco",
        split: str = "validation",
    ) -> None:
        validate_unit_interval(confidence_threshold, "confidence_threshold")
        validate_unit_interval(iou_threshold, "iou_threshold")
        if num_images is not None and num_images <= 0:
            raise ValueError("num_images must be greater than zero.")

        self.model_name = model_name
        self.images_dir = images_dir
        self.annotation_path = annotation_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.dataset_name = dataset_name
        self.split = split
        self.annotation_data = load_coco_annotations(annotation_path)
        self.images = select_coco_images(
            self.annotation_data,
            images_dir,
            num_images,
        )
        self.ground_truth_by_image = group_ground_truth_boxes(
            self.annotation_data,
            {image.image_id for image in self.images},
        )
        if not any(
            not box.is_crowd
            for boxes in self.ground_truth_by_image.values()
            for box in boxes
        ):
            raise ValueError("The selected COCO images contain no evaluable boxes.")

        self.adapter = ObjectDetectionInferenceAdapter(model_name, model, device)
        self.category_id_by_class_index = build_category_mapping(
            model.names,
            self.annotation_data["categories"],
        )

    def run(self) -> DetectionBenchmarkResult:
        predictions_by_image: dict[int, list[tuple[int, PredictedBox]]] = {}
        coco_predictions = []

        with torch.inference_mode():
            for completed, image in enumerate(self.images, start=1):
                mapped_predictions = [
                    (self.category_id_by_class_index[prediction.class_index], prediction)
                    for prediction in self.adapter.predict(image.image_path)
                ]
                predictions_by_image[image.image_id] = mapped_predictions
                coco_predictions.extend(
                    prediction_to_coco(image.image_id, category_id, prediction)
                    for category_id, prediction in mapped_predictions
                )
                print_progress(completed, len(self.images))

        map_50_95, map_50, map_75 = calculate_coco_map(
            self.annotation_path,
            [image.image_id for image in self.images],
            coco_predictions,
        )
        counts = match_detections(
            self.ground_truth_by_image,
            predictions_by_image,
            self.confidence_threshold,
            self.iou_threshold,
        )
        precision = divide(
            counts.true_positive,
            counts.true_positive + counts.false_positive,
        )
        recall = divide(
            counts.true_positive,
            counts.true_positive + counts.false_negative,
        )
        f1_score = divide(2 * precision * recall, precision + recall)

        return DetectionBenchmarkResult(
            model_name=self.model_name,
            task=OBJECT_DETECTION_TASK,
            dataset_name=self.dataset_name,
            split=self.split,
            num_images=len(self.images),
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            map_50_95=map_50_95,
            map_50=map_50,
            map_75=map_75,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mean_iou=(
                sum(counts.matched_ious) / len(counts.matched_ious)
                if counts.matched_ious
                else 0.0
            ),
            top1_accuracy=None,
            top5_accuracy=None,
        )


def load_coco_annotations(annotation_path: Path) -> dict[str, Any]:
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"COCO annotation file does not exist: {annotation_path}"
        )
    with annotation_path.open(encoding="utf-8") as annotation_file:
        data = json.load(annotation_file)

    required_keys = {"images", "annotations", "categories"}
    if not required_keys.issubset(data):
        raise ValueError(
            f"COCO annotation file is missing required keys: {annotation_path}"
        )
    return data


def select_coco_images(
    annotation_data: dict[str, Any],
    images_dir: Path,
    num_images: int | None,
) -> list[CocoImage]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO images directory does not exist: {images_dir}")

    image_entries = sorted(
        annotation_data["images"],
        key=lambda image: image["file_name"],
    )
    if num_images is not None:
        image_entries = image_entries[:num_images]
    if not image_entries:
        raise ValueError("The COCO annotation file contains no images.")

    images = []
    for image_entry in image_entries:
        image_path = images_dir / image_entry["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image does not exist: {image_path}")
        images.append(
            CocoImage(
                image_id=int(image_entry["id"]),
                image_path=image_path,
            )
        )
    return images


def group_ground_truth_boxes(
    annotation_data: dict[str, Any],
    selected_image_ids: set[int],
) -> dict[int, list[GroundTruthBox]]:
    grouped: dict[int, list[GroundTruthBox]] = defaultdict(list)
    for annotation in annotation_data["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in selected_image_ids:
            continue

        x, y, width, height = (float(value) for value in annotation["bbox"])
        grouped[image_id].append(
            GroundTruthBox(
                category_id=int(annotation["category_id"]),
                xyxy=(x, y, x + width, y + height),
                is_crowd=bool(annotation.get("iscrowd", 0)),
            )
        )
    return grouped


def build_category_mapping(
    model_names: dict[int, str] | list[str],
    coco_categories: list[dict[str, Any]],
) -> dict[int, int]:
    category_id_by_name = {
        str(category["name"]): int(category["id"]) for category in coco_categories
    }
    names = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    mapping = {}
    missing_names = []
    for class_index, class_name in names:
        if class_name not in category_id_by_name:
            missing_names.append(str(class_name))
            continue
        mapping[int(class_index)] = category_id_by_name[class_name]

    if missing_names:
        raise ValueError(
            "Model classes do not match the COCO categories: "
            + ", ".join(missing_names)
        )
    return mapping


def prediction_to_coco(
    image_id: int,
    category_id: int,
    prediction: PredictedBox,
) -> dict[str, float | int | list[float]]:
    x1, y1, x2, y2 = prediction.xyxy
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "score": prediction.score,
    }


def calculate_coco_map(
    annotation_path: Path,
    image_ids: list[int],
    predictions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for COCO mAP evaluation. Run `uv sync`."
        ) from exc

    with redirect_stdout(io.StringIO()):
        coco_ground_truth = COCO(str(annotation_path))
        if predictions:
            coco_predictions = coco_ground_truth.loadRes(predictions)
        else:
            coco_predictions = COCO()
            coco_predictions.dataset = {
                "images": coco_ground_truth.dataset["images"],
                "categories": coco_ground_truth.dataset["categories"],
                "annotations": [],
            }
            coco_predictions.createIndex()

        evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return tuple(max(0.0, float(evaluator.stats[index])) for index in (0, 1, 2))


def match_detections(
    ground_truth_by_image: dict[int, list[GroundTruthBox]],
    predictions_by_image: dict[int, list[tuple[int, PredictedBox]]],
    confidence_threshold: float,
    iou_threshold: float,
) -> DetectionCounts:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    matched_ious = []

    image_ids = ground_truth_by_image.keys() | predictions_by_image.keys()
    for image_id in image_ids:
        ground_truth = ground_truth_by_image.get(image_id, [])
        regular_boxes = [box for box in ground_truth if not box.is_crowd]
        crowd_boxes = [box for box in ground_truth if box.is_crowd]
        matched_ground_truth_indices: set[int] = set()

        predictions = sorted(
            (
                (category_id, prediction)
                for category_id, prediction in predictions_by_image.get(image_id, [])
                if prediction.score >= confidence_threshold
            ),
            key=lambda item: item[1].score,
            reverse=True,
        )

        for category_id, prediction in predictions:
            candidate_ious = [
                (
                    index,
                    box_iou(prediction.xyxy, ground_truth_box.xyxy),
                )
                for index, ground_truth_box in enumerate(regular_boxes)
                if index not in matched_ground_truth_indices
                and ground_truth_box.category_id == category_id
            ]
            best_match = max(candidate_ious, key=lambda item: item[1], default=None)
            if best_match is not None and best_match[1] >= iou_threshold:
                matched_ground_truth_indices.add(best_match[0])
                true_positive += 1
                matched_ious.append(best_match[1])
                continue

            ignored_by_crowd = any(
                crowd_overlap(prediction.xyxy, crowd_box.xyxy) >= iou_threshold
                for crowd_box in crowd_boxes
                if crowd_box.category_id == category_id
            )
            if not ignored_by_crowd:
                false_positive += 1

        false_negative += len(regular_boxes) - len(matched_ground_truth_indices)

    return DetectionCounts(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        matched_ious=tuple(matched_ious),
    )


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = intersection_area(first, second)
    first_area = box_area(first)
    second_area = box_area(second)
    return divide(intersection, first_area + second_area - intersection)


def crowd_overlap(
    prediction: tuple[float, float, float, float],
    crowd: tuple[float, float, float, float],
) -> float:
    return divide(intersection_area(prediction, crowd), box_area(prediction))


def intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
