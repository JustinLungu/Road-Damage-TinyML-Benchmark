from pathlib import Path

from src.constants import DETECTION_BENCHMARK_RESULTS_CSV
from src.detection_benchmark.constants import (
    IMAGE_CLASSIFICATION_MODELS,
    OBJECT_DETECTION_MODELS,
    SUPPORTED_MODELS,
)


ALL_LOADED = "all-loaded"
RESULTS_CSV: Path = DETECTION_BENCHMARK_RESULTS_CSV

IMAGE_CLASSIFICATION_DIR = "image_classification"
OBJECT_DETECTION_DIR = "object_detection"
COMMON_METRICS_DIR = "common_metrics"

IMAGE_CLASSIFICATION_METRICS = (
    "top1_accuracy",
    "top5_accuracy",
    "precision",
    "recall",
    "f1_score",
)
OBJECT_DETECTION_METRICS = (
    "map_50_95",
    "map_50",
    "map_75",
    "precision",
    "recall",
    "f1_score",
    "mean_iou",
)
COMMON_METRICS = (
    "precision",
    "recall",
    "f1_score",
)

TASK_COLORS = {
    IMAGE_CLASSIFICATION_DIR: "#4C78A8",
    OBJECT_DETECTION_DIR: "#F58518",
}

__all__ = [
    "ALL_LOADED",
    "COMMON_METRICS",
    "COMMON_METRICS_DIR",
    "IMAGE_CLASSIFICATION_DIR",
    "IMAGE_CLASSIFICATION_METRICS",
    "IMAGE_CLASSIFICATION_MODELS",
    "OBJECT_DETECTION_DIR",
    "OBJECT_DETECTION_METRICS",
    "OBJECT_DETECTION_MODELS",
    "RESULTS_CSV",
    "SUPPORTED_MODELS",
    "TASK_COLORS",
]
