from src.constants import (
    CLASSIFICATION_BENCHMARK_RESULTS_CSV,
    OBJECT_DETECTION_BENCHMARK_RESULTS_CSV,
)

ALL_LOADED = "all-loaded"

CLASSIFICATION_RESULTS_CSV = CLASSIFICATION_BENCHMARK_RESULTS_CSV
OBJECT_DETECTION_RESULTS_CSV = OBJECT_DETECTION_BENCHMARK_RESULTS_CSV

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

CLASSIFICATION_RESULT_KEYS = (
    "model_name",
    "dataset_name",
    "split",
    "num_images",
)
OBJECT_DETECTION_RESULT_KEYS = (
    "model_name",
    "dataset_name",
    "split",
    "num_images",
    "confidence_threshold",
    "iou_threshold",
)
