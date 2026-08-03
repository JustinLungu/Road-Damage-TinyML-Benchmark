from dataclasses import dataclass


@dataclass
class ClassificationBenchmarkResult:
    """Metrics from one image-classification benchmark run."""

    model_name: str
    dataset_name: str
    split: str
    num_images: int
    precision: float
    recall: float
    f1_score: float
    top1_accuracy: float
    top5_accuracy: float


@dataclass
class ObjectDetectionBenchmarkResult:
    """Metrics from one object-detection benchmark run."""

    model_name: str
    dataset_name: str
    split: str
    num_images: int
    confidence_threshold: float
    iou_threshold: float
    map_50_95: float
    map_50: float
    map_75: float
    precision: float
    recall: float
    f1_score: float
    mean_iou: float
