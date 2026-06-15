from dataclasses import dataclass


@dataclass
class DetectionBenchmarkResult:
    """One task-aware row in the detection benchmark results table."""

    model_name: str
    task: str
    dataset_name: str
    split: str
    num_images: int
    confidence_threshold: float | None
    iou_threshold: float | None
    map_50_95: float | None
    map_50: float | None
    map_75: float | None
    precision: float
    recall: float
    f1_score: float
    mean_iou: float | None
    top1_accuracy: float | None
    top5_accuracy: float | None
