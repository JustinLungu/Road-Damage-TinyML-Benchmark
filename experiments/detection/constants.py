from pathlib import Path

from src.constants import DETECTION_BENCHMARK_RESULTS_CSV
from src.detection_benchmark.constants import (
    IMAGE_CLASSIFICATION_MODELS,
    OBJECT_DETECTION_MODELS,
    SUPPORTED_MODELS,
)


ALL_LOADED = "all-loaded"
RESULTS_CSV: Path = DETECTION_BENCHMARK_RESULTS_CSV

__all__ = [
    "ALL_LOADED",
    "IMAGE_CLASSIFICATION_MODELS",
    "OBJECT_DETECTION_MODELS",
    "RESULTS_CSV",
    "SUPPORTED_MODELS",
]
