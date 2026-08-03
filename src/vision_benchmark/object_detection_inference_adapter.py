from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.vision_benchmark.constants import (
    OBJECT_DETECTION_MODELS,
    PREDICTION_CONFIDENCE_FLOOR,
)


@dataclass(frozen=True)
class PredictedBox:
    class_index: int
    score: float
    xyxy: tuple[float, float, float, float]


class ObjectDetectionInferenceAdapter:
    """Normalize Ultralytics predictions into scored xyxy boxes."""

    def __init__(self, model_name: str, model: Any, device: torch.device) -> None:
        if model_name not in OBJECT_DETECTION_MODELS:
            raise ValueError(f"Model is not an object detector: {model_name}")

        self.model = model
        self.device = device

    def predict(self, image_path: Path) -> list[PredictedBox]:
        result = self.model.predict(
            source=str(image_path),
            device=str(self.device),
            conf=PREDICTION_CONFIDENCE_FLOOR,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        xyxy_values = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        class_indices = result.boxes.cls.detach().cpu().tolist()
        return [
            PredictedBox(
                class_index=int(class_index),
                score=float(score),
                xyxy=tuple(float(value) for value in xyxy),
            )
            for xyxy, score, class_index in zip(
                xyxy_values,
                scores,
                class_indices,
            )
        ]
