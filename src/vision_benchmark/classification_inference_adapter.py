from pathlib import Path
from typing import Any

import torch

from src.image_classification_inference_adapter import (
    ImageClassificationInferenceAdapter,
)
from src.vision_benchmark.constants import IMAGE_CLASSIFICATION_MODELS


class ClassificationInferenceAdapter(ImageClassificationInferenceAdapter):
    """Return one CPU logits vector from a pretrained image classifier."""

    def __init__(self, model_name: str, model: Any, device: torch.device) -> None:
        if model_name not in IMAGE_CLASSIFICATION_MODELS:
            raise ValueError(f"Model is not an image classifier: {model_name}")
        super().__init__(model_name, model, device)

    def predict(self, image_path: Path) -> torch.Tensor:
        logits = self.infer(image_path)
        if logits.ndim == 2 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 1:
            raise ValueError(
                f"Expected one logits vector, received shape {tuple(logits.shape)}."
            )
        return logits.detach().cpu()
