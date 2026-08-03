from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from src.constants import (
    EFFICIENTFORMER_MODEL_IDS,
    EFFICIENTNET_MODEL_CHECKPOINTS,
    INCEPTION_MODEL_CHECKPOINTS,
    MOBILENET_MODEL_CHECKPOINTS,
    MOBILEVIT_MODEL_IDS,
    RESNET_MODEL_CHECKPOINTS,
    SHUFFLENET_MODEL_CHECKPOINTS,
    VIT_DIR,
)
from src.performance_benchmark.utils import load_rgb_image, move_inputs_to_device
from src.vision_benchmark.constants import IMAGE_CLASSIFICATION_MODELS


PredictionFunction = Callable[[Path], torch.Tensor]


class ClassificationInferenceAdapter:
    """Return one logits vector for each supported image classifier."""

    def __init__(self, model_name: str, model: Any, device: torch.device) -> None:
        if model_name not in IMAGE_CLASSIFICATION_MODELS:
            raise ValueError(f"Model is not an image classifier: {model_name}")

        self.model_name = model_name
        self.model = model.to(device).eval()
        self.device = device
        self.transform: Any | None = None
        self.processor: Any | None = None
        self.prediction_function = self._prepare_prediction()

    def predict(self, image_path: Path) -> torch.Tensor:
        logits = self.prediction_function(image_path)
        if logits.ndim == 2 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 1:
            raise ValueError(
                f"Expected one logits vector, received shape {tuple(logits.shape)}."
            )
        return logits.detach().cpu()

    def _prepare_prediction(self) -> PredictionFunction:
        if self.model_name in MOBILENET_MODEL_CHECKPOINTS:
            self._prepare_mobilenet()
            return self._predict_transformed_tensor
        if self.model_name in EFFICIENTNET_MODEL_CHECKPOINTS:
            from torchvision.models import EfficientNet_B0_Weights

            self.transform = EfficientNet_B0_Weights.DEFAULT.transforms()
            return self._predict_transformed_tensor
        if self.model_name in RESNET_MODEL_CHECKPOINTS:
            from torchvision.models import ResNet18_Weights

            self.transform = ResNet18_Weights.DEFAULT.transforms()
            return self._predict_transformed_tensor
        if self.model_name in SHUFFLENET_MODEL_CHECKPOINTS:
            from torchvision.models import ShuffleNet_V2_X0_5_Weights

            self.transform = ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms()
            return self._predict_transformed_tensor
        if self.model_name in INCEPTION_MODEL_CHECKPOINTS:
            from torchvision.models import Inception_V3_Weights

            self.transform = Inception_V3_Weights.DEFAULT.transforms()
            return self._predict_transformed_tensor
        if self.model_name in MOBILEVIT_MODEL_IDS:
            from transformers import AutoImageProcessor

            self.processor = AutoImageProcessor.from_pretrained(
                MOBILEVIT_MODEL_IDS[self.model_name],
                cache_dir=str(VIT_DIR / self.model_name),
                use_fast=False,
            )
            return self._predict_mobilevit
        if self.model_name in EFFICIENTFORMER_MODEL_IDS:
            import timm

            data_config = timm.data.resolve_model_data_config(self.model)
            self.transform = timm.data.create_transform(
                **data_config,
                is_training=False,
            )
            return self._predict_transformed_tensor
        raise ValueError(
            f"No classification adapter is defined for model: {self.model_name}"
        )

    def _prepare_mobilenet(self) -> None:
        from torchvision.models import (
            MobileNet_V2_Weights,
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
        )

        if self.model_name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT
        elif self.model_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = MobileNet_V3_Large_Weights.DEFAULT
        self.transform = weights.transforms()

    def _predict_transformed_tensor(self, image_path: Path) -> torch.Tensor:
        if self.transform is None:
            raise RuntimeError("Image transform has not been initialized.")

        image = load_rgb_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(input_tensor)

    def _predict_mobilevit(self, image_path: Path) -> torch.Tensor:
        if self.processor is None:
            raise RuntimeError("Image processor has not been initialized.")

        image = load_rgb_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        output = self.model(**move_inputs_to_device(inputs, self.device))
        return output.logits
