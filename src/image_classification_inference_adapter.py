from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from src.constants import (
    CUSTOM_IMAGE_CLASSIFICATION_MODELS,
    EFFICIENTFORMER_MODEL_IDS,
    EFFICIENTNET_MODEL_CHECKPOINTS,
    IMAGE_CLASSIFICATION_MODELS,
    INCEPTION_MODEL_CHECKPOINTS,
    MOBILENET_MODEL_CHECKPOINTS,
    MOBILEVIT_MODEL_IDS,
    RESNET_MODEL_CHECKPOINTS,
    SHUFFLENET_MODEL_CHECKPOINTS,
    VIT_DIR,
)


InferenceFunction = Callable[[Path], torch.Tensor]


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def move_inputs_to_device(
    inputs: Any,
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


class ImageClassificationInferenceAdapter:
    """Prepare and run every supported image-classification model family."""

    def __init__(self, model_name: str, model: Any, device: torch.device) -> None:
        if model_name not in IMAGE_CLASSIFICATION_MODELS:
            raise ValueError(f"Model is not an image classifier: {model_name}")

        self.model_name = model_name
        self.model = model.to(device).eval()
        self.device = device
        self.transform: Any | None = None
        self.processor: Any | None = None
        self.inference_function = self._prepare_inference()

    def infer(self, image_path: Path) -> torch.Tensor:
        return self.inference_function(image_path)

    def _prepare_inference(self) -> InferenceFunction:
        if self.model_name in MOBILENET_MODEL_CHECKPOINTS:
            self.transform = self._mobilenet_weights().transforms()
        elif self.model_name in EFFICIENTNET_MODEL_CHECKPOINTS:
            from torchvision.models import EfficientNet_B0_Weights

            self.transform = EfficientNet_B0_Weights.DEFAULT.transforms()
        elif self.model_name in RESNET_MODEL_CHECKPOINTS:
            from torchvision.models import ResNet18_Weights

            self.transform = ResNet18_Weights.DEFAULT.transforms()
        elif self.model_name in SHUFFLENET_MODEL_CHECKPOINTS:
            from torchvision.models import ShuffleNet_V2_X0_5_Weights

            self.transform = ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms()
        elif self.model_name in INCEPTION_MODEL_CHECKPOINTS:
            from torchvision.models import Inception_V3_Weights

            self.transform = Inception_V3_Weights.DEFAULT.transforms()
        elif self.model_name in MOBILEVIT_MODEL_IDS:
            from transformers import AutoImageProcessor

            self.processor = AutoImageProcessor.from_pretrained(
                MOBILEVIT_MODEL_IDS[self.model_name],
                cache_dir=str(VIT_DIR / self.model_name),
                use_fast=False,
            )
            return self._infer_mobilevit
        elif self.model_name in EFFICIENTFORMER_MODEL_IDS:
            import timm

            data_config = timm.data.resolve_model_data_config(self.model)
            self.transform = timm.data.create_transform(
                **data_config,
                is_training=False,
            )
        elif self.model_name in CUSTOM_IMAGE_CLASSIFICATION_MODELS:
            from torchvision import transforms

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

        return self._infer_transformed_tensor

    def _mobilenet_weights(self) -> Any:
        from torchvision.models import (
            MobileNet_V2_Weights,
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
        )

        weights_by_model = {
            "mobilenet_v2": MobileNet_V2_Weights.DEFAULT,
            "mobilenet_v3_small": MobileNet_V3_Small_Weights.DEFAULT,
            "mobilenet_v3_large": MobileNet_V3_Large_Weights.DEFAULT,
        }
        return weights_by_model[self.model_name]

    def _infer_transformed_tensor(self, image_path: Path) -> torch.Tensor:
        image = load_rgb_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(input_tensor)

    def _infer_mobilevit(self, image_path: Path) -> Any:
        image = load_rgb_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        output = self.model(**move_inputs_to_device(inputs, self.device))
        return getattr(output, "logits", output)
