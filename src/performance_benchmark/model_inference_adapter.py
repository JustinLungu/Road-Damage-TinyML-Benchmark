from pathlib import Path
from typing import Any, Callable

import torch

from src.constants import VIT_DIR, VLM_DIR
from src.performance_benchmark.constants import (
    CUSTOM_IMAGE_CLASSIFICATION_MODELS,
    EFFICIENTNET_MODELS,
    EFFICIENTFORMER_MODELS,
    INCEPTION_MODELS,
    MOBILEVIT_MODELS,
    MOBILENET_MODELS,
    RESNET_MODELS,
    SHUFFLENET_MODELS,
    SMOLVLM_MODELS,
    SMOLVLM_PROMPT,
    YOLO_MODELS,
)
from src.performance_benchmark.utils import load_rgb_image, move_inputs_to_device


InferenceFunction = Callable[[Path], None]


class ModelInferenceAdapter:
    """Normalize different model APIs behind one image-path inference call."""

    def __init__(self, model_name: str, model: Any, device: torch.device) -> None:
        self.model_name = model_name
        self.model = model
        self.device = device
        self.transform: Any | None = None
        self.processor: Any | None = None
        self.prompt: str | None = None
        self.inference_function = self._prepare_inference()

    def infer(self, image_path: Path) -> None:
        self.inference_function(image_path)

    def _prepare_inference(self) -> InferenceFunction:
        # Each model family needs different preprocessing and call signatures.
        if self.model_name in YOLO_MODELS:
            return self._infer_yolo
        if self.model_name in MOBILENET_MODELS:
            return self._prepare_mobilenet()
        if self.model_name in EFFICIENTNET_MODELS:
            return self._prepare_efficientnet()
        if self.model_name in RESNET_MODELS:
            return self._prepare_resnet()
        if self.model_name in SHUFFLENET_MODELS:
            return self._prepare_shufflenet()
        if self.model_name in INCEPTION_MODELS:
            return self._prepare_inception()
        if self.model_name in MOBILEVIT_MODELS:
            return self._prepare_mobilevit()
        if self.model_name in EFFICIENTFORMER_MODELS:
            return self._prepare_efficientformer()
        if self.model_name in CUSTOM_IMAGE_CLASSIFICATION_MODELS:
            return self._prepare_custom_image_classifier()
        if self.model_name in SMOLVLM_MODELS:
            return self._prepare_smolvlm()

        raise ValueError(
            f"No inference adapter is defined for model: {self.model_name}"
        )

    def _infer_yolo(self, image_path: Path) -> None:
        # Ultralytics handles image loading and preprocessing from the file path.
        self.model.predict(
            source=str(image_path), device=str(self.device), verbose=False
        )

    def _prepare_mobilenet(self) -> InferenceFunction:
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
        self.model.to(self.device).eval()
        return self._infer_transformed_tensor

    def _prepare_efficientnet(self) -> InferenceFunction:
        from torchvision.models import EfficientNet_B0_Weights

        self.transform = EfficientNet_B0_Weights.DEFAULT.transforms()
        self.model.to(self.device).eval()
        return self._infer_transformed_tensor

    def _prepare_resnet(self) -> InferenceFunction:
        from torchvision.models import ResNet18_Weights

        self.transform = ResNet18_Weights.DEFAULT.transforms()
        self.model.to(self.device).eval()
        return self._infer_transformed_tensor

    def _prepare_shufflenet(self) -> InferenceFunction:
        from torchvision.models import ShuffleNet_V2_X0_5_Weights

        self.transform = ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms()
        self.model.to(self.device).eval()
        return self._infer_transformed_tensor

    def _prepare_inception(self) -> InferenceFunction:
        from torchvision.models import Inception_V3_Weights

        self.transform = Inception_V3_Weights.DEFAULT.transforms()
        self.model.to(self.device).eval()
        return self._infer_transformed_tensor

    def _prepare_mobilevit(self) -> InferenceFunction:
        from transformers import AutoImageProcessor

        model_id = MOBILEVIT_MODELS[self.model_name]
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=str(VIT_DIR / self.model_name),
            use_fast=False,
        )
        self.model.to(self.device).eval()
        return self._infer_mobilevit

    def _prepare_efficientformer(self) -> InferenceFunction:
        import timm

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.model.to(self.device).eval()
        return self._infer_transformed_tensor

    def _prepare_custom_image_classifier(self) -> InferenceFunction:
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
        self.model.to(self.device).eval()
        return self._infer_transformed_tensor

    def _prepare_smolvlm(self) -> InferenceFunction:
        from transformers import AutoProcessor

        model_id = SMOLVLM_MODELS[self.model_name]
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(VLM_DIR / self.model_name),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SMOLVLM_PROMPT},
                ],
            }
        ]
        # The benchmark measures one forward pass with a fixed prompt, no decoding.
        self.prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        self.model.to(self.device).eval()
        return self._infer_smolvlm

    def _infer_transformed_tensor(self, image_path: Path) -> None:
        if self.transform is None:
            raise RuntimeError("Image transform has not been initialized.")

        # Torchvision and timm models consume a batched image tensor.
        image = load_rgb_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        self.model(input_tensor)

    def _infer_mobilevit(self, image_path: Path) -> None:
        if self.processor is None:
            raise RuntimeError("Image processor has not been initialized.")

        # Hugging Face processors return dict-like tensors accepted by the model.
        image = load_rgb_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        self.model(**move_inputs_to_device(inputs, self.device))

    def _infer_smolvlm(self, image_path: Path) -> None:
        if self.processor is None or self.prompt is None:
            raise RuntimeError("SmolVLM processor has not been initialized.")

        image = load_rgb_image(image_path)
        inputs = self.processor(text=self.prompt, images=[image], return_tensors="pt")
        self.model(**move_inputs_to_device(inputs, self.device))
