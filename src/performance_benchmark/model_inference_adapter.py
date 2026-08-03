from pathlib import Path
from typing import Any, Callable

import torch

from src.constants import (
    IMAGE_CLASSIFICATION_MODELS,
    OBJECT_DETECTION_MODELS,
    SMOLVLM_MODEL_IDS,
    VLM_DIR,
    VISION_LANGUAGE_MODELS,
)
from src.image_classification_inference_adapter import (
    ImageClassificationInferenceAdapter,
)
from src.performance_benchmark.constants import SMOLVLM_PROMPT
from src.performance_benchmark.utils import load_rgb_image, move_inputs_to_device


InferenceFunction = Callable[[Path], None]


class ModelInferenceAdapter:
    """Normalize model APIs behind one image-path inference call."""

    def __init__(self, model_name: str, model: Any, device: torch.device) -> None:
        self.model_name = model_name
        self.model = model
        self.device = device
        self.classification_adapter: ImageClassificationInferenceAdapter | None = None
        self.processor: Any | None = None
        self.prompt: str | None = None
        self.inference_function = self._prepare_inference()

    def infer(self, image_path: Path) -> None:
        self.inference_function(image_path)

    def _prepare_inference(self) -> InferenceFunction:
        if self.model_name in OBJECT_DETECTION_MODELS:
            return self._infer_yolo
        if self.model_name in IMAGE_CLASSIFICATION_MODELS:
            self.classification_adapter = ImageClassificationInferenceAdapter(
                self.model_name,
                self.model,
                self.device,
            )
            return self._infer_image_classifier
        if self.model_name in VISION_LANGUAGE_MODELS:
            return self._prepare_smolvlm()
        raise ValueError(
            f"No inference adapter is defined for model: {self.model_name}"
        )

    def _infer_yolo(self, image_path: Path) -> None:
        self.model.predict(
            source=str(image_path),
            device=str(self.device),
            verbose=False,
        )

    def _infer_image_classifier(self, image_path: Path) -> None:
        if self.classification_adapter is None:
            raise RuntimeError("Classification adapter has not been initialized.")
        self.classification_adapter.infer(image_path)

    def _prepare_smolvlm(self) -> InferenceFunction:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            SMOLVLM_MODEL_IDS[self.model_name],
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
        self.prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        self.model.to(self.device).eval()
        return self._infer_smolvlm

    def _infer_smolvlm(self, image_path: Path) -> None:
        if self.processor is None or self.prompt is None:
            raise RuntimeError("SmolVLM processor has not been initialized.")

        image = load_rgb_image(image_path)
        inputs = self.processor(text=self.prompt, images=[image], return_tensors="pt")
        self.model(**move_inputs_to_device(inputs, self.device))
