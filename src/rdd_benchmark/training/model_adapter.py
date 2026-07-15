from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.performance_benchmark.constants import (
    CUSTOM_IMAGE_CLASSIFICATION_MODELS,
    EFFICIENTFORMER_MODELS,
    EFFICIENTNET_MODELS,
    INCEPTION_MODELS,
    MOBILEVIT_MODELS,
    MOBILENET_MODELS,
    RESNET_MODELS,
    SHUFFLENET_MODELS,
)
from src.rdd_benchmark.constants import (
    NUM_BINARY_CLASSES,
)
from src.rdd_benchmark.training.constants import RDD_IMAGE_CLASSIFICATION_MODELS
from src.rdd_benchmark.training.utils import (
    make_replacement_linear,
    replace_last_linear,
    replace_linear_attribute,
    require_attribute,
    require_linear_attribute,
    update_hugging_face_label_config,
)


class BinaryPotholeModelAdapter:
    """Replace pretrained image-classification heads with a binary RDD head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_BINARY_CLASSES,
    ) -> None:
        if model_name not in RDD_IMAGE_CLASSIFICATION_MODELS:
            raise ValueError(f"Model is not supported for RDD fine-tuning: {model_name}")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        self.model_name = model_name
        self.num_classes = num_classes

    def adapt(self, model: Any) -> Any:
        if self.model_name in MOBILENET_MODELS | EFFICIENTNET_MODELS:
            return self._adapt_classifier_sequence(model)
        if self.model_name in RESNET_MODELS:
            return self._adapt_linear_attribute(model, "fc")
        if self.model_name in SHUFFLENET_MODELS:
            return self._adapt_linear_attribute(model, "fc")
        if self.model_name in INCEPTION_MODELS:
            return self._adapt_inception(model)
        if self.model_name in MOBILEVIT_MODELS:
            return self._adapt_mobilevit(model)
        if self.model_name in EFFICIENTFORMER_MODELS:
            return self._adapt_efficientformer(model)
        if self.model_name in CUSTOM_IMAGE_CLASSIFICATION_MODELS:
            if hasattr(model, "fc"):
                return self._adapt_linear_attribute(model, "fc")
            return self._adapt_linear_attribute(model, "classifier")

        raise ValueError(f"No RDD model adapter is defined for: {self.model_name}")

    def _adapt_classifier_sequence(self, model: Any) -> Any:
        classifier = require_attribute(model, "classifier")
        replace_last_linear(classifier, self.num_classes)
        return model

    def _adapt_linear_attribute(self, model: Any, attribute_name: str) -> Any:
        linear_layer = require_linear_attribute(model, attribute_name)
        setattr(
            model,
            attribute_name,
            make_replacement_linear(linear_layer, self.num_classes),
        )
        return model

    def _adapt_inception(self, model: Any) -> Any:
        self._adapt_linear_attribute(model, "fc")

        aux_logits = getattr(model, "AuxLogits", None)
        if aux_logits is not None and hasattr(aux_logits, "fc"):
            replace_linear_attribute(aux_logits, "fc", self.num_classes)

        # Keep downstream training output handling simple: one logits tensor.
        if hasattr(model, "aux_logits"):
            model.aux_logits = False
        if hasattr(model, "AuxLogits"):
            model.AuxLogits = None

        return model

    def _adapt_mobilevit(self, model: Any) -> Any:
        self._adapt_linear_attribute(model, "classifier")
        update_hugging_face_label_config(model, self.num_classes)
        return model

    def _adapt_efficientformer(self, model: Any) -> Any:
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes=self.num_classes)
            return model

        if hasattr(model, "head"):
            replace_linear_attribute(model, "head", self.num_classes)
        else:
            raise ValueError(
                f"EfficientFormer model {self.model_name} has no reset_classifier "
                "method or head attribute."
            )

        if hasattr(model, "head_dist") and isinstance(model.head_dist, nn.Linear):
            replace_linear_attribute(model, "head_dist", self.num_classes)

        return model
