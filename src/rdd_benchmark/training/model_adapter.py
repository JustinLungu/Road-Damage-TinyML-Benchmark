from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.constants import (
    CUSTOM_IMAGE_CLASSIFICATION_MODELS,
    EFFICIENTFORMER_MODEL_IDS,
    EFFICIENTNET_MODEL_CHECKPOINTS,
    INCEPTION_MODEL_CHECKPOINTS,
    MOBILENET_MODEL_CHECKPOINTS,
    MOBILEVIT_MODEL_IDS,
    RESNET_MODEL_CHECKPOINTS,
    SHUFFLENET_MODEL_CHECKPOINTS,
)
from src.rdd_benchmark.constants import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NUM_BINARY_CLASSES,
)
from src.rdd_benchmark.training.constants import RDD_IMAGE_CLASSIFICATION_MODELS


def require_attribute(model: Any, attribute_name: str) -> Any:
    if not hasattr(model, attribute_name):
        raise ValueError(f"Model has no {attribute_name} attribute.")
    return getattr(model, attribute_name)


def require_linear_attribute(model: Any, attribute_name: str) -> nn.Linear:
    layer = require_attribute(model, attribute_name)
    if not isinstance(layer, nn.Linear):
        raise ValueError(f"Expected {attribute_name} to be nn.Linear.")
    return layer


def make_replacement_linear(layer: nn.Linear, num_classes: int) -> nn.Linear:
    return nn.Linear(
        in_features=layer.in_features,
        out_features=num_classes,
        bias=layer.bias is not None,
    )


def replace_linear_attribute(
    model: Any,
    attribute_name: str,
    num_classes: int,
) -> None:
    layer = require_linear_attribute(model, attribute_name)
    setattr(model, attribute_name, make_replacement_linear(layer, num_classes))


def replace_last_linear(module: Any, num_classes: int) -> None:
    if isinstance(module, nn.Linear):
        raise ValueError("Expected a classifier container, received nn.Linear.")
    if not hasattr(module, "__len__") or not hasattr(module, "__getitem__"):
        raise ValueError("Classifier does not support indexed layer replacement.")

    for index in reversed(range(len(module))):
        layer = module[index]
        if isinstance(layer, nn.Linear):
            module[index] = make_replacement_linear(layer, num_classes)
            return

    raise ValueError("Classifier contains no nn.Linear layer to replace.")


def update_hugging_face_label_config(model: Any, num_classes: int) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return

    config.num_labels = num_classes
    config.id2label = dict(ID_TO_LABEL)
    config.label2id = dict(LABEL_TO_ID)


class BinaryPotholeModelAdapter:
    """Replace pretrained image-classification heads with a binary RDD head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_BINARY_CLASSES,
    ) -> None:
        if model_name not in RDD_IMAGE_CLASSIFICATION_MODELS:
            raise ValueError(
                f"Model is not supported for RDD fine-tuning: {model_name}"
            )
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        self.model_name = model_name
        self.num_classes = num_classes

    def adapt(self, model: Any) -> Any:
        if self.model_name in (
            MOBILENET_MODEL_CHECKPOINTS.keys() | EFFICIENTNET_MODEL_CHECKPOINTS.keys()
        ):
            return self._adapt_classifier_sequence(model)
        if self.model_name in RESNET_MODEL_CHECKPOINTS:
            return self._adapt_linear_attribute(model, "fc")
        if self.model_name in SHUFFLENET_MODEL_CHECKPOINTS:
            return self._adapt_linear_attribute(model, "fc")
        if self.model_name in INCEPTION_MODEL_CHECKPOINTS:
            return self._adapt_inception(model)
        if self.model_name in MOBILEVIT_MODEL_IDS:
            return self._adapt_mobilevit(model)
        if self.model_name in EFFICIENTFORMER_MODEL_IDS:
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


def adapt_model_for_binary_pothole(model_name: str, model: Any) -> Any:
    return BinaryPotholeModelAdapter(model_name).adapt(model)


def load_and_adapt_model_for_binary_pothole(model_name: str) -> Any:
    from src.load_model import load_model

    return adapt_model_for_binary_pothole(model_name, load_model(model_name))
