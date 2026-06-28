from types import SimpleNamespace

import pytest
import torch.nn as nn

from src.rdd_training.constants import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NUM_BINARY_CLASSES,
    RDD_IMAGE_CLASSIFICATION_MODELS,
)
import src.rdd_training.utils as rdd_utils
from src.rdd_training.model_adapter import BinaryPotholeModelAdapter
from src.rdd_training.utils import (
    adapt_model_for_binary_pothole,
    load_and_adapt_all_binary_pothole_models,
)


class FakeClassifierModel:
    def __init__(self) -> None:
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(16, 1000),
        )


class FakeResNetModel:
    def __init__(self) -> None:
        self.fc = nn.Linear(32, 1000)


class FakeInceptionModel:
    def __init__(self) -> None:
        self.fc = nn.Linear(64, 1000)
        self.AuxLogits = SimpleNamespace(fc=nn.Linear(24, 1000))
        self.aux_logits = True


class FakeMobileVitModel:
    def __init__(self) -> None:
        self.classifier = nn.Linear(40, 1000)
        self.config = SimpleNamespace(
            num_labels=1000,
            id2label={0: "old"},
            label2id={"old": 0},
        )


class FakeEfficientFormerWithReset:
    def __init__(self) -> None:
        self.reset_calls = []

    def reset_classifier(self, num_classes: int) -> None:
        self.reset_calls.append(num_classes)


class FakeEfficientFormerWithHeads:
    def __init__(self) -> None:
        self.head = nn.Linear(80, 1000)
        self.head_dist = nn.Linear(80, 1000)


def assert_binary_linear(layer: nn.Linear, in_features: int) -> None:
    assert layer.in_features == in_features
    assert layer.out_features == NUM_BINARY_CLASSES


def test_supported_rdd_fine_tuning_model_set_excludes_detection_and_vlm() -> None:
    assert "mobilenet_v2" in RDD_IMAGE_CLASSIFICATION_MODELS
    assert "efficientnet_b0" in RDD_IMAGE_CLASSIFICATION_MODELS
    assert "resnet18" in RDD_IMAGE_CLASSIFICATION_MODELS
    assert "inception_v3" in RDD_IMAGE_CLASSIFICATION_MODELS
    assert "mobilevit_xxs" in RDD_IMAGE_CLASSIFICATION_MODELS
    assert "efficientformer_l1" in RDD_IMAGE_CLASSIFICATION_MODELS
    assert "yolov8n" not in RDD_IMAGE_CLASSIFICATION_MODELS
    assert "smolvlm_256m" not in RDD_IMAGE_CLASSIFICATION_MODELS


@pytest.mark.parametrize(
    "model_name",
    [
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
    ],
)
def test_adapts_classifier_sequence_models(model_name: str) -> None:
    model = FakeClassifierModel()

    adapted = adapt_model_for_binary_pothole(model_name, model)

    assert adapted is model
    assert_binary_linear(model.classifier[-1], in_features=16)


def test_adapts_resnet_fc() -> None:
    model = FakeResNetModel()

    adapt_model_for_binary_pothole("resnet18", model)

    assert_binary_linear(model.fc, in_features=32)


def test_adapts_inception_main_and_auxiliary_heads() -> None:
    model = FakeInceptionModel()

    adapt_model_for_binary_pothole("inception_v3", model)

    assert_binary_linear(model.fc, in_features=64)
    assert_binary_linear(model.AuxLogits.fc, in_features=24)
    assert model.aux_logits is False


def test_adapts_mobilevit_classifier_and_config() -> None:
    model = FakeMobileVitModel()

    adapt_model_for_binary_pothole("mobilevit_xxs", model)

    assert_binary_linear(model.classifier, in_features=40)
    assert model.config.num_labels == NUM_BINARY_CLASSES
    assert model.config.id2label == ID_TO_LABEL
    assert model.config.label2id == LABEL_TO_ID


def test_adapts_efficientformer_with_reset_classifier() -> None:
    model = FakeEfficientFormerWithReset()

    adapt_model_for_binary_pothole("efficientformer_l1", model)

    assert model.reset_calls == [NUM_BINARY_CLASSES]


def test_adapts_efficientformer_head_fallback() -> None:
    model = FakeEfficientFormerWithHeads()

    adapt_model_for_binary_pothole("efficientformer_l1", model)

    assert_binary_linear(model.head, in_features=80)
    assert_binary_linear(model.head_dist, in_features=80)


def test_rejects_unsupported_or_malformed_models() -> None:
    with pytest.raises(ValueError, match="not supported"):
        BinaryPotholeModelAdapter("yolov8n")
    with pytest.raises(ValueError, match="not supported"):
        BinaryPotholeModelAdapter("smolvlm_256m")
    with pytest.raises(ValueError, match="greater than one"):
        BinaryPotholeModelAdapter("resnet18", num_classes=1)

    malformed_classifier = SimpleNamespace(classifier=nn.Linear(8, 1000))
    with pytest.raises(ValueError, match="classifier container"):
        adapt_model_for_binary_pothole("mobilenet_v2", malformed_classifier)

    malformed_resnet = SimpleNamespace(fc=object())
    with pytest.raises(ValueError, match="nn.Linear"):
        adapt_model_for_binary_pothole("resnet18", malformed_resnet)


def test_load_and_adapt_all_models_iterates_in_stable_order(monkeypatch) -> None:
    adapted = {}

    def fake_load_and_adapt_model_for_binary_pothole(model_name: str):
        adapted[model_name] = object()
        return adapted[model_name]

    monkeypatch.setattr(
        rdd_utils,
        "load_and_adapt_model_for_binary_pothole",
        fake_load_and_adapt_model_for_binary_pothole,
    )

    result = load_and_adapt_all_binary_pothole_models(
        model_names={"resnet18", "mobilenet_v2"},
    )

    assert list(result) == ["mobilenet_v2", "resnet18"]
    assert result == adapted
