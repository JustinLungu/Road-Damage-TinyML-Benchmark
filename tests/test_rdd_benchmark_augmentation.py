from PIL import Image
import pytest
import torch
from torchvision import transforms

from src.rdd_benchmark.data_preprocessing.augmentation import (
    make_eval_transform,
    make_minority_strong_train_transform,
    make_rdd_image_transform,
    make_standard_train_transform,
)
from src.rdd_benchmark.training import utils as training_utils


def transform_class_names(transform):
    return [type(step).__name__ for step in transform.transforms]


def test_eval_transform_is_deterministic_resize_tensor_normalize():
    transform = make_eval_transform(image_size=32)

    assert transform_class_names(transform) == ["Resize", "ToTensor", "Normalize"]


def test_standard_train_transform_uses_light_augmentation():
    transform = make_standard_train_transform(image_size=32)

    assert transform_class_names(transform) == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "ToTensor",
        "Normalize",
    ]


def test_minority_strong_train_transform_adds_extra_augmentation():
    transform = make_minority_strong_train_transform(image_size=32)

    assert transform_class_names(transform) == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomApply",
        "RandomApply",
        "RandomPerspective",
        "ToTensor",
        "Normalize",
    ]


def test_no_augmentation_train_strategy_uses_eval_transform():
    transform = make_rdd_image_transform(
        image_size=32,
        is_train=True,
        augmentation_strategy="none",
    )

    assert transform_class_names(transform) == ["Resize", "ToTensor", "Normalize"]


def test_eval_transform_ignores_training_augmentation_strategy():
    transform = make_rdd_image_transform(
        image_size=32,
        is_train=False,
        augmentation_strategy="minority_strong",
    )

    assert transform_class_names(transform) == ["Resize", "ToTensor", "Normalize"]


def test_rdd_image_transform_returns_tensor_with_expected_shape():
    image = Image.new("RGB", (48, 40), color=(128, 128, 128))
    transform = make_rdd_image_transform(
        image_size=32,
        is_train=False,
        augmentation_strategy="standard",
    )

    transformed_image = transform(image)

    assert isinstance(transformed_image, torch.Tensor)
    assert tuple(transformed_image.shape) == (3, 32, 32)


def test_rdd_image_transform_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="augmentation_strategy"):
        make_rdd_image_transform(
            image_size=32,
            is_train=True,
            augmentation_strategy="very_strong",
        )


def test_training_image_transform_forwards_model_size_and_strategy(monkeypatch):
    captured = {}

    def fake_make_rdd_image_transform(image_size, is_train, augmentation_strategy):
        captured["args"] = (image_size, is_train, augmentation_strategy)
        return transforms.Compose([])

    monkeypatch.setattr(
        training_utils,
        "make_rdd_image_transform",
        fake_make_rdd_image_transform,
    )

    transform = training_utils.make_image_transform(
        model_name="inception_v3",
        is_train=True,
        augmentation_strategy="minority_strong",
    )

    assert isinstance(transform, transforms.Compose)
    assert captured["args"] == (299, True, "minority_strong")
