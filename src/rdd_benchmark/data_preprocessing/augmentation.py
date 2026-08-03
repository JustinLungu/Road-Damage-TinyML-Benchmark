from __future__ import annotations

from torchvision import transforms

from src.rdd_benchmark.data_preprocessing.constants import (
    RDD_AUGMENTATION_NONE,
    RDD_AUGMENTATION_STANDARD,
    RDD_AUGMENTATION_STRONG,
    RDD_SUPPORTED_AUGMENTATION_STRATEGIES,
)


IMAGENET_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZE_STD = (0.229, 0.224, 0.225)


def make_eval_transform(image_size: int):
    """Deterministic transform used for validation, testing, and no augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZE_MEAN,
                std=IMAGENET_NORMALIZE_STD,
            ),
        ]
    )


def make_standard_train_transform(image_size: int):
    """Light augmentation for natural full-image baseline training."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZE_MEAN,
                std=IMAGENET_NORMALIZE_STD,
            ),
        ]
    )


def make_strong_train_transform(image_size: int):
    """Stronger road-safe augmentation applied to every training image."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.60, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.20,
                        hue=0.04,
                    )
                ],
                p=0.60,
            ),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=8)],
                p=0.35,
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZE_MEAN,
                std=IMAGENET_NORMALIZE_STD,
            ),
        ]
    )


def make_rdd_image_transform(
    image_size: int,
    is_train: bool,
    augmentation_strategy: str = RDD_AUGMENTATION_STANDARD,
):
    if augmentation_strategy not in RDD_SUPPORTED_AUGMENTATION_STRATEGIES:
        raise ValueError(
            "augmentation_strategy must be one of: "
            f"{', '.join(RDD_SUPPORTED_AUGMENTATION_STRATEGIES)}."
        )
    if not is_train or augmentation_strategy == RDD_AUGMENTATION_NONE:
        return make_eval_transform(image_size)
    if augmentation_strategy == RDD_AUGMENTATION_STANDARD:
        return make_standard_train_transform(image_size)
    if augmentation_strategy == RDD_AUGMENTATION_STRONG:
        return make_strong_train_transform(image_size)

    raise ValueError(f"Unsupported augmentation_strategy: {augmentation_strategy}.")
