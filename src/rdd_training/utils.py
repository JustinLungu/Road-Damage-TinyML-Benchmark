from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch.nn as nn
from PIL import Image

from src.constants import PROJECT_ROOT
from src.rdd_training.constants import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RDD_IMAGE_CLASSIFICATION_MODELS,
    REQUIRED_MANIFEST_COLUMNS,
)


########### Manifest Validation ###########


def validate_manifest_columns(
    fieldnames: list[str] | None,
    manifest_path: Path,
) -> None:
    if fieldnames is None:
        raise ValueError(f"Manifest has no header: {manifest_path}")

    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(fieldnames)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {', '.join(sorted(missing_columns))}"
        )


def parse_binary_label(raw_label: str, row_number: int) -> int:
    try:
        label = int(raw_label)
    except ValueError as exc:
        raise ValueError(f"Invalid label on row {row_number}: {raw_label}") from exc

    if label not in {NEGATIVE_LABEL, POSITIVE_LABEL}:
        raise ValueError(
            f"Binary pothole label must be {NEGATIVE_LABEL} or {POSITIVE_LABEL} "
            f"on row {row_number}."
        )
    return label


########### Manifest Paths ###########


def expected_label_name(label: int) -> str:
    return "pothole" if label == POSITIVE_LABEL else "non_pothole"


def resolve_manifest_path(
    raw_path: str,
    project_root: Path,
    manifest_path: Path,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    project_relative_path = project_root / path
    if project_relative_path.exists():
        return project_relative_path

    return manifest_path.parent / path


def validate_existing_file(path: Path, column_name: str, row_number: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{column_name} on row {row_number} does not exist: {path}"
        )


########### Images ###########


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


########### Manifest Rows ###########


def parse_manifest_row(
    row: dict[str, str],
    row_number: int,
    manifest_path: Path,
    sample_class,
    project_root: Path = PROJECT_ROOT,
    expected_split: str | None = None,
):
    split = row["split"].strip()
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"Expected split {expected_split}, got {split} on row {row_number}."
        )

    label = parse_binary_label(row["label"], row_number)
    label_name = row["label_name"].strip()
    expected_name = expected_label_name(label)
    if label_name != expected_name:
        raise ValueError(
            f"label_name must be {expected_name} for label {label} "
            f"on row {row_number}."
        )

    image_path = resolve_manifest_path(row["image_path"], project_root, manifest_path)
    annotation_path = resolve_manifest_path(
        row["annotation_path"],
        project_root,
        manifest_path,
    )
    validate_existing_file(image_path, "image_path", row_number)
    validate_existing_file(annotation_path, "annotation_path", row_number)

    country = row["country"].strip()
    if not country:
        raise ValueError(f"Missing country on row {row_number}.")

    return sample_class(
        image_path=image_path,
        annotation_path=annotation_path,
        label=label,
        label_name=label_name,
        country=country,
        split=split,
    )


def validate_manifest_split(
    samples,
    expected_split: str,
    manifest_path: Path,
) -> None:
    bad_splits = sorted(
        {sample.split for sample in samples if sample.split != expected_split}
    )
    if bad_splits:
        raise ValueError(
            f"Manifest {manifest_path} contains split values other than "
            f"{expected_split}: {', '.join(bad_splits)}"
        )


########### Model Adaptation ###########


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


def adapt_model_for_binary_pothole(model_name: str, model: Any) -> Any:
    from src.rdd_training.model_adapter import BinaryPotholeModelAdapter

    return BinaryPotholeModelAdapter(model_name).adapt(model)


def load_and_adapt_model_for_binary_pothole(model_name: str) -> Any:
    from src.load_model import load_model

    model = load_model(model_name)
    return adapt_model_for_binary_pothole(model_name, model)


def load_and_adapt_all_binary_pothole_models(
    model_names: Iterable[str] = RDD_IMAGE_CLASSIFICATION_MODELS,
) -> dict[str, Any]:
    adapted_models = {}

    for model_name in sorted(model_names):
        adapted_models[model_name] = load_and_adapt_model_for_binary_pothole(model_name)

    return adapted_models
