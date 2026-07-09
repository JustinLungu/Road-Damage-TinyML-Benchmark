from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.constants import (
    PROJECT_ROOT,
    RDD2022_BINARY_POTHOLE_DIR,
    RDD2022_BINARY_POTHOLE_PATCH_DIR,
)
from src.rdd_benchmark.constants import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RDD_SUPPORTED_TRAINING_INPUT_MODES,
    RDD_TRAINING_INPUT_MODE,
)
from src.rdd_benchmark.data_loader.constants import REQUIRED_MANIFEST_COLUMNS


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


def parse_manifest_int(
    row: dict[str, str],
    column_name: str,
    row_number: int,
) -> int:
    try:
        return int(row[column_name])
    except KeyError as exc:
        raise ValueError(
            f"Manifest row {row_number} is missing column: {column_name}"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer for {column_name} on row {row_number}: "
            f"{row.get(column_name, '')}"
        ) from exc


def parse_patch_coordinates(
    row: dict[str, str],
    row_number: int,
) -> tuple[int, int, int, int]:
    patch_xmin = parse_manifest_int(row, "patch_xmin", row_number)
    patch_ymin = parse_manifest_int(row, "patch_ymin", row_number)
    patch_xmax = parse_manifest_int(row, "patch_xmax", row_number)
    patch_ymax = parse_manifest_int(row, "patch_ymax", row_number)

    if patch_xmin < 0 or patch_ymin < 0:
        raise ValueError(f"Patch coordinates must be non-negative on row {row_number}.")
    if patch_xmax <= patch_xmin or patch_ymax <= patch_ymin:
        raise ValueError(f"Patch box has invalid bounds on row {row_number}.")

    return patch_xmin, patch_ymin, patch_xmax, patch_ymax


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


def select_rdd_manifest_path(
    split: str,
    input_mode: str | None = None,
) -> Path:
    mode = RDD_TRAINING_INPUT_MODE if input_mode is None else input_mode
    if mode not in RDD_SUPPORTED_TRAINING_INPUT_MODES:
        raise ValueError(
            "RDD training input mode must be one of: "
            f"{', '.join(RDD_SUPPORTED_TRAINING_INPUT_MODES)}."
        )
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported RDD split: {split}")

    if mode == "full_image":
        return RDD2022_BINARY_POTHOLE_DIR / f"{split}.csv"
    if mode == "annotation_patch":
        return RDD2022_BINARY_POTHOLE_PATCH_DIR / f"{split}.csv"

    raise ValueError(f"Unsupported RDD training input mode: {mode}")


def make_rdd_dataset(
    split: str,
    input_mode: str | None = None,
    transform=None,
    target_transform=None,
):
    from src.rdd_benchmark.data_loader.dataset import (
        BinaryPotholeDataset,
        BinaryPotholePatchDataset,
    )

    mode = RDD_TRAINING_INPUT_MODE if input_mode is None else input_mode
    manifest_path = select_rdd_manifest_path(split=split, input_mode=mode)

    if mode == "full_image":
        return BinaryPotholeDataset(
            manifest_path,
            transform=transform,
            target_transform=target_transform,
            expected_split=split,
        )
    if mode == "annotation_patch":
        return BinaryPotholePatchDataset(
            manifest_path,
            transform=transform,
            target_transform=target_transform,
            expected_split=split,
        )

    raise ValueError(f"Unsupported RDD training input mode: {mode}")


def make_rdd_training_dataset(
    input_mode: str | None = None,
    transform=None,
    target_transform=None,
):
    return make_rdd_dataset(
        split="train",
        input_mode=input_mode,
        transform=transform,
        target_transform=target_transform,
    )


def make_rdd_validation_dataset(
    input_mode: str | None = None,
    transform=None,
    target_transform=None,
):
    return make_rdd_dataset(
        split="validation",
        input_mode=input_mode,
        transform=transform,
        target_transform=target_transform,
    )


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def collate_binary_pothole_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.as_tensor(
            [int(item["label"]) for item in batch],
            dtype=torch.long,
        ),
        "image_path": [item["image_path"] for item in batch],
        "label_name": [item["label_name"] for item in batch],
        "country": [item["country"] for item in batch],
        "split": [item["split"] for item in batch],
    }
