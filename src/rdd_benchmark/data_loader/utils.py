from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.constants import PROJECT_ROOT
from src.rdd_benchmark.constants import NEGATIVE_LABEL, POSITIVE_LABEL
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


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        validate_manifest_columns(reader.fieldnames, manifest_path)
        return list(reader)


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
    return (
        project_relative_path
        if project_relative_path.exists()
        else manifest_path.parent / path
    )


def parse_manifest_row(
    row: dict[str, str],
    row_number: int,
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
    expected_split: str | None = None,
) -> dict[str, Any]:
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
            f"label_name must be {expected_name} for label {label} on row {row_number}."
        )

    image_path = resolve_manifest_path(row["image_path"], project_root, manifest_path)
    annotation_path = resolve_manifest_path(
        row["annotation_path"], project_root, manifest_path
    )
    for path, column_name in (
        (image_path, "image_path"),
        (annotation_path, "annotation_path"),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{column_name} on row {row_number} does not exist: {path}"
            )

    country = row["country"].strip()
    if not country:
        raise ValueError(f"Missing country on row {row_number}.")

    return {
        "image_path": image_path,
        "annotation_path": annotation_path,
        "label": label,
        "label_name": label_name,
        "country": country,
        "split": split,
    }


def validate_manifest_split(samples, expected_split: str, manifest_path: Path) -> None:
    bad_splits = sorted(
        {sample.split for sample in samples if sample.split != expected_split}
    )
    if bad_splits:
        raise ValueError(
            f"Manifest {manifest_path} contains split values other than "
            f"{expected_split}: {', '.join(bad_splits)}"
        )


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def collate_binary_pothole_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.as_tensor(
            [int(item["label"]) for item in batch], dtype=torch.long
        ),
    }
