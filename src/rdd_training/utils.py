from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.constants import PROJECT_ROOT
from src.rdd_training.constants import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    REQUIRED_MANIFEST_COLUMNS,
)


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


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


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
