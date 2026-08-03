from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationSample:
    image_path: Path
    class_id: int


def load_classification_manifest(
    manifest_path: Path,
    num_images: int | None = None,
) -> list[ClassificationSample]:
    """Load image paths and zero-based class IDs from a CSV manifest."""

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Classification label manifest does not exist: {manifest_path}"
        )
    if num_images is not None and num_images <= 0:
        raise ValueError("num_images must be greater than zero.")

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_columns = {"image_path", "class_id"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "Classification manifest must contain image_path and class_id columns."
            )

        samples = []
        for row_number, row in enumerate(reader, start=2):
            raw_image_path = row["image_path"].strip()
            if not raw_image_path:
                raise ValueError(f"Missing image_path on manifest row {row_number}.")

            image_path = Path(raw_image_path).expanduser()
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image on manifest row {row_number} does not exist: {image_path}"
                )

            try:
                class_id = int(row["class_id"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid class_id on manifest row {row_number}: {row['class_id']}"
                ) from exc
            if class_id < 0:
                raise ValueError(
                    f"class_id must be zero or greater on manifest row {row_number}."
                )

            samples.append(
                ClassificationSample(
                    image_path=image_path.resolve(),
                    class_id=class_id,
                )
            )
            if num_images is not None and len(samples) == num_images:
                break

    if not samples:
        raise ValueError(f"Classification manifest is empty: {manifest_path}")

    return samples
