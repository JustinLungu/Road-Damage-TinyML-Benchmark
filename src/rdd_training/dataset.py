from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from src.constants import PROJECT_ROOT
from src.rdd_training.constants import PATCH_MANIFEST_COLUMNS, POSITIVE_LABEL
from src.rdd_training.utils import (
    load_rgb_image,
    parse_manifest_int,
    parse_patch_coordinates,
    validate_manifest_columns,
    parse_manifest_row,
    validate_manifest_split,
)


ImageTransform = Callable[[Image.Image], Any]
TargetTransform = Callable[[int], Any]


@dataclass(frozen=True)
class BinaryPotholeSample:
    image_path: Path
    annotation_path: Path
    label: int
    label_name: str
    country: str
    split: str


@dataclass(frozen=True)
class BinaryPotholePatchSample(BinaryPotholeSample):
    source_object_label: str
    patch_source: str
    bbox: tuple[int, int, int, int]
    patch_box: tuple[int, int, int, int]
    image_size: tuple[int, int]


class BinaryPotholeManifest:
    """Validated rows from one binary pothole manifest CSV."""

    def __init__(
        self,
        samples: list[BinaryPotholeSample],
        manifest_path: Path,
    ) -> None:
        if not samples:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        self.samples = samples
        self.manifest_path = manifest_path

    @classmethod
    def from_csv(
        cls,
        manifest_path: Path,
        expected_split: str | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> "BinaryPotholeManifest":
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Binary pothole manifest does not exist: {manifest_path}"
            )

        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            reader = csv.DictReader(manifest_file)
            validate_manifest_columns(reader.fieldnames, manifest_path)
            samples = [
                parse_manifest_row(
                    row=row,
                    row_number=row_number,
                    manifest_path=manifest_path,
                    sample_class=BinaryPotholeSample,
                    project_root=project_root,
                    expected_split=expected_split,
                )
                for row_number, row in enumerate(reader, start=2)
            ]

        return cls(samples=samples, manifest_path=manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def class_counts(self) -> Counter[int]:
        return Counter(sample.label for sample in self.samples)

    def country_counts(self) -> Counter[str]:
        return Counter(sample.country for sample in self.samples)

    def positive_fraction(self) -> float:
        return self.class_counts()[POSITIVE_LABEL] / len(self.samples)


class BinaryPotholeDataset(Dataset):
    """PyTorch dataset for RDD2022 binary pothole image classification."""

    def __init__(
        self,
        manifest: BinaryPotholeManifest | Path,
        transform: ImageTransform | None = None,
        target_transform: TargetTransform | None = None,
        expected_split: str | None = None,
    ) -> None:
        if isinstance(manifest, Path):
            manifest = BinaryPotholeManifest.from_csv(
                manifest,
                expected_split=expected_split,
            )
        elif expected_split is not None:
            validate_manifest_split(
                manifest.samples,
                expected_split,
                manifest.manifest_path,
            )

        self.manifest = manifest
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.manifest.samples[index]
        image = load_rgb_image(sample.image_path)
        label = sample.label

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return {
            "image": image,
            "label": label,
            "image_path": sample.image_path,
            "annotation_path": sample.annotation_path,
            "label_name": sample.label_name,
            "country": sample.country,
            "split": sample.split,
        }


class BinaryPotholePatchManifest:
    """Validated rows from one binary pothole patch manifest CSV."""

    def __init__(
        self,
        samples: list[BinaryPotholePatchSample],
        manifest_path: Path,
    ) -> None:
        if not samples:
            raise ValueError(f"Patch manifest is empty: {manifest_path}")

        self.samples = samples
        self.manifest_path = manifest_path

    @classmethod
    def from_csv(
        cls,
        manifest_path: Path,
        expected_split: str | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> "BinaryPotholePatchManifest":
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Binary pothole patch manifest does not exist: {manifest_path}"
            )

        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            reader = csv.DictReader(manifest_file)
            validate_manifest_columns(reader.fieldnames, manifest_path)
            validate_patch_manifest_columns(reader.fieldnames, manifest_path)
            samples = [
                parse_patch_manifest_row(
                    row=row,
                    row_number=row_number,
                    manifest_path=manifest_path,
                    project_root=project_root,
                    expected_split=expected_split,
                )
                for row_number, row in enumerate(reader, start=2)
            ]

        return cls(samples=samples, manifest_path=manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def class_counts(self) -> Counter[int]:
        return Counter(sample.label for sample in self.samples)

    def country_counts(self) -> Counter[str]:
        return Counter(sample.country for sample in self.samples)

    def positive_fraction(self) -> float:
        return self.class_counts()[POSITIVE_LABEL] / len(self.samples)


def validate_patch_manifest_columns(
    fieldnames: list[str] | None,
    manifest_path: Path,
) -> None:
    if fieldnames is None:
        raise ValueError(f"Patch manifest has no header: {manifest_path}")

    missing_columns = set(PATCH_MANIFEST_COLUMNS) - set(fieldnames)
    if missing_columns:
        raise ValueError(
            "Patch manifest is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )


def parse_patch_manifest_row(
    row: dict[str, str],
    row_number: int,
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
    expected_split: str | None = None,
) -> BinaryPotholePatchSample:
    base_sample = parse_manifest_row(
        row=row,
        row_number=row_number,
        manifest_path=manifest_path,
        sample_class=BinaryPotholeSample,
        project_root=project_root,
        expected_split=expected_split,
    )
    patch_box = parse_patch_coordinates(row, row_number)
    source_object_label = row["source_object_label"].strip()
    patch_source = row["patch_source"].strip()
    image_size = (
        parse_manifest_int(row, "image_width", row_number),
        parse_manifest_int(row, "image_height", row_number),
    )
    if not source_object_label:
        raise ValueError(f"Missing source_object_label on row {row_number}.")
    if not patch_source:
        raise ValueError(f"Missing patch_source on row {row_number}.")
    if patch_box[2] > image_size[0] or patch_box[3] > image_size[1]:
        raise ValueError(f"Patch box exceeds image bounds on row {row_number}.")

    return BinaryPotholePatchSample(
        image_path=base_sample.image_path,
        annotation_path=base_sample.annotation_path,
        label=base_sample.label,
        label_name=base_sample.label_name,
        country=base_sample.country,
        split=base_sample.split,
        source_object_label=source_object_label,
        patch_source=patch_source,
        bbox=(
            parse_manifest_int(row, "bbox_xmin", row_number),
            parse_manifest_int(row, "bbox_ymin", row_number),
            parse_manifest_int(row, "bbox_xmax", row_number),
            parse_manifest_int(row, "bbox_ymax", row_number),
        ),
        patch_box=patch_box,
        image_size=image_size,
    )


class BinaryPotholePatchDataset(Dataset):
    """PyTorch dataset for annotation-derived RDD2022 binary pothole patches."""

    def __init__(
        self,
        manifest: BinaryPotholePatchManifest | Path,
        transform: ImageTransform | None = None,
        target_transform: TargetTransform | None = None,
        expected_split: str | None = None,
    ) -> None:
        if isinstance(manifest, Path):
            manifest = BinaryPotholePatchManifest.from_csv(
                manifest,
                expected_split=expected_split,
            )
        elif expected_split is not None:
            validate_manifest_split(
                manifest.samples,
                expected_split,
                manifest.manifest_path,
            )

        self.manifest = manifest
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.manifest.samples[index]
        image = load_rgb_image(sample.image_path)
        patch = image.crop(sample.patch_box)
        label = sample.label

        if self.transform is not None:
            patch = self.transform(patch)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return {
            "image": patch,
            "label": label,
            "image_path": sample.image_path,
            "annotation_path": sample.annotation_path,
            "label_name": sample.label_name,
            "country": sample.country,
            "split": sample.split,
            "source_object_label": sample.source_object_label,
            "patch_source": sample.patch_source,
            "bbox": sample.bbox,
            "patch_box": sample.patch_box,
            "image_size": sample.image_size,
        }
