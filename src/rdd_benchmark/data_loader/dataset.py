from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from src.constants import PROJECT_ROOT
from src.rdd_benchmark.constants import POSITIVE_LABEL
from src.rdd_benchmark.data_loader.utils import (
    load_rgb_image,
    parse_manifest_row,
    validate_manifest_columns,
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


class BinaryPotholeManifest:
    """Validated rows from one binary pothole manifest CSV."""

    def __init__(self, samples: list[BinaryPotholeSample], manifest_path: Path) -> None:
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
                    row,
                    row_number,
                    manifest_path,
                    BinaryPotholeSample,
                    project_root,
                    expected_split,
                )
                for row_number, row in enumerate(reader, start=2)
            ]
        return cls(samples, manifest_path)

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
            manifest = BinaryPotholeManifest.from_csv(manifest, expected_split)
        elif expected_split is not None:
            validate_manifest_split(
                manifest.samples, expected_split, manifest.manifest_path
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
