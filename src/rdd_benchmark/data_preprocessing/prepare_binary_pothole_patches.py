from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.constants import (
    PROJECT_ROOT,
    RDD2022_BINARY_POTHOLE_PATCH_DIR,
    RDD2022_DIR,
)
from src.rdd_benchmark.constants import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    POTHOLE_LABEL,
    RDD_SPLIT_MODE,
)
from src.rdd_benchmark.data_loader.constants import PATCH_MANIFEST_COLUMNS
from src.rdd_benchmark.data_preprocessing.constants import (
    PATCH_SUMMARY_COLUMNS,
    RDD_AVAILABLE_COUNTRIES,
    RDD_MIN_BOX_AREA,
    RDD_PATCH_PADDING,
    RDD_SPLIT_FRACTIONS,
    RDD_SPLIT_RANDOM_SEED,
    SPLIT_COUNTRIES,
)
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole import (
    build_manifest_rows,
    validate_split_countries,
)


@dataclass(frozen=True)
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class RDDObjectAnnotation:
    label: str
    bbox: BoundingBox


@dataclass(frozen=True)
class RDDImageAnnotation:
    image_path: Path
    annotation_path: Path
    country: str
    split: str
    width: int
    height: int
    objects: tuple[RDDObjectAnnotation, ...]


class BinaryPotholePatchManifestPreprocessor:
    def __init__(
        self,
        rdd_root: Path,
        output_dir: Path,
        split_countries: dict[str, tuple[str, ...]],
        split_mode: str = "country_holdout",
        countries: tuple[str, ...] = RDD_AVAILABLE_COUNTRIES,
        split_fractions: dict[str, float] = RDD_SPLIT_FRACTIONS,
        random_seed: int = RDD_SPLIT_RANDOM_SEED,
        patch_padding: float = RDD_PATCH_PADDING,
        min_box_area: int = RDD_MIN_BOX_AREA,
    ) -> None:
        self.rdd_root = rdd_root
        self.output_dir = output_dir
        self.split_countries = split_countries
        self.split_mode = split_mode
        self.countries = countries
        self.split_fractions = split_fractions
        self.random_seed = random_seed
        self.patch_padding = patch_padding
        self.min_box_area = min_box_area

    def build_rows(self) -> list[dict[str, str | int]]:
        if not self.rdd_root.is_dir():
            raise FileNotFoundError(f"RDD2022 root does not exist: {self.rdd_root}")

        if self.split_mode == "stratified_by_country":
            return self._build_rows_from_full_image_split()

        rows = []
        for split, countries in self.split_countries.items():
            for country in countries:
                rows.extend(self._parse_country(country, split))

        if not rows:
            raise ValueError("No RDD2022 patch rows were created.")

        return rows

    def _build_rows_from_full_image_split(self) -> list[dict[str, str | int]]:
        full_image_rows = build_manifest_rows(
            rdd_root=self.rdd_root,
            split_mode=self.split_mode,
            split_countries=self.split_countries,
            countries=self.countries,
            split_fractions=self.split_fractions,
            random_seed=self.random_seed,
        )

        patch_rows = []
        for full_image_row in full_image_rows:
            image_annotation = parse_image_annotation(
                annotation_path=repo_path(str(full_image_row["annotation_path"])),
                image_dir=repo_path(str(full_image_row["image_path"])).parent,
                country=str(full_image_row["country"]),
                split=str(full_image_row["split"]),
            )
            patch_rows.extend(
                make_patch_rows(
                    image_annotation=image_annotation,
                    patch_padding=self.patch_padding,
                    min_box_area=self.min_box_area,
                )
            )

        if not patch_rows:
            raise ValueError("No RDD2022 patch rows were created.")

        return patch_rows

    def write_manifests(self, rows: list[dict[str, str | int]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(self.output_dir / "all.csv", rows, PATCH_MANIFEST_COLUMNS)

        for split in ("train", "validation", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            write_csv(self.output_dir / f"{split}.csv", split_rows, PATCH_MANIFEST_COLUMNS)

        summary_rows = build_patch_summary_rows(rows)
        write_csv(self.output_dir / "summary.csv", summary_rows, PATCH_SUMMARY_COLUMNS)

    def _parse_country(self, country: str, split: str) -> list[dict[str, str | int]]:
        country_dir = self.rdd_root / country
        image_dir = country_dir / "train" / "images"
        annotation_dir = country_dir / "train" / "annotations" / "xmls"

        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Expected RDD2022 train images and XMLs for country: {country}"
            )

        rows = []
        for annotation_path in sorted(annotation_dir.glob("*.xml")):
            image_annotation = parse_image_annotation(
                annotation_path=annotation_path,
                image_dir=image_dir,
                country=country,
                split=split,
            )
            rows.extend(
                make_patch_rows(
                    image_annotation=image_annotation,
                    patch_padding=self.patch_padding,
                    min_box_area=self.min_box_area,
                )
            )

        if not rows:
            raise ValueError(f"No valid patch rows found for country: {country}")

        return rows


def parse_image_annotation(
    annotation_path: Path,
    image_dir: Path,
    country: str,
    split: str,
) -> RDDImageAnnotation:
    root = ET.parse(annotation_path).getroot()
    filename = root.findtext("filename") or f"{annotation_path.stem}.jpg"
    image_path = image_dir / filename
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image referenced by annotation does not exist: {image_path}"
        )

    size = root.find("size")
    image_width = int(size.findtext("width", 0)) if size is not None else 0
    image_height = int(size.findtext("height", 0)) if size is not None else 0

    objects = tuple(
        object_annotation
        for raw_object in root.findall("object")
        if (object_annotation := parse_object_annotation(raw_object)) is not None
    )

    return RDDImageAnnotation(
        image_path=image_path,
        annotation_path=annotation_path,
        country=country,
        split=split,
        width=image_width,
        height=image_height,
        objects=objects,
    )


def parse_object_annotation(raw_object: ET.Element) -> RDDObjectAnnotation | None:
    label = (raw_object.findtext("name") or "").strip()
    box = raw_object.find("bndbox")
    if not label or box is None:
        return None

    return RDDObjectAnnotation(
        label=label,
        bbox=BoundingBox(
            xmin=int(float(box.findtext("xmin", 0))),
            ymin=int(float(box.findtext("ymin", 0))),
            xmax=int(float(box.findtext("xmax", 0))),
            ymax=int(float(box.findtext("ymax", 0))),
        ),
    )


def make_patch_rows(
    image_annotation: RDDImageAnnotation,
    patch_padding: float,
    min_box_area: int,
) -> list[dict[str, str | int]]:
    rows = []
    for object_annotation in image_annotation.objects:
        if object_annotation.bbox.area < min_box_area:
            continue

        label = (
            POSITIVE_LABEL
            if object_annotation.label == POTHOLE_LABEL
            else NEGATIVE_LABEL
        )
        patch_box = make_padded_square_patch_box(
            bbox=object_annotation.bbox,
            image_width=image_annotation.width,
            image_height=image_annotation.height,
            padding=patch_padding,
        )
        rows.append(
            make_patch_row(
                image_annotation=image_annotation,
                object_annotation=object_annotation,
                patch_box=patch_box,
                label=label,
            )
        )

    return rows


def make_padded_square_patch_box(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
    padding: float,
) -> BoundingBox:
    padded_size = max(bbox.width, bbox.height) * (1.0 + 2.0 * padding)
    side_length = max(1, int(round(padded_size)))
    side_length = min(side_length, image_width, image_height)

    center_x = (bbox.xmin + bbox.xmax) / 2
    center_y = (bbox.ymin + bbox.ymax) / 2
    xmin = int(round(center_x - side_length / 2))
    ymin = int(round(center_y - side_length / 2))
    xmin = min(max(0, xmin), max(0, image_width - side_length))
    ymin = min(max(0, ymin), max(0, image_height - side_length))

    return BoundingBox(
        xmin=xmin,
        ymin=ymin,
        xmax=xmin + side_length,
        ymax=ymin + side_length,
    )


def make_patch_row(
    image_annotation: RDDImageAnnotation,
    object_annotation: RDDObjectAnnotation,
    patch_box: BoundingBox,
    label: int,
) -> dict[str, str | int]:
    return {
        "image_path": repo_relative_path(image_annotation.image_path),
        "annotation_path": repo_relative_path(image_annotation.annotation_path),
        "country": image_annotation.country,
        "split": image_annotation.split,
        "label": label,
        "label_name": "pothole" if label == POSITIVE_LABEL else "non_pothole",
        "source_object_label": object_annotation.label,
        "patch_source": "annotation_box",
        "bbox_xmin": object_annotation.bbox.xmin,
        "bbox_ymin": object_annotation.bbox.ymin,
        "bbox_xmax": object_annotation.bbox.xmax,
        "bbox_ymax": object_annotation.bbox.ymax,
        "patch_xmin": patch_box.xmin,
        "patch_ymin": patch_box.ymin,
        "patch_xmax": patch_box.xmax,
        "patch_ymax": patch_box.ymax,
        "image_width": image_annotation.width,
        "image_height": image_annotation.height,
    }


def build_patch_summary_rows(
    rows: list[dict[str, str | int]],
) -> list[dict[str, str | int | float]]:
    summary_rows = []
    for split in ("train", "validation", "test", "all"):
        split_rows = (
            rows
            if split == "all"
            else [row for row in rows if row["split"] == split]
        )
        countries = sorted({str(row["country"]) for row in split_rows})
        for country in countries:
            country_rows = [row for row in split_rows if row["country"] == country]
            summary_rows.append(make_patch_summary_row(split, country, country_rows))
        if split != "all":
            summary_rows.append(make_patch_summary_row(split, "__split_total__", split_rows))

    summary_rows.append(make_patch_summary_row("all", "__dataset_total__", rows))
    return summary_rows


def make_patch_summary_row(
    split: str,
    country: str,
    rows: list[dict[str, str | int]],
) -> dict[str, str | int | float]:
    patches = len(rows)
    pothole_patches = sum(int(row["label"]) == POSITIVE_LABEL for row in rows)
    return {
        "split": split,
        "country": country,
        "patches": patches,
        "pothole_patches": pothole_patches,
        "non_pothole_patches": patches - pothole_patches,
        "pothole_fraction": pothole_patches / patches if patches else 0.0,
    }


def write_csv(
    output_path: Path,
    rows: list[dict[str, str | int | float]],
    fieldnames: list[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int]]) -> None:
    split_counts = Counter(str(row["split"]) for row in rows)
    positive_counts = Counter(str(row["split"]) for row in rows if int(row["label"]))

    for split in ("train", "validation", "test"):
        total = split_counts[split]
        positives = positive_counts[split]
        negatives = total - positives
        fraction = positives / total if total else 0.0
        print(
            f"{split}: {total} patches, {positives} pothole, "
            f"{negatives} non-pothole, pothole fraction {fraction:.3f}"
        )


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def repo_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def prepare_binary_pothole_patch_manifests() -> list[dict[str, str | int]]:
    if RDD_SPLIT_MODE == "country_holdout":
        validate_split_countries(SPLIT_COUNTRIES)
    preprocessor = BinaryPotholePatchManifestPreprocessor(
        rdd_root=RDD2022_DIR,
        output_dir=RDD2022_BINARY_POTHOLE_PATCH_DIR,
        split_countries=SPLIT_COUNTRIES,
        split_mode=RDD_SPLIT_MODE,
        countries=RDD_AVAILABLE_COUNTRIES,
        split_fractions=RDD_SPLIT_FRACTIONS,
        random_seed=RDD_SPLIT_RANDOM_SEED,
    )
    rows = preprocessor.build_rows()
    preprocessor.write_manifests(rows)

    print(f"Binary pothole patch manifests written to: {RDD2022_BINARY_POTHOLE_PATCH_DIR}")
    print_summary(rows)
    return rows
