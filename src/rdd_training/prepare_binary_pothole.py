from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from src.constants import PROJECT_ROOT, RDD2022_BINARY_POTHOLE_DIR, RDD2022_DIR
from src.rdd_training.constants import (
    MANIFEST_COLUMNS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    POTHOLE_LABEL,
    SPLIT_COUNTRIES,
    SUMMARY_COLUMNS,
)


def main() -> None:
    validate_split_countries(SPLIT_COUNTRIES)

    manifest_rows = build_manifest_rows(
        rdd_root=RDD2022_DIR,
        split_countries=SPLIT_COUNTRIES,
    )
    write_manifests(manifest_rows, RDD2022_BINARY_POTHOLE_DIR)

    print(f"Binary pothole manifests written to: {RDD2022_BINARY_POTHOLE_DIR}")
    print_summary(manifest_rows)


def validate_split_countries(split_countries: dict[str, tuple[str, ...]]) -> None:
    seen_countries: dict[str, str] = {}
    for split, countries in split_countries.items():
        if not countries:
            raise ValueError(f"{split} must contain at least one country.")
        for country in countries:
            if country in seen_countries:
                raise ValueError(
                    f"Country {country} appears in both {seen_countries[country]} "
                    f"and {split}."
                )
            seen_countries[country] = split


def build_manifest_rows(
    rdd_root: Path,
    split_countries: dict[str, tuple[str, ...]],
) -> list[dict[str, str | int]]:
    if not rdd_root.is_dir():
        raise FileNotFoundError(f"RDD2022 root does not exist: {rdd_root}")

    rows = []
    for split, countries in split_countries.items():
        for country in countries:
            rows.extend(parse_country(rdd_root, country, split))

    if not rows:
        raise ValueError("No RDD2022 images were found for the selected countries.")

    return rows


def parse_country(
    rdd_root: Path,
    country: str,
    split: str,
) -> list[dict[str, str | int]]:
    country_dir = rdd_root / country
    image_dir = country_dir / "train" / "images"
    annotation_dir = country_dir / "train" / "annotations" / "xmls"

    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Expected RDD2022 train images and XMLs for country: {country}"
        )

    rows = []
    for annotation_path in sorted(annotation_dir.glob("*.xml")):
        rows.append(parse_annotation(annotation_path, image_dir, country, split))

    if not rows:
        raise ValueError(f"No XML annotations found for country: {country}")

    return rows


def parse_annotation(
    annotation_path: Path,
    image_dir: Path,
    country: str,
    split: str,
) -> dict[str, str | int]:
    root = ET.parse(annotation_path).getroot()
    filename = root.findtext("filename") or f"{annotation_path.stem}.jpg"
    image_path = image_dir / filename
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image referenced by annotation does not exist: {image_path}"
        )

    labels = [
        label
        for label in (
            (obj.findtext("name") or "").strip() for obj in root.findall("object")
        )
        if label
    ]
    unique_labels = sorted(set(labels))
    num_pothole_objects = labels.count(POTHOLE_LABEL)
    has_pothole = num_pothole_objects > 0

    size = root.find("size")
    image_width = int(size.findtext("width", 0)) if size is not None else 0
    image_height = int(size.findtext("height", 0)) if size is not None else 0

    return {
        "image_path": repo_relative_path(image_path),
        "annotation_path": repo_relative_path(annotation_path),
        "country": country,
        "split": split,
        "label": POSITIVE_LABEL if has_pothole else NEGATIVE_LABEL,
        "label_name": "pothole" if has_pothole else "non_pothole",
        "has_pothole": int(has_pothole),
        "num_objects": len(labels),
        "num_pothole_objects": num_pothole_objects,
        "unique_object_labels": "|".join(unique_labels),
        "object_labels": "|".join(labels),
        "image_width": image_width,
        "image_height": image_height,
    }


def write_manifests(rows: list[dict[str, str | int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "all.csv", rows, MANIFEST_COLUMNS)

    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        write_csv(output_dir / f"{split}.csv", split_rows, MANIFEST_COLUMNS)

    summary_rows = build_summary_rows(rows)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)


def build_summary_rows(
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
            country_rows = [
                row
                for row in split_rows
                if split == "all" or row["country"] == country
            ]
            if split == "all":
                country_rows = [row for row in rows if row["country"] == country]
            summary_rows.append(make_summary_row(split, country, country_rows))
        if split != "all":
            summary_rows.append(make_summary_row(split, "__split_total__", split_rows))
    summary_rows.append(make_summary_row("all", "__dataset_total__", rows))
    return summary_rows


def make_summary_row(
    split: str,
    country: str,
    rows: list[dict[str, str | int]],
) -> dict[str, str | int | float]:
    images = len(rows)
    pothole_images = sum(int(row["has_pothole"]) for row in rows)
    objects = sum(int(row["num_objects"]) for row in rows)
    pothole_objects = sum(int(row["num_pothole_objects"]) for row in rows)
    return {
        "split": split,
        "country": country,
        "images": images,
        "pothole_images": pothole_images,
        "non_pothole_images": images - pothole_images,
        "pothole_fraction": pothole_images / images if images else 0.0,
        "objects": objects,
        "pothole_objects": pothole_objects,
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
    positive_counts = Counter(
        str(row["split"]) for row in rows if int(row["has_pothole"])
    )

    for split in ("train", "validation", "test"):
        total = split_counts[split]
        positives = positive_counts[split]
        negatives = total - positives
        fraction = positives / total if total else 0.0
        print(
            f"{split}: {total} images, {positives} pothole, "
            f"{negatives} non-pothole, pothole fraction {fraction:.3f}"
        )


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
