from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from src.constants import PROJECT_ROOT, RDD2022_BINARY_POTHOLE_DIR, RDD2022_DIR
from src.rdd_training.constants import (
    MANIFEST_COLUMNS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    POTHOLE_LABEL,
    RDD_AVAILABLE_COUNTRIES,
    RDD_SPLIT_FRACTIONS,
    RDD_SPLIT_MODE,
    RDD_SPLIT_RANDOM_SEED,
    RDD_SUPPORTED_SPLIT_MODES,
    SPLIT_COUNTRIES,
    SUMMARY_COLUMNS,
)


def prepare_binary_pothole_manifests() -> list[dict[str, str | int]]:
    manifest_rows = build_manifest_rows(
        rdd_root=RDD2022_DIR,
        split_mode=RDD_SPLIT_MODE,
        split_countries=SPLIT_COUNTRIES,
        countries=RDD_AVAILABLE_COUNTRIES,
        split_fractions=RDD_SPLIT_FRACTIONS,
        random_seed=RDD_SPLIT_RANDOM_SEED,
    )
    write_manifests(manifest_rows, RDD2022_BINARY_POTHOLE_DIR)

    print(f"Binary pothole manifests written to: {RDD2022_BINARY_POTHOLE_DIR}")
    print_summary(manifest_rows)
    return manifest_rows


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
    split_mode: str = "country_holdout",
    countries: tuple[str, ...] = RDD_AVAILABLE_COUNTRIES,
    split_fractions: dict[str, float] = RDD_SPLIT_FRACTIONS,
    random_seed: int = RDD_SPLIT_RANDOM_SEED,
) -> list[dict[str, str | int]]:
    if not rdd_root.is_dir():
        raise FileNotFoundError(f"RDD2022 root does not exist: {rdd_root}")

    validate_split_mode(split_mode)
    if split_mode == "stratified_by_country":
        rows = build_stratified_country_rows(
            rdd_root=rdd_root,
            countries=countries,
            split_fractions=split_fractions,
            random_seed=random_seed,
        )
        if not rows:
            raise ValueError("No RDD2022 images were found for the selected countries.")
        return rows

    validate_split_countries(split_countries)
    rows = []
    for split, countries in split_countries.items():
        for country in countries:
            rows.extend(parse_country(rdd_root, country, split))

    if not rows:
        raise ValueError("No RDD2022 images were found for the selected countries.")

    return rows


def validate_split_mode(split_mode: str) -> None:
    if split_mode not in RDD_SUPPORTED_SPLIT_MODES:
        raise ValueError(
            f"Unsupported RDD split mode: {split_mode}. "
            f"Expected one of: {', '.join(RDD_SUPPORTED_SPLIT_MODES)}."
        )


def validate_split_fractions(split_fractions: dict[str, float]) -> None:
    expected_splits = {"train", "validation", "test"}
    if set(split_fractions) != expected_splits:
        raise ValueError("RDD_SPLIT_FRACTIONS must define train, validation, and test.")
    if any(fraction <= 0 for fraction in split_fractions.values()):
        raise ValueError("RDD split fractions must be positive.")
    total = sum(split_fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("RDD split fractions must sum to 1.0.")


def build_stratified_country_rows(
    rdd_root: Path,
    countries: tuple[str, ...],
    split_fractions: dict[str, float],
    random_seed: int,
) -> list[dict[str, str | int]]:
    validate_split_fractions(split_fractions)

    rows = []
    for country in countries:
        country_rows = parse_country(rdd_root, country, split="unassigned")
        rows.extend(
            assign_country_stratified_splits(
                rows=country_rows,
                split_fractions=split_fractions,
                random_seed=random_seed,
            )
        )

    return rows


def assign_country_stratified_splits(
    rows: list[dict[str, str | int]],
    split_fractions: dict[str, float],
    random_seed: int,
) -> list[dict[str, str | int]]:
    by_label: dict[int, list[dict[str, str | int]]] = defaultdict(list)
    country = str(rows[0]["country"]) if rows else "unknown"

    for row in rows:
        by_label[int(row["label"])].append(row)

    assigned_rows = []
    for label, label_rows in sorted(by_label.items()):
        shuffled_rows = list(label_rows)
        random.Random(f"{random_seed}:{country}:{label}").shuffle(shuffled_rows)
        assigned_rows.extend(
            assign_split_to_rows(shuffled_rows, split_fractions=split_fractions)
        )

    return sorted(assigned_rows, key=lambda row: str(row["image_path"]))


def assign_split_to_rows(
    rows: list[dict[str, str | int]],
    split_fractions: dict[str, float],
) -> list[dict[str, str | int]]:
    split_counts = calculate_split_counts(len(rows), split_fractions)
    assigned_rows = []
    start_index = 0
    for split in ("train", "validation", "test"):
        end_index = start_index + split_counts[split]
        for row in rows[start_index:end_index]:
            split_row = dict(row)
            split_row["split"] = split
            assigned_rows.append(split_row)
        start_index = end_index
    return assigned_rows


def calculate_split_counts(
    total_rows: int,
    split_fractions: dict[str, float],
) -> dict[str, int]:
    validation_count = round(total_rows * split_fractions["validation"])
    test_count = round(total_rows * split_fractions["test"])
    train_count = total_rows - validation_count - test_count

    if total_rows >= 3:
        counts = {
            "train": max(1, train_count),
            "validation": max(1, validation_count),
            "test": max(1, test_count),
        }
        while sum(counts.values()) > total_rows:
            largest_split = max(counts, key=counts.get)
            counts[largest_split] -= 1
        while sum(counts.values()) < total_rows:
            counts["train"] += 1
        return counts

    validation_count = round(total_rows * split_fractions["validation"])
    test_count = round(total_rows * split_fractions["test"])
    return {
        "train": total_rows - validation_count - test_count,
        "validation": validation_count,
        "test": test_count,
    }


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
