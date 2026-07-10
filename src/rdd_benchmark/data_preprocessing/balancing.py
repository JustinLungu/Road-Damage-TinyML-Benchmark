from __future__ import annotations

import csv
import random
from pathlib import Path

from src.constants import (
    RDD2022_BINARY_POTHOLE_DIR,
    RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
)
from src.rdd_benchmark.constants import NEGATIVE_LABEL, POSITIVE_LABEL
from src.rdd_benchmark.data_loader.constants import MANIFEST_COLUMNS
from src.rdd_benchmark.data_preprocessing.constants import RDD_SPLIT_RANDOM_SEED
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole import write_manifests


def build_balanced_manifest_output_dir(
    experiment_name: str,
    output_root: Path = RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
) -> Path:
    if not experiment_name:
        raise ValueError("experiment_name cannot be empty.")
    return output_root / experiment_name


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = set(MANIFEST_COLUMNS) - fieldnames
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        return list(reader)


def validate_non_potholes_per_pothole(non_potholes_per_pothole: int | None) -> int:
    if non_potholes_per_pothole is None:
        raise ValueError("A non-pothole per pothole ratio is required.")
    if non_potholes_per_pothole < 1:
        raise ValueError("non-potholes per pothole ratio must be at least 1.")
    return non_potholes_per_pothole


def downsample_majority_rows(
    rows: list[dict[str, str]],
    non_potholes_per_pothole: int | None,
    random_seed: int = RDD_SPLIT_RANDOM_SEED,
) -> list[dict[str, str]]:
    ratio = validate_non_potholes_per_pothole(non_potholes_per_pothole)
    pothole_rows = [
        row for row in rows if int(row["label"]) == POSITIVE_LABEL
    ]
    non_pothole_rows = [
        row for row in rows if int(row["label"]) == NEGATIVE_LABEL
    ]
    if not pothole_rows:
        raise ValueError("Cannot downsample majority class without pothole rows.")
    if not non_pothole_rows:
        raise ValueError("Cannot downsample majority class without non-pothole rows.")

    target_non_pothole_count = min(
        len(non_pothole_rows),
        len(pothole_rows) * ratio,
    )
    rng = random.Random(f"{random_seed}:majority_downsample:{ratio}")
    selected_non_pothole_rows = rng.sample(
        non_pothole_rows,
        k=target_non_pothole_count,
    )
    balanced_rows = [*pothole_rows, *selected_non_pothole_rows]
    rng.shuffle(balanced_rows)
    return balanced_rows


def prepare_balanced_binary_pothole_manifests(
    experiment_name: str,
    non_potholes_per_pothole: int | None,
    source_dir: Path = RDD2022_BINARY_POTHOLE_DIR,
    output_root: Path = RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
    random_seed: int = RDD_SPLIT_RANDOM_SEED,
) -> Path:
    train_rows = load_manifest_rows(source_dir / "train.csv")
    validation_rows = load_manifest_rows(source_dir / "validation.csv")
    test_rows = load_manifest_rows(source_dir / "test.csv")

    balanced_train_rows = downsample_majority_rows(
        train_rows,
        non_potholes_per_pothole,
        random_seed=random_seed,
    )
    output_dir = build_balanced_manifest_output_dir(
        experiment_name,
        output_root=output_root,
    )
    write_manifests(
        [*balanced_train_rows, *validation_rows, *test_rows],
        output_dir,
    )
    return output_dir
