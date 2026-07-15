from __future__ import annotations

import random
from pathlib import Path

from src.constants import (
    RDD2022_BINARY_POTHOLE_DIR,
    RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
    RDD2022_SYNTHETIC_POTHOLE_DIR,
)
from src.rdd_benchmark.constants import POSITIVE_LABEL
from src.rdd_benchmark.data_loader.constants import MANIFEST_COLUMNS
from src.rdd_benchmark.data_loader.utils import load_manifest_rows
from src.rdd_benchmark.data_preprocessing.constants import RDD_SPLIT_RANDOM_SEED
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole import write_manifests


SYNTHETIC_MANIFEST_NAME = "manifest.csv"


def resolve_synthetic_manifest_path(
    synthetic_source_dir: Path = RDD2022_SYNTHETIC_POTHOLE_DIR,
) -> Path:
    manifest_path = synthetic_source_dir / SYNTHETIC_MANIFEST_NAME
    if not synthetic_source_dir.is_dir():
        raise FileNotFoundError(
            "Synthetic pothole source folder is missing. Expected folder: "
            f"{synthetic_source_dir}. Add generated pothole images and "
            f"{SYNTHETIC_MANIFEST_NAME} before running Experiment E."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Synthetic pothole manifest is missing. Expected file: "
            f"{manifest_path}. The manifest must use the binary pothole CSV columns."
        )
    return manifest_path


def load_synthetic_pothole_rows(
    synthetic_source_dir: Path = RDD2022_SYNTHETIC_POTHOLE_DIR,
) -> list[dict[str, str]]:
    manifest_path = resolve_synthetic_manifest_path(synthetic_source_dir)
    rows = load_manifest_rows(manifest_path)
    synthetic_rows = [
        ensure_synthetic_train_pothole_row(row)
        for row in rows
        if int(row["label"]) == POSITIVE_LABEL
    ]
    if not synthetic_rows:
        raise ValueError(f"Synthetic manifest has no pothole rows: {manifest_path}")
    return synthetic_rows


def ensure_synthetic_train_pothole_row(row: dict[str, str]) -> dict[str, str]:
    synthetic_row = {column: row.get(column, "") for column in MANIFEST_COLUMNS}
    synthetic_row["split"] = "train"
    synthetic_row["label"] = str(POSITIVE_LABEL)
    synthetic_row["label_name"] = "pothole"
    synthetic_row["has_pothole"] = "1"
    if not synthetic_row["num_pothole_objects"]:
        synthetic_row["num_pothole_objects"] = "1"
    return synthetic_row


def calculate_synthetic_pothole_limit(
    train_rows: list[dict[str, str]],
    synthetic_pothole_ratio: float,
) -> int:
    if synthetic_pothole_ratio <= 0:
        raise ValueError("synthetic_pothole_ratio must be greater than 0.")

    real_pothole_count = sum(
        int(row["label"]) == POSITIVE_LABEL for row in train_rows
    )
    if real_pothole_count == 0:
        raise ValueError("Cannot add synthetic potholes without real pothole rows.")
    return max(1, round(real_pothole_count * synthetic_pothole_ratio))


def select_synthetic_pothole_rows(
    synthetic_rows: list[dict[str, str]],
    max_synthetic_rows: int,
    random_seed: int = RDD_SPLIT_RANDOM_SEED,
) -> list[dict[str, str]]:
    if max_synthetic_rows < 1:
        raise ValueError("max_synthetic_rows must be at least 1.")
    rng = random.Random(f"{random_seed}:synthetic_potholes:{max_synthetic_rows}")
    if len(synthetic_rows) <= max_synthetic_rows:
        selected_rows = list(synthetic_rows)
    else:
        selected_rows = rng.sample(synthetic_rows, k=max_synthetic_rows)
    rng.shuffle(selected_rows)
    return selected_rows


def merge_synthetic_potholes_into_train_rows(
    train_rows: list[dict[str, str]],
    synthetic_rows: list[dict[str, str]],
    synthetic_pothole_ratio: float,
    random_seed: int = RDD_SPLIT_RANDOM_SEED,
) -> list[dict[str, str]]:
    max_synthetic_rows = calculate_synthetic_pothole_limit(
        train_rows,
        synthetic_pothole_ratio,
    )
    selected_synthetic_rows = select_synthetic_pothole_rows(
        synthetic_rows,
        max_synthetic_rows,
        random_seed=random_seed,
    )
    return [*train_rows, *selected_synthetic_rows]


def prepare_synthetic_binary_pothole_manifests(
    experiment_name: str,
    synthetic_pothole_ratio: float,
    source_dir: Path = RDD2022_BINARY_POTHOLE_DIR,
    synthetic_source_dir: Path = RDD2022_SYNTHETIC_POTHOLE_DIR,
    output_root: Path = RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
    random_seed: int = RDD_SPLIT_RANDOM_SEED,
) -> Path:
    train_rows = load_manifest_rows(source_dir / "train.csv")
    validation_rows = load_manifest_rows(source_dir / "validation.csv")
    test_rows = load_manifest_rows(source_dir / "test.csv")
    synthetic_rows = load_synthetic_pothole_rows(synthetic_source_dir)
    merged_train_rows = merge_synthetic_potholes_into_train_rows(
        train_rows,
        synthetic_rows,
        synthetic_pothole_ratio,
        random_seed=random_seed,
    )
    output_dir = output_root / experiment_name
    write_manifests(
        [*merged_train_rows, *validation_rows, *test_rows],
        output_dir,
    )
    return output_dir
