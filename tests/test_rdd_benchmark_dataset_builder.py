import csv

import pytest

from src.rdd_benchmark.constants import NEGATIVE_LABEL, POSITIVE_LABEL
from src.rdd_benchmark.data_loader.constants import MANIFEST_COLUMNS
from src.rdd_benchmark.data_preprocessing.synthetic import SYNTHETIC_MANIFEST_NAME
from src.rdd_benchmark.experiments.dataset_builder import (
    build_experiment_manifest_paths,
    make_manifest_paths,
    validate_manifest_paths,
)
from src.rdd_benchmark.experiments.experiment_registry import (
    get_rdd_experiment_config,
)


def make_row(row_id: str, split: str, label: int) -> dict[str, str]:
    return {
        "image_path": f"images/{row_id}.jpg",
        "annotation_path": f"annotations/{row_id}.xml",
        "country": "Builderland",
        "split": split,
        "label": str(label),
        "label_name": "pothole" if label == POSITIVE_LABEL else "non_pothole",
        "has_pothole": str(int(label == POSITIVE_LABEL)),
        "num_objects": "1",
        "num_pothole_objects": str(int(label == POSITIVE_LABEL)),
        "unique_object_labels": "D40" if label == POSITIVE_LABEL else "D00",
        "object_labels": "D40" if label == POSITIVE_LABEL else "D00",
        "image_width": "640",
        "image_height": "480",
    }


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def write_base_manifests(source_dir):
    write_manifest(
        source_dir / "train.csv",
        [
            *[
                make_row(f"train_pothole_{index}", "train", POSITIVE_LABEL)
                for index in range(2)
            ],
            *[
                make_row(f"train_non_pothole_{index}", "train", NEGATIVE_LABEL)
                for index in range(10)
            ],
        ],
    )
    write_manifest(
        source_dir / "validation.csv",
        [
            make_row("validation_pothole", "validation", POSITIVE_LABEL),
            make_row("validation_non_pothole", "validation", NEGATIVE_LABEL),
        ],
    )
    write_manifest(
        source_dir / "test.csv",
        [
            make_row("test_pothole", "test", POSITIVE_LABEL),
            make_row("test_non_pothole", "test", NEGATIVE_LABEL),
        ],
    )


def count_label(rows, label):
    return sum(int(row["label"]) == label for row in rows)


def test_validate_manifest_paths_rejects_missing_manifest(tmp_path):
    manifest_paths = make_manifest_paths("missing", tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="Experiment manifest is missing"):
        validate_manifest_paths(manifest_paths)


@pytest.mark.parametrize("experiment_id", ["A", "B", "C", "F"])
def test_build_experiment_manifest_paths_uses_original_manifests_for_non_downsampled(
    tmp_path,
    experiment_id,
):
    source_dir = tmp_path / "source"
    write_base_manifests(source_dir)
    config = get_rdd_experiment_config(experiment_id)

    manifest_paths = build_experiment_manifest_paths(config, source_dir=source_dir)

    assert manifest_paths.dataset_dir == source_dir
    assert manifest_paths.train_manifest_path == source_dir / "train.csv"
    assert manifest_paths.validation_manifest_path == source_dir / "validation.csv"
    assert manifest_paths.test_manifest_path == source_dir / "test.csv"


@pytest.mark.parametrize(
    ("experiment_id", "expected_non_potholes"),
    [("D", 6), ("G", 10), ("H", 10)],
)
def test_build_experiment_manifest_paths_creates_downsampled_manifests(
    tmp_path,
    experiment_id,
    expected_non_potholes,
):
    source_dir = tmp_path / "source"
    output_root = tmp_path / "experiments"
    write_base_manifests(source_dir)
    config = get_rdd_experiment_config(experiment_id)

    manifest_paths = build_experiment_manifest_paths(
        config,
        source_dir=source_dir,
        experiment_output_root=output_root,
    )
    train_rows = read_manifest(manifest_paths.train_manifest_path)
    validation_rows = read_manifest(manifest_paths.validation_manifest_path)

    assert manifest_paths.dataset_dir == output_root / config.experiment_name
    assert count_label(train_rows, POSITIVE_LABEL) == 2
    assert count_label(train_rows, NEGATIVE_LABEL) == expected_non_potholes
    assert len(validation_rows) == 2


def test_build_experiment_manifest_paths_requires_best_previous_for_e(tmp_path):
    config = get_rdd_experiment_config("E")

    with pytest.raises(ValueError, match="best_previous_experiment_name"):
        build_experiment_manifest_paths(config, experiment_output_root=tmp_path)


def test_build_experiment_manifest_paths_adds_synthetic_rows_for_e(tmp_path):
    source_dir = tmp_path / "source"
    output_root = tmp_path / "experiments"
    synthetic_dir = tmp_path / "synthetic"
    best_previous_name = "best_previous"
    best_previous_dir = output_root / best_previous_name
    write_base_manifests(source_dir)
    write_base_manifests(best_previous_dir)
    write_manifest(
        synthetic_dir / SYNTHETIC_MANIFEST_NAME,
        [
            make_row(f"synthetic_{index}", "synthetic", POSITIVE_LABEL)
            for index in range(10)
        ],
    )
    config = get_rdd_experiment_config("E")

    manifest_paths = build_experiment_manifest_paths(
        config,
        source_dir=source_dir,
        experiment_output_root=output_root,
        synthetic_source_dir=synthetic_dir,
        best_previous_experiment_name=best_previous_name,
    )
    train_rows = read_manifest(manifest_paths.train_manifest_path)

    assert manifest_paths.dataset_dir == output_root / config.experiment_name
    assert len(train_rows) == 13
    assert count_label(train_rows, POSITIVE_LABEL) == 3
