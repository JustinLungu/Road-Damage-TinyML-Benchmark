import csv

import pytest

from src.rdd_benchmark.constants import NEGATIVE_LABEL, POSITIVE_LABEL
from src.rdd_benchmark.data_loader.constants import MANIFEST_COLUMNS
from src.rdd_benchmark.experiments.dataset_builder import (
    RDDExperimentManifestPaths,
    build_experiment_manifest_paths,
    validate_manifest_paths,
)
from src.rdd_benchmark.experiments.experiment_registry import get_rdd_experiment_config


def make_row(row_id: str, split: str, label: int) -> dict[str, str]:
    return {
        "image_path": f"images/{row_id}.jpg",
        "annotation_path": f"annotations/{row_id}.xml",
        "country": "Builderland",
        "split": split,
        "label": str(label),
        "label_name": "pothole" if label else "non_pothole",
        "has_pothole": str(label),
        "num_objects": "1",
        "num_pothole_objects": str(label),
        "unique_object_labels": "D40" if label else "D00",
        "object_labels": "D40" if label else "D00",
        "image_width": "640",
        "image_height": "480",
    }


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_base_manifests(source_dir):
    write_manifest(
        source_dir / "train.csv",
        [
            *[make_row(f"p{index}", "train", POSITIVE_LABEL) for index in range(2)],
            *[make_row(f"n{index}", "train", NEGATIVE_LABEL) for index in range(12)],
        ],
    )
    write_manifest(source_dir / "validation.csv", [make_row("v", "validation", 0)])
    write_manifest(source_dir / "test.csv", [make_row("t", "test", 1)])


def test_validate_manifest_paths_rejects_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="Experiment manifest is missing"):
        validate_manifest_paths(RDDExperimentManifestPaths(tmp_path))


@pytest.mark.parametrize("experiment_id", ["A", "B", "C", "F"])
def test_non_downsampled_experiments_use_original_manifests(tmp_path, experiment_id):
    write_base_manifests(tmp_path)
    paths = build_experiment_manifest_paths(
        get_rdd_experiment_config(experiment_id),
        source_dir=tmp_path,
    )
    assert paths.dataset_dir == tmp_path


@pytest.mark.parametrize(
    ("experiment_id", "expected_negatives"), [("D", 6), ("G", 10), ("H", 10)]
)
def test_downsampled_experiments_create_expected_train_ratio(
    tmp_path,
    experiment_id,
    expected_negatives,
):
    source_dir = tmp_path / "source"
    write_base_manifests(source_dir)
    paths = build_experiment_manifest_paths(
        get_rdd_experiment_config(experiment_id),
        source_dir=source_dir,
        experiment_output_root=tmp_path / "experiments",
    )
    with paths.train_manifest_path.open(newline="", encoding="utf-8") as manifest:
        rows = list(csv.DictReader(manifest))
    assert sum(int(row["label"]) == POSITIVE_LABEL for row in rows) == 2
    assert (
        sum(int(row["label"]) == NEGATIVE_LABEL for row in rows) == expected_negatives
    )
