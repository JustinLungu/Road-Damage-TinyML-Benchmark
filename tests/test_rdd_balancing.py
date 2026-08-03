import csv

import pytest

from src.rdd_benchmark.constants import NEGATIVE_LABEL, POSITIVE_LABEL
from src.rdd_benchmark.data_loader.constants import MANIFEST_COLUMNS
from src.rdd_benchmark.data_loader.utils import load_manifest_rows
from src.rdd_benchmark.data_preprocessing.balancing import (
    downsample_majority_rows,
    prepare_balanced_binary_pothole_manifests,
    validate_non_potholes_per_pothole,
)


def make_row(row_id: str, split: str, label: int) -> dict[str, str]:
    return {
        "image_path": f"images/{row_id}.jpg",
        "annotation_path": f"annotations/{row_id}.xml",
        "country": "Testland",
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


def count_label(rows, label):
    return sum(int(row["label"]) == label for row in rows)


def test_load_manifest_rows_validates_required_columns(tmp_path):
    manifest_path = tmp_path / "train.csv"
    manifest_path.write_text("image_path,label\nx.jpg,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest_rows(manifest_path)


def test_validate_non_potholes_per_pothole_returns_ratio():
    assert validate_non_potholes_per_pothole(3) == 3


def test_validate_non_potholes_per_pothole_rejects_missing_ratio():
    with pytest.raises(ValueError, match="ratio is required"):
        validate_non_potholes_per_pothole(None)


def test_validate_non_potholes_per_pothole_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="at least 1"):
        validate_non_potholes_per_pothole(0)


def test_downsample_majority_rows_keeps_all_potholes_and_samples_ratio():
    rows = [
        *[make_row(f"pothole_{index}", "train", POSITIVE_LABEL) for index in range(2)],
        *[
            make_row(f"non_pothole_{index}", "train", NEGATIVE_LABEL)
            for index in range(10)
        ],
    ]

    balanced_rows = downsample_majority_rows(
        rows,
        non_potholes_per_pothole=3,
        random_seed=7,
    )

    assert len(balanced_rows) == 8
    assert count_label(balanced_rows, POSITIVE_LABEL) == 2
    assert count_label(balanced_rows, NEGATIVE_LABEL) == 6


def test_downsample_majority_rows_is_deterministic():
    rows = [
        *[make_row(f"pothole_{index}", "train", POSITIVE_LABEL) for index in range(2)],
        *[
            make_row(f"non_pothole_{index}", "train", NEGATIVE_LABEL)
            for index in range(10)
        ],
    ]

    first_rows = downsample_majority_rows(rows, 3, random_seed=7)
    second_rows = downsample_majority_rows(rows, 3, random_seed=7)

    assert [row["image_path"] for row in first_rows] == [
        row["image_path"] for row in second_rows
    ]


def test_downsample_majority_rows_does_not_duplicate_non_potholes():
    rows = [
        make_row("pothole_0", "train", POSITIVE_LABEL),
        make_row("non_pothole_0", "train", NEGATIVE_LABEL),
        make_row("non_pothole_1", "train", NEGATIVE_LABEL),
    ]

    balanced_rows = downsample_majority_rows(rows, 3, random_seed=7)

    assert len(balanced_rows) == 3
    assert count_label(balanced_rows, NEGATIVE_LABEL) == 2


def test_downsample_majority_rows_requires_both_classes():
    with pytest.raises(ValueError, match="without pothole rows"):
        downsample_majority_rows(
            [make_row("non_pothole_0", "train", NEGATIVE_LABEL)],
            3,
        )

    with pytest.raises(ValueError, match="without non-pothole rows"):
        downsample_majority_rows(
            [make_row("pothole_0", "train", POSITIVE_LABEL)],
            3,
        )


def test_prepare_balanced_manifests_only_changes_train_split(tmp_path):
    source_dir = tmp_path / "source"
    output_root = tmp_path / "balanced"
    train_rows = [
        *[
            make_row(f"train_pothole_{index}", "train", POSITIVE_LABEL)
            for index in range(2)
        ],
        *[
            make_row(f"train_non_pothole_{index}", "train", NEGATIVE_LABEL)
            for index in range(10)
        ],
    ]
    validation_rows = [
        make_row("validation_pothole", "validation", POSITIVE_LABEL),
        make_row("validation_non_pothole", "validation", NEGATIVE_LABEL),
    ]
    test_rows = [
        make_row("test_pothole", "test", POSITIVE_LABEL),
        make_row("test_non_pothole", "test", NEGATIVE_LABEL),
    ]
    write_manifest(source_dir / "train.csv", train_rows)
    write_manifest(source_dir / "validation.csv", validation_rows)
    write_manifest(source_dir / "test.csv", test_rows)

    output_dir = prepare_balanced_binary_pothole_manifests(
        experiment_name="experiment_d",
        non_potholes_per_pothole=3,
        source_dir=source_dir,
        output_root=output_root,
        random_seed=7,
    )

    balanced_train_rows = read_manifest(output_dir / "train.csv")
    balanced_validation_rows = read_manifest(output_dir / "validation.csv")
    balanced_test_rows = read_manifest(output_dir / "test.csv")

    assert output_dir == output_root / "experiment_d"
    assert count_label(balanced_train_rows, POSITIVE_LABEL) == 2
    assert count_label(balanced_train_rows, NEGATIVE_LABEL) == 6
    assert balanced_validation_rows == validation_rows
    assert balanced_test_rows == test_rows
    assert len(read_manifest(output_dir / "all.csv")) == 12
    assert (output_dir / "summary.csv").is_file()
