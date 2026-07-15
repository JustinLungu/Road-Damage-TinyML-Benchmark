import csv

import pytest

from src.rdd_benchmark.constants import NEGATIVE_LABEL, POSITIVE_LABEL
from src.rdd_benchmark.data_loader.constants import MANIFEST_COLUMNS
from src.rdd_benchmark.data_preprocessing.synthetic import (
    SYNTHETIC_MANIFEST_NAME,
    calculate_synthetic_pothole_limit,
    load_synthetic_pothole_rows,
    merge_synthetic_potholes_into_train_rows,
    prepare_synthetic_binary_pothole_manifests,
    resolve_synthetic_manifest_path,
    select_synthetic_pothole_rows,
)


def make_row(row_id: str, split: str, label: int) -> dict[str, str]:
    return {
        "image_path": f"images/{row_id}.jpg",
        "annotation_path": f"annotations/{row_id}.xml",
        "country": "Syntheticland" if row_id.startswith("synthetic") else "Realland",
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


def test_resolve_synthetic_manifest_path_reports_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="source folder is missing"):
        resolve_synthetic_manifest_path(tmp_path / "missing")


def test_resolve_synthetic_manifest_path_reports_missing_manifest(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="manifest is missing"):
        resolve_synthetic_manifest_path(synthetic_dir)


def test_load_synthetic_pothole_rows_keeps_only_positive_rows(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    write_manifest(
        synthetic_dir / SYNTHETIC_MANIFEST_NAME,
        [
            make_row("synthetic_pothole", "unused", POSITIVE_LABEL),
            make_row("synthetic_negative", "unused", NEGATIVE_LABEL),
        ],
    )

    rows = load_synthetic_pothole_rows(synthetic_dir)

    assert len(rows) == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["label"] == str(POSITIVE_LABEL)
    assert rows[0]["label_name"] == "pothole"


def test_load_synthetic_pothole_rows_rejects_manifest_without_potholes(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    write_manifest(
        synthetic_dir / SYNTHETIC_MANIFEST_NAME,
        [make_row("synthetic_negative", "train", NEGATIVE_LABEL)],
    )

    with pytest.raises(ValueError, match="no pothole rows"):
        load_synthetic_pothole_rows(synthetic_dir)


def test_calculate_synthetic_pothole_limit_caps_by_real_pothole_count():
    train_rows = [
        make_row(f"real_pothole_{index}", "train", POSITIVE_LABEL)
        for index in range(10)
    ]

    assert calculate_synthetic_pothole_limit(train_rows, 0.2) == 2


def test_calculate_synthetic_pothole_limit_keeps_at_least_one_row():
    train_rows = [make_row("real_pothole", "train", POSITIVE_LABEL)]

    assert calculate_synthetic_pothole_limit(train_rows, 0.2) == 1


def test_calculate_synthetic_pothole_limit_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="greater than 0"):
        calculate_synthetic_pothole_limit(
            [make_row("real_pothole", "train", POSITIVE_LABEL)],
            0,
        )


def test_select_synthetic_pothole_rows_is_deterministic_and_capped():
    synthetic_rows = [
        make_row(f"synthetic_{index}", "train", POSITIVE_LABEL)
        for index in range(10)
    ]

    first_rows = select_synthetic_pothole_rows(
        synthetic_rows,
        max_synthetic_rows=3,
        random_seed=7,
    )
    second_rows = select_synthetic_pothole_rows(
        synthetic_rows,
        max_synthetic_rows=3,
        random_seed=7,
    )

    assert len(first_rows) == 3
    assert [row["image_path"] for row in first_rows] == [
        row["image_path"] for row in second_rows
    ]


def test_merge_synthetic_potholes_into_train_rows_caps_synthetic_addition():
    train_rows = [
        *[
            make_row(f"real_pothole_{index}", "train", POSITIVE_LABEL)
            for index in range(10)
        ],
        make_row("real_non_pothole", "train", NEGATIVE_LABEL),
    ]
    synthetic_rows = [
        make_row(f"synthetic_{index}", "train", POSITIVE_LABEL)
        for index in range(10)
    ]

    merged_rows = merge_synthetic_potholes_into_train_rows(
        train_rows,
        synthetic_rows,
        synthetic_pothole_ratio=0.2,
        random_seed=7,
    )

    assert len(merged_rows) == len(train_rows) + 2


def test_prepare_synthetic_manifests_only_adds_to_train_split(tmp_path):
    source_dir = tmp_path / "source"
    synthetic_dir = tmp_path / "synthetic"
    output_root = tmp_path / "experiments"
    train_rows = [
        *[
            make_row(f"real_pothole_{index}", "train", POSITIVE_LABEL)
            for index in range(10)
        ],
        make_row("real_non_pothole", "train", NEGATIVE_LABEL),
    ]
    validation_rows = [
        make_row("validation_pothole", "validation", POSITIVE_LABEL),
        make_row("validation_non_pothole", "validation", NEGATIVE_LABEL),
    ]
    test_rows = [
        make_row("test_pothole", "test", POSITIVE_LABEL),
        make_row("test_non_pothole", "test", NEGATIVE_LABEL),
    ]
    synthetic_rows = [
        make_row(f"synthetic_{index}", "synthetic", POSITIVE_LABEL)
        for index in range(10)
    ]
    write_manifest(source_dir / "train.csv", train_rows)
    write_manifest(source_dir / "validation.csv", validation_rows)
    write_manifest(source_dir / "test.csv", test_rows)
    write_manifest(synthetic_dir / SYNTHETIC_MANIFEST_NAME, synthetic_rows)

    output_dir = prepare_synthetic_binary_pothole_manifests(
        experiment_name="experiment_e",
        synthetic_pothole_ratio=0.2,
        source_dir=source_dir,
        synthetic_source_dir=synthetic_dir,
        output_root=output_root,
        random_seed=7,
    )

    merged_train_rows = read_manifest(output_dir / "train.csv")
    merged_validation_rows = read_manifest(output_dir / "validation.csv")
    merged_test_rows = read_manifest(output_dir / "test.csv")

    assert output_dir == output_root / "experiment_e"
    assert len(merged_train_rows) == len(train_rows) + 2
    assert count_label(merged_train_rows, POSITIVE_LABEL) == 12
    assert merged_validation_rows == validation_rows
    assert merged_test_rows == test_rows
    assert (output_dir / "summary.csv").is_file()
