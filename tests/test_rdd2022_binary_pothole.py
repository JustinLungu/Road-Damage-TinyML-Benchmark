from pathlib import Path

import pytest

from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole import (
    build_manifest_rows,
    build_summary_rows,
    calculate_split_counts,
    parse_annotation,
    validate_split_countries,
    write_manifests,
)


def write_rdd_sample(
    root: Path,
    country: str,
    filename: str,
    labels: list[str],
    box_size: int = 30,
) -> Path:
    image_dir = root / country / "train" / "images"
    annotation_dir = root / country / "train" / "annotations" / "xmls"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / filename).write_bytes(b"image")

    objects = "\n".join(
        f"""
        <object>
            <name>{label}</name>
            <bndbox>
                <xmin>1</xmin>
                <ymin>2</ymin>
                <xmax>{1 + box_size}</xmax>
                <ymax>{2 + box_size}</ymax>
            </bndbox>
        </object>
        """
        for label in labels
    )
    annotation_path = annotation_dir / f"{Path(filename).stem}.xml"
    annotation_path.write_text(
        f"""
        <annotation>
            <filename>{filename}</filename>
            <size>
                <width>640</width>
                <height>480</height>
            </size>
            {objects}
        </annotation>
        """,
        encoding="utf-8",
    )
    return annotation_path


def test_parse_annotation_assigns_binary_pothole_label(tmp_path) -> None:
    annotation_path = write_rdd_sample(
        tmp_path,
        "India",
        "positive.jpg",
        ["D00", "D40", "D40"],
    )
    row = parse_annotation(
        annotation_path,
        tmp_path / "India" / "train" / "images",
        "India",
        "train",
    )

    assert row["label"] == 1
    assert row["label_name"] == "pothole"
    assert row["has_pothole"] == 1
    assert row["num_objects"] == 3
    assert row["num_pothole_objects"] == 2
    assert row["unique_object_labels"] == "D00|D40"
    assert row["object_labels"] == "D00|D40|D40"
    assert row["image_width"] == 640
    assert row["image_height"] == 480

    negative_annotation = write_rdd_sample(
        tmp_path,
        "India",
        "negative.jpg",
        ["D00", "D20"],
    )
    negative_row = parse_annotation(
        negative_annotation,
        tmp_path / "India" / "train" / "images",
        "India",
        "validation",
    )

    assert negative_row["label"] == 0
    assert negative_row["label_name"] == "non_pothole"
    assert negative_row["has_pothole"] == 0


def test_parse_annotation_ignores_tiny_potholes(tmp_path) -> None:
    annotation_path = write_rdd_sample(
        tmp_path,
        "India",
        "tiny_pothole.jpg",
        ["D40"],
        box_size=10,
    )

    row = parse_annotation(
        annotation_path,
        tmp_path / "India" / "train" / "images",
        "India",
        "train",
    )

    assert row["label"] == 0
    assert row["label_name"] == "non_pothole"
    assert row["has_pothole"] == 0
    assert row["num_objects"] == 1
    assert row["num_pothole_objects"] == 0
    assert row["object_labels"] == "D40"


def test_build_manifest_rows_writes_split_manifests_and_summary(tmp_path) -> None:
    write_rdd_sample(tmp_path, "TrainCountry", "train_pos.jpg", ["D40"])
    write_rdd_sample(tmp_path, "TrainCountry", "train_neg.jpg", ["D10"])
    write_rdd_sample(tmp_path, "ValidationCountry", "val.jpg", [])
    write_rdd_sample(tmp_path, "TestCountry", "test.jpg", ["D40", "D20"])

    rows = build_manifest_rows(
        tmp_path,
        {
            "train": ("TrainCountry",),
            "validation": ("ValidationCountry",),
            "test": ("TestCountry",),
        },
    )

    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert sum(int(row["has_pothole"]) for row in rows) == 2

    output_dir = tmp_path / "manifests"
    write_manifests(rows, output_dir)

    assert (output_dir / "all.csv").is_file()
    assert (output_dir / "train.csv").read_text(encoding="utf-8").count("\n") == 3
    assert (output_dir / "validation.csv").read_text(encoding="utf-8").count("\n") == 2
    assert (output_dir / "test.csv").read_text(encoding="utf-8").count("\n") == 2

    summary_rows = build_summary_rows(rows)
    train_total = next(
        row for row in summary_rows
        if row["split"] == "train" and row["country"] == "__split_total__"
    )
    dataset_total = next(
        row for row in summary_rows
        if row["split"] == "all" and row["country"] == "__dataset_total__"
    )

    assert train_total["images"] == 2
    assert train_total["pothole_images"] == 1
    assert dataset_total["images"] == 4
    assert dataset_total["pothole_images"] == 2


def test_build_manifest_rows_can_use_stratified_country_split(tmp_path) -> None:
    for index in range(10):
        write_rdd_sample(tmp_path, "India", f"positive_{index}.jpg", ["D40"])
        write_rdd_sample(tmp_path, "India", f"negative_{index}.jpg", ["D10"])

    rows = build_manifest_rows(
        tmp_path,
        split_countries={
            "train": ("UnusedTrainCountry",),
            "validation": ("UnusedValidationCountry",),
            "test": ("UnusedTestCountry",),
        },
        split_mode="stratified_by_country",
        countries=("India",),
        split_fractions={"train": 0.70, "validation": 0.15, "test": 0.15},
        random_seed=42,
    )

    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    positive_counts = {
        split: sum(row["split"] == split and int(row["has_pothole"]) for row in rows)
        for split in ("train", "validation", "test")
    }

    assert split_counts == {"train": 12, "validation": 4, "test": 4}
    assert positive_counts == {"train": 6, "validation": 2, "test": 2}
    assert {row["country"] for row in rows} == {"India"}


def test_calculate_split_counts_keeps_small_groups_available_for_training() -> None:
    assert calculate_split_counts(
        total_rows=1,
        split_fractions={"train": 0.70, "validation": 0.15, "test": 0.15},
    ) == {"train": 1, "validation": 0, "test": 0}


def test_binary_pothole_manifest_validation_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation must contain"):
        validate_split_countries(
            {
                "train": ("India",),
                "validation": (),
                "test": ("Japan",),
            }
        )

    with pytest.raises(ValueError, match="appears in both"):
        validate_split_countries(
            {
                "train": ("India",),
                "validation": ("India",),
                "test": ("Japan",),
            }
        )

    with pytest.raises(FileNotFoundError, match="RDD2022 root"):
        build_manifest_rows(
            tmp_path / "missing",
            {
                "train": ("India",),
                "validation": ("United_States",),
                "test": ("Japan",),
            },
        )

    with pytest.raises(FileNotFoundError, match="Expected RDD2022"):
        build_manifest_rows(
            tmp_path,
            {
                "train": ("MissingCountry",),
                "validation": ("United_States",),
                "test": ("Japan",),
            },
        )
