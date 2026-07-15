from pathlib import Path

import pytest

from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole_patches import (
    BinaryPotholePatchManifestPreprocessor,
    BoundingBox,
    build_patch_summary_rows,
    make_padded_square_patch_box,
    make_patch_rows,
    parse_image_annotation,
)


def write_rdd_patch_sample(
    root: Path,
    country: str,
    filename: str,
    objects: list[tuple[str, tuple[int, int, int, int]]],
) -> Path:
    image_dir = root / country / "train" / "images"
    annotation_dir = root / country / "train" / "annotations" / "xmls"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / filename).write_bytes(b"image")

    xml_objects = "\n".join(
        f"""
        <object>
            <name>{label}</name>
            <bndbox>
                <xmin>{box[0]}</xmin>
                <ymin>{box[1]}</ymin>
                <xmax>{box[2]}</xmax>
                <ymax>{box[3]}</ymax>
            </bndbox>
        </object>
        """
        for label, box in objects
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
            {xml_objects}
        </annotation>
        """,
        encoding="utf-8",
    )
    return annotation_path


def test_make_padded_square_patch_box_clips_to_image_bounds() -> None:
    patch_box = make_padded_square_patch_box(
        bbox=BoundingBox(xmin=1, ymin=2, xmax=41, ymax=82),
        image_width=100,
        image_height=90,
        padding=0.25,
    )

    assert patch_box.xmin == 0
    assert patch_box.ymin == 0
    assert patch_box.xmax == 90
    assert patch_box.ymax == 90


def test_parse_image_annotation_and_make_patch_rows(tmp_path) -> None:
    annotation_path = write_rdd_patch_sample(
        tmp_path,
        "India",
        "sample.jpg",
        [
            ("D40", (10, 20, 50, 60)),
            ("D00", (100, 110, 150, 170)),
            ("D10", (1, 1, 5, 5)),
        ],
    )

    image_annotation = parse_image_annotation(
        annotation_path=annotation_path,
        image_dir=tmp_path / "India" / "train" / "images",
        country="India",
        split="train",
    )
    rows = make_patch_rows(
        image_annotation=image_annotation,
        patch_padding=0.0,
        min_box_area=400,
    )

    assert len(rows) == 2
    assert rows[0]["label"] == 1
    assert rows[0]["label_name"] == "pothole"
    assert rows[0]["source_object_label"] == "D40"
    assert rows[0]["patch_source"] == "annotation_box"
    assert rows[1]["label"] == 0
    assert rows[1]["label_name"] == "non_pothole"
    assert rows[1]["source_object_label"] == "D00"
    assert rows[0]["image_width"] == 640
    assert rows[0]["image_height"] == 480


def test_patch_preprocessor_writes_split_manifests_and_summary(tmp_path) -> None:
    write_rdd_patch_sample(
        tmp_path,
        "TrainCountry",
        "train.jpg",
        [("D40", (10, 10, 50, 50)), ("D20", (100, 100, 150, 150))],
    )
    write_rdd_patch_sample(
        tmp_path,
        "ValidationCountry",
        "validation.jpg",
        [("D00", (20, 20, 80, 80))],
    )
    write_rdd_patch_sample(
        tmp_path,
        "TestCountry",
        "test.jpg",
        [("D40", (30, 30, 90, 90))],
    )

    output_dir = tmp_path / "patch_manifests"
    preprocessor = BinaryPotholePatchManifestPreprocessor(
        rdd_root=tmp_path,
        output_dir=output_dir,
        split_countries={
            "train": ("TrainCountry",),
            "validation": ("ValidationCountry",),
            "test": ("TestCountry",),
        },
        patch_padding=0.0,
        min_box_area=400,
    )

    rows = preprocessor.build_rows()
    preprocessor.write_manifests(rows)

    assert len(rows) == 4
    assert (output_dir / "all.csv").is_file()
    assert (output_dir / "train.csv").read_text(encoding="utf-8").count("\n") == 3
    assert (output_dir / "validation.csv").read_text(encoding="utf-8").count("\n") == 2
    assert (output_dir / "test.csv").read_text(encoding="utf-8").count("\n") == 2

    summary_rows = build_patch_summary_rows(rows)
    train_total = next(
        row
        for row in summary_rows
        if row["split"] == "train" and row["country"] == "__split_total__"
    )
    dataset_total = next(
        row
        for row in summary_rows
        if row["split"] == "all" and row["country"] == "__dataset_total__"
    )

    assert train_total["patches"] == 2
    assert train_total["pothole_patches"] == 1
    assert dataset_total["patches"] == 4
    assert dataset_total["pothole_patches"] == 2


def test_patch_preprocessor_can_inherit_stratified_full_image_split(tmp_path) -> None:
    for index in range(3):
        write_rdd_patch_sample(
            tmp_path,
            "India",
            f"positive_{index}.jpg",
            [("D40", (10, 10, 50, 50))],
        )
        write_rdd_patch_sample(
            tmp_path,
            "India",
            f"negative_{index}.jpg",
            [("D20", (100, 100, 150, 150))],
        )

    preprocessor = BinaryPotholePatchManifestPreprocessor(
        rdd_root=tmp_path,
        output_dir=tmp_path / "patch_manifests",
        split_countries={
            "train": ("UnusedTrainCountry",),
            "validation": ("UnusedValidationCountry",),
            "test": ("UnusedTestCountry",),
        },
        split_mode="stratified_by_country",
        countries=("India",),
        split_fractions={"train": 0.70, "validation": 0.15, "test": 0.15},
        random_seed=42,
        patch_padding=0.0,
        min_box_area=400,
    )

    rows = preprocessor.build_rows()

    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    } == {"train": 2, "validation": 2, "test": 2}


def test_patch_preprocessor_validation_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="RDD2022 root"):
        BinaryPotholePatchManifestPreprocessor(
            rdd_root=tmp_path / "missing",
            output_dir=tmp_path / "out",
            split_countries={"train": ("India",)},
        ).build_rows()

    with pytest.raises(FileNotFoundError, match="Expected RDD2022"):
        BinaryPotholePatchManifestPreprocessor(
            rdd_root=tmp_path,
            output_dir=tmp_path / "out",
            split_countries={"train": ("MissingCountry",)},
        ).build_rows()
