import runpy
from pathlib import Path

import pytest
import torch
from PIL import Image

import src.rdd_benchmark.constants as rdd_constants
import src.rdd_benchmark.data_loader.utils as data_loader_utils
import src.rdd_benchmark.training.utils as training_utils
from src.rdd_benchmark.data_loader.dataset import (
    BinaryPotholeDataset,
    BinaryPotholeManifest,
    BinaryPotholePatchDataset,
    BinaryPotholePatchManifest,
    BinaryPotholePatchSample,
    BinaryPotholeSample,
)
from src.rdd_benchmark.data_loader.utils import (
    expected_label_name,
    make_rdd_training_dataset,
    make_rdd_validation_dataset,
    parse_binary_label,
    resolve_manifest_path,
    select_rdd_manifest_path,
)


def write_image(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (4, 3), color=128).save(path)


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "image_path,annotation_path,country,split,label,label_name",
                *[
                    ",".join(
                        [
                            row["image_path"],
                            row["annotation_path"],
                            row["country"],
                            row["split"],
                            row["label"],
                            row["label_name"],
                        ]
                    )
                    for row in rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_patch_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "image_path",
        "annotation_path",
        "country",
        "split",
        "label",
        "label_name",
        "source_object_label",
        "patch_source",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "patch_xmin",
        "patch_ymin",
        "patch_xmax",
        "patch_ymax",
        "image_width",
        "image_height",
    ]
    path.write_text(
        ",".join(header)
        + "\n"
        + "\n".join(",".join(row[column] for column in header) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def make_row(
    image_path: Path,
    annotation_path: Path,
    label: int,
    label_name: str,
    split: str = "train",
    country: str = "India",
) -> dict[str, str]:
    return {
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
        "country": country,
        "split": split,
        "label": str(label),
        "label_name": label_name,
    }


def make_patch_row(
    image_path: Path,
    annotation_path: Path,
    label: int,
    label_name: str,
    split: str = "train",
    country: str = "India",
) -> dict[str, str]:
    return {
        **make_row(image_path, annotation_path, label, label_name, split, country),
        "source_object_label": "D40" if label == 1 else "D00",
        "patch_source": "annotation_box",
        "bbox_xmin": "1",
        "bbox_ymin": "1",
        "bbox_xmax": "3",
        "bbox_ymax": "3",
        "patch_xmin": "1",
        "patch_ymin": "0",
        "patch_xmax": "4",
        "patch_ymax": "2",
        "image_width": "4",
        "image_height": "3",
    }


def test_binary_pothole_manifest_loads_samples_counts_and_paths(tmp_path) -> None:
    image_a = tmp_path / "images" / "a.jpg"
    image_b = tmp_path / "images" / "b.jpg"
    annotation_a = tmp_path / "annotations" / "a.xml"
    annotation_b = tmp_path / "annotations" / "b.xml"
    write_image(image_a)
    write_image(image_b)
    annotation_a.parent.mkdir(parents=True)
    annotation_a.write_text("<annotation />", encoding="utf-8")
    annotation_b.write_text("<annotation />", encoding="utf-8")

    manifest_path = tmp_path / "train.csv"
    write_manifest(
        manifest_path,
        [
            make_row(image_a, annotation_a, 1, "pothole", country="India"),
            make_row(image_b, annotation_b, 0, "non_pothole", country="Czech"),
        ],
    )

    manifest = BinaryPotholeManifest.from_csv(manifest_path, expected_split="train")

    assert len(manifest) == 2
    assert list(manifest)[0] == BinaryPotholeSample(
        image_path=image_a,
        annotation_path=annotation_a,
        label=1,
        label_name="pothole",
        country="India",
        split="train",
    )
    assert manifest.class_counts() == {1: 1, 0: 1}
    assert manifest.country_counts() == {"India": 1, "Czech": 1}
    assert manifest.positive_fraction() == pytest.approx(0.5)


def test_binary_pothole_dataset_returns_image_label_and_metadata(tmp_path) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    annotation_path = tmp_path / "annotations" / "sample.xml"
    write_image(image_path, mode="L")
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text("<annotation />", encoding="utf-8")

    manifest_path = tmp_path / "validation.csv"
    write_manifest(
        manifest_path,
        [
            make_row(
                image_path,
                annotation_path,
                1,
                "pothole",
                split="validation",
                country="United_States",
            )
        ],
    )

    def image_transform(image: Image.Image) -> torch.Tensor:
        assert image.mode == "RGB"
        return torch.ones(3, image.height, image.width)

    dataset = BinaryPotholeDataset(
        manifest_path,
        expected_split="validation",
        transform=image_transform,
        target_transform=lambda label: torch.tensor(label),
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image"].shape == (3, 3, 4)
    assert item["label"].item() == 1
    assert item["image_path"] == image_path
    assert item["annotation_path"] == annotation_path
    assert item["label_name"] == "pothole"
    assert item["country"] == "United_States"
    assert item["split"] == "validation"


def test_binary_pothole_patch_manifest_loads_samples_counts_and_paths(tmp_path) -> None:
    image_a = tmp_path / "images" / "a.jpg"
    image_b = tmp_path / "images" / "b.jpg"
    annotation_a = tmp_path / "annotations" / "a.xml"
    annotation_b = tmp_path / "annotations" / "b.xml"
    write_image(image_a)
    write_image(image_b)
    annotation_a.parent.mkdir(parents=True)
    annotation_a.write_text("<annotation />", encoding="utf-8")
    annotation_b.write_text("<annotation />", encoding="utf-8")

    manifest_path = tmp_path / "patch_train.csv"
    write_patch_manifest(
        manifest_path,
        [
            make_patch_row(image_a, annotation_a, 1, "pothole", country="India"),
            make_patch_row(image_b, annotation_b, 0, "non_pothole", country="Czech"),
        ],
    )

    manifest = BinaryPotholePatchManifest.from_csv(
        manifest_path,
        expected_split="train",
    )

    assert len(manifest) == 2
    assert list(manifest)[0] == BinaryPotholePatchSample(
        image_path=image_a,
        annotation_path=annotation_a,
        label=1,
        label_name="pothole",
        country="India",
        split="train",
        source_object_label="D40",
        patch_source="annotation_box",
        bbox=(1, 1, 3, 3),
        patch_box=(1, 0, 4, 2),
        image_size=(4, 3),
    )
    assert manifest.class_counts() == {1: 1, 0: 1}
    assert manifest.country_counts() == {"India": 1, "Czech": 1}
    assert manifest.positive_fraction() == pytest.approx(0.5)


def test_binary_pothole_patch_dataset_returns_cropped_patch_and_metadata(
    tmp_path,
) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    annotation_path = tmp_path / "annotations" / "sample.xml"
    write_image(image_path)
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text("<annotation />", encoding="utf-8")

    manifest_path = tmp_path / "patch_validation.csv"
    write_patch_manifest(
        manifest_path,
        [
            make_patch_row(
                image_path,
                annotation_path,
                1,
                "pothole",
                split="validation",
                country="United_States",
            )
        ],
    )

    def image_transform(image: Image.Image) -> torch.Tensor:
        assert image.mode == "RGB"
        assert image.size == (3, 2)
        return torch.ones(3, image.height, image.width)

    dataset = BinaryPotholePatchDataset(
        manifest_path,
        expected_split="validation",
        transform=image_transform,
        target_transform=lambda label: torch.tensor(label),
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image"].shape == (3, 2, 3)
    assert item["label"].item() == 1
    assert item["image_path"] == image_path
    assert item["annotation_path"] == annotation_path
    assert item["label_name"] == "pothole"
    assert item["country"] == "United_States"
    assert item["split"] == "validation"
    assert item["source_object_label"] == "D40"
    assert item["patch_source"] == "annotation_box"
    assert item["bbox"] == (1, 1, 3, 3)
    assert item["patch_box"] == (1, 0, 4, 2)
    assert item["image_size"] == (4, 3)


def test_binary_pothole_manifest_rejects_invalid_rows(tmp_path) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    annotation_path = tmp_path / "annotations" / "sample.xml"
    write_image(image_path)
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text("<annotation />", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="manifest"):
        BinaryPotholeManifest.from_csv(tmp_path / "missing.csv")

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("image_path,label\nx.jpg,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        BinaryPotholeManifest.from_csv(bad_header)

    bad_label = tmp_path / "bad_label.csv"
    write_manifest(
        bad_label,
        [make_row(image_path, annotation_path, 2, "pothole")],
    )
    with pytest.raises(ValueError, match="must be 0 or 1"):
        BinaryPotholeManifest.from_csv(bad_label)

    bad_label_name = tmp_path / "bad_label_name.csv"
    write_manifest(
        bad_label_name,
        [make_row(image_path, annotation_path, 1, "non_pothole")],
    )
    with pytest.raises(ValueError, match="label_name must be pothole"):
        BinaryPotholeManifest.from_csv(bad_label_name)

    bad_split = tmp_path / "bad_split.csv"
    write_manifest(
        bad_split,
        [make_row(image_path, annotation_path, 0, "non_pothole", split="test")],
    )
    with pytest.raises(ValueError, match="Expected split train"):
        BinaryPotholeManifest.from_csv(bad_split, expected_split="train")

    bad_country = tmp_path / "bad_country.csv"
    write_manifest(
        bad_country,
        [make_row(image_path, annotation_path, 0, "non_pothole", country="")],
    )
    with pytest.raises(ValueError, match="Missing country"):
        BinaryPotholeManifest.from_csv(bad_country)


def test_binary_pothole_patch_manifest_rejects_invalid_rows(tmp_path) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    annotation_path = tmp_path / "annotations" / "sample.xml"
    write_image(image_path)
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text("<annotation />", encoding="utf-8")

    bad_header = tmp_path / "bad_patch_header.csv"
    bad_header.write_text(
        "image_path,annotation_path,country,split,label,label_name\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Patch manifest is missing"):
        BinaryPotholePatchManifest.from_csv(bad_header)

    bad_bounds = tmp_path / "bad_patch_bounds.csv"
    row = make_patch_row(image_path, annotation_path, 1, "pothole")
    row["patch_xmax"] = "1"
    write_patch_manifest(bad_bounds, [row])
    with pytest.raises(ValueError, match="invalid bounds"):
        BinaryPotholePatchManifest.from_csv(bad_bounds)

    out_of_bounds = tmp_path / "out_of_bounds.csv"
    row = make_patch_row(image_path, annotation_path, 1, "pothole")
    row["patch_xmax"] = "5"
    write_patch_manifest(out_of_bounds, [row])
    with pytest.raises(ValueError, match="exceeds image bounds"):
        BinaryPotholePatchManifest.from_csv(out_of_bounds)


def test_rdd_dataset_factories_select_full_image_and_patch_datasets(
    tmp_path,
    monkeypatch,
) -> None:
    full_manifest_dir = tmp_path / "binary_pothole"
    patch_manifest_dir = tmp_path / "binary_pothole_patches"
    monkeypatch.setattr(
        data_loader_utils,
        "RDD2022_BINARY_POTHOLE_DIR",
        full_manifest_dir,
    )
    monkeypatch.setattr(
        data_loader_utils,
        "RDD2022_BINARY_POTHOLE_PATCH_DIR",
        patch_manifest_dir,
    )

    train_image = tmp_path / "images" / "train.jpg"
    validation_image = tmp_path / "images" / "validation.jpg"
    train_annotation = tmp_path / "annotations" / "train.xml"
    validation_annotation = tmp_path / "annotations" / "validation.xml"
    write_image(train_image)
    write_image(validation_image)
    train_annotation.parent.mkdir(parents=True)
    train_annotation.write_text("<annotation />", encoding="utf-8")
    validation_annotation.write_text("<annotation />", encoding="utf-8")

    write_manifest(
        full_manifest_dir / "train.csv",
        [make_row(train_image, train_annotation, 1, "pothole")],
    )
    write_patch_manifest(
        patch_manifest_dir / "validation.csv",
        [
            make_patch_row(
                validation_image,
                validation_annotation,
                0,
                "non_pothole",
                split="validation",
            )
        ],
    )

    assert select_rdd_manifest_path("train", "full_image") == (
        full_manifest_dir / "train.csv"
    )
    assert select_rdd_manifest_path("validation", "annotation_patch") == (
        patch_manifest_dir / "validation.csv"
    )

    full_dataset = make_rdd_training_dataset(input_mode="full_image")
    patch_dataset = make_rdd_validation_dataset(input_mode="annotation_patch")

    assert isinstance(full_dataset, BinaryPotholeDataset)
    assert isinstance(patch_dataset, BinaryPotholePatchDataset)
    assert len(full_dataset) == 1
    assert len(patch_dataset) == 1


def test_rdd_dataset_factories_reject_invalid_modes_and_splits() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        select_rdd_manifest_path("train", "bad_mode")

    with pytest.raises(ValueError, match="Unsupported RDD split"):
        select_rdd_manifest_path("holdout", "full_image")


def test_rdd_training_utils_parse_labels_and_resolve_paths(tmp_path) -> None:
    manifest_path = tmp_path / "manifests" / "train.csv"
    manifest_path.parent.mkdir()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    project_file = project_root / "datasets" / "image.jpg"
    write_image(project_file)

    assert parse_binary_label("0", 2) == 0
    assert parse_binary_label("1", 3) == 1
    assert expected_label_name(0) == "non_pothole"
    assert expected_label_name(1) == "pothole"
    assert resolve_manifest_path(
        "datasets/image.jpg",
        project_root,
        manifest_path,
    ) == project_file

    fallback_file = manifest_path.parent / "relative.jpg"
    write_image(fallback_file)
    assert resolve_manifest_path("relative.jpg", project_root, manifest_path) == (
        fallback_file
    )

    with pytest.raises(ValueError, match="Invalid label"):
        parse_binary_label("bad", 4)


def test_rdd_training_main_runs_selected_experiment_runner(monkeypatch, capsys) -> None:
    runner_configs = []

    class FakeRunner:
        def __init__(self, config) -> None:
            runner_configs.append(config)

        def run(self):
            return None

    monkeypatch.setattr(rdd_constants, "RDD_MODEL_MODE", "single")
    monkeypatch.setattr(rdd_constants, "RDD_SINGLE_MODEL", "mobilenet_v3_small")
    monkeypatch.setattr(training_utils, "RDD_MODEL_MODE", "single")
    monkeypatch.setattr(training_utils, "RDD_SINGLE_MODEL", "mobilenet_v3_small")
    monkeypatch.setattr(rdd_constants, "RDD_ACTIVE_EXPERIMENT_IDS", ("C",))
    monkeypatch.setattr(rdd_constants, "RUN_RDD_FULL_IMAGE_PREPROCESSING", False)
    monkeypatch.setattr(rdd_constants, "RUN_RDD_PATCH_PREPROCESSING", False)
    monkeypatch.setattr(
        "src.rdd_benchmark.experiments.RDDExperimentRunner",
        FakeRunner,
    )

    runpy.run_module("src.rdd_benchmark.main", run_name="__main__")
    output = capsys.readouterr().out

    assert "RDD2022 experiment plan" in output
    assert "models: mobilenet_v3_small" in output
    assert "C: C_full_image_weighted_sampler_minority_aug" in output
    assert len(runner_configs) == 1
    assert runner_configs[0].model_names == ("mobilenet_v3_small",)
    assert runner_configs[0].experiment_config.experiment_id == "C"
