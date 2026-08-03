from pathlib import Path

import pytest
import torch
from PIL import Image

from src.rdd_benchmark.data_loader.dataset import (
    BinaryPotholeDataset,
    BinaryPotholeManifest,
    BinaryPotholeSample,
)
from src.rdd_benchmark.data_loader.utils import (
    expected_label_name,
    parse_binary_label,
    resolve_manifest_path,
)


def write_image(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (4, 3), color=128).save(path)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "image_path,annotation_path,country,split,label,label_name"
    body = [",".join(row[key] for key in header.split(",")) for row in rows]
    path.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")


def make_row(image: Path, annotation: Path, label: int, split: str = "train"):
    return {
        "image_path": str(image),
        "annotation_path": str(annotation),
        "country": "India",
        "split": split,
        "label": str(label),
        "label_name": expected_label_name(label),
    }


def make_files(tmp_path, name="sample"):
    image = tmp_path / "images" / f"{name}.jpg"
    annotation = tmp_path / "annotations" / f"{name}.xml"
    write_image(image)
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text("<annotation />", encoding="utf-8")
    return image, annotation


def test_manifest_loads_samples_and_counts(tmp_path):
    image_a, annotation_a = make_files(tmp_path, "a")
    image_b, annotation_b = make_files(tmp_path, "b")
    manifest_path = tmp_path / "train.csv"
    write_manifest(
        manifest_path,
        [make_row(image_a, annotation_a, 1), make_row(image_b, annotation_b, 0)],
    )

    manifest = BinaryPotholeManifest.from_csv(manifest_path, expected_split="train")

    assert manifest.samples[0] == BinaryPotholeSample(
        image_a,
        annotation_a,
        1,
        "pothole",
        "India",
        "train",
    )
    assert manifest.class_counts() == {1: 1, 0: 1}


def test_dataset_loads_rgb_image_and_metadata(tmp_path):
    image, annotation = make_files(tmp_path)
    manifest_path = tmp_path / "validation.csv"
    write_manifest(manifest_path, [make_row(image, annotation, 1, "validation")])

    dataset = BinaryPotholeDataset(
        manifest_path,
        expected_split="validation",
        transform=lambda value: torch.ones(3, value.height, value.width),
    )
    item = dataset[0]

    assert item["image"].shape == (3, 3, 4)
    assert item["label"] == 1
    assert item["country"] == "India"


def test_manifest_rejects_invalid_label_and_split(tmp_path):
    image, annotation = make_files(tmp_path)
    manifest_path = tmp_path / "train.csv"
    row = make_row(image, annotation, 1, "test")
    row["label"] = "2"
    write_manifest(manifest_path, [row])

    with pytest.raises(ValueError, match="must be 0 or 1"):
        BinaryPotholeManifest.from_csv(manifest_path)

    row = make_row(image, annotation, 1, "test")
    write_manifest(manifest_path, [row])
    with pytest.raises(ValueError, match="Expected split train"):
        BinaryPotholeManifest.from_csv(manifest_path, expected_split="train")


def test_manifest_helpers_parse_labels_and_paths(tmp_path):
    manifest_path = tmp_path / "manifests" / "train.csv"
    manifest_path.parent.mkdir()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    image = project_root / "datasets" / "image.jpg"
    write_image(image)

    assert parse_binary_label("1", 2) == 1
    assert expected_label_name(0) == "non_pothole"
    assert (
        resolve_manifest_path("datasets/image.jpg", project_root, manifest_path)
        == image
    )
    with pytest.raises(ValueError, match="Invalid label"):
        parse_binary_label("bad", 3)
