import runpy
from pathlib import Path

import pytest
import torch
from PIL import Image

import src.rdd_training.constants as rdd_constants
import src.rdd_training.dataset as dataset_module
import src.rdd_training.utils as rdd_utils
from src.rdd_training.dataset import (
    BinaryPotholeDataset,
    BinaryPotholeManifest,
    BinaryPotholeSample,
)
from src.rdd_training.utils import (
    expected_label_name,
    parse_binary_label,
    resolve_manifest_path,
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


def test_rdd_training_main_runs_dataset_and_adapter_smoke(monkeypatch, capsys) -> None:
    class FakeManifest:
        def __init__(self, split: str) -> None:
            self.split = split

        @classmethod
        def from_csv(cls, manifest_path, expected_split):
            assert manifest_path.name == f"{expected_split}.csv"
            return cls(expected_split)

        def class_counts(self):
            return {0: 3, 1: 2}

        def positive_fraction(self):
            return 0.4

        def country_counts(self):
            return {"India": 5}

    class FakeDataset:
        def __init__(self, manifest) -> None:
            self.manifest = manifest

        def __len__(self):
            return 5

        def __getitem__(self, index):
            assert index == 0
            return {
                "image": Image.new("RGB", (2, 2)),
                "label": 1,
                "label_name": "pothole",
                "country": "India",
            }

    class FakeModel:
        pass

    adapted_calls = []

    def fake_load_and_adapt_model_for_binary_pothole(model_name):
        adapted_calls.append(model_name)
        return FakeModel()

    monkeypatch.setattr(dataset_module, "BinaryPotholeManifest", FakeManifest)
    monkeypatch.setattr(dataset_module, "BinaryPotholeDataset", FakeDataset)
    monkeypatch.setattr(
        rdd_utils,
        "load_and_adapt_model_for_binary_pothole",
        fake_load_and_adapt_model_for_binary_pothole,
    )
    monkeypatch.setattr(rdd_constants, "RUN_RDD_TRAINING", False)
    monkeypatch.setattr(rdd_constants, "RUN_RDD_EVALUATION", False)

    runpy.run_module("src.rdd_training.main", run_name="__main__")
    output = capsys.readouterr().out

    assert "RDD2022 binary pothole dataset loader" in output
    assert "train:" in output
    assert "validation:" in output
    assert "test:" in output
    assert "pothole_fraction: 0.400" in output
    assert "Model adaptation smoke test" in output
    assert "adapted_model: mobilenet_v3_small -> FakeModel" in output
    assert adapted_calls == ["mobilenet_v3_small"]
