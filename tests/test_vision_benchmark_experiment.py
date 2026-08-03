import argparse
import subprocess
import sys
import pytest
import torch

import experiments.vision.vision_benchmark as experiment
from src.vision_benchmark.classification_dataset import ClassificationSample
from src.vision_benchmark.results import (
    ClassificationBenchmarkResult,
    ObjectDetectionBenchmarkResult,
)


def classification_result() -> ClassificationBenchmarkResult:
    return ClassificationBenchmarkResult(
        "resnet18", "imagenette", "validation", 1, 0.5, 0.5, 0.5, 0.5, 1.0
    )


def detection_result() -> ObjectDetectionBenchmarkResult:
    return ObjectDetectionBenchmarkResult(
        "yolov8n",
        "coco",
        "validation",
        1,
        0.25,
        0.5,
        0.4,
        0.5,
        0.3,
        0.7,
        0.6,
        0.64,
        0.8,
    )


def test_model_selection_excludes_untrained_custom_models(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    assert experiment.resolve_model_names(
        ["yolov8n", "yolov8n", "resnet18"], parser
    ) == ["yolov8n", "resnet18"]

    monkeypatch.setattr(
        experiment,
        "get_downloaded_model_names",
        lambda: ["tiny_cnn", "smolvlm_2b", "yolov8n", "resnet18"],
    )
    assert experiment.resolve_model_names([experiment.ALL_LOADED], parser) == [
        "yolov8n",
        "resnet18",
    ]


def test_task_validation_requires_classification_manifest(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        experiment.validate_task_arguments(["resnet18"], None, parser)
    experiment.validate_task_arguments(["yolov8n"], None, parser)

    manifest = tmp_path / "labels.csv"
    manifest.write_text("image_path,class_id\n", encoding="utf-8")
    experiment.validate_task_arguments(["resnet18"], manifest, parser)


def test_routes_each_task_to_its_own_result_file(monkeypatch, tmp_path) -> None:
    loaded_model = object()
    saved = []
    monkeypatch.setattr(experiment, "load_model", lambda name: loaded_model)
    monkeypatch.setattr(
        experiment,
        "save_result_csv",
        lambda result, path, keys: saved.append((result, path, keys)),
    )

    class FakeDetector:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model"] is loaded_model

        def run(self):
            return detection_result()

    monkeypatch.setattr(experiment, "ObjectDetectionBenchmark", FakeDetector)
    experiment.run_model_benchmark(
        "yolov8n",
        torch.device("cpu"),
        1,
        None,
        "imagenette",
        "validation",
        tmp_path,
        tmp_path / "instances.json",
        0.25,
        0.5,
    )
    assert saved[-1] == (
        detection_result(),
        experiment.OBJECT_DETECTION_RESULTS_CSV,
        experiment.OBJECT_DETECTION_RESULT_KEYS,
    )

    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    manifest = tmp_path / "labels.csv"
    manifest.write_text("image_path,class_id\nimage.jpg,0\n", encoding="utf-8")

    class FakeClassifier:
        def __init__(self, **kwargs) -> None:
            assert kwargs["samples"] == [ClassificationSample(image.resolve(), 0)]

        def run(self):
            return classification_result()

    monkeypatch.setattr(experiment, "ClassificationBenchmark", FakeClassifier)
    experiment.run_model_benchmark(
        "resnet18",
        torch.device("cpu"),
        None,
        manifest,
        "imagenette",
        "validation",
        tmp_path,
        tmp_path / "instances.json",
        0.25,
        0.5,
    )
    assert saved[-1] == (
        classification_result(),
        experiment.CLASSIFICATION_RESULTS_CSV,
        experiment.CLASSIFICATION_RESULT_KEYS,
    )


def test_multiple_models_run_in_subprocesses(monkeypatch, tmp_path) -> None:
    args = argparse.Namespace(
        device="cpu",
        confidence_threshold=0.25,
        iou_threshold=0.5,
        coco_images_dir=tmp_path / "coco",
        coco_annotations=tmp_path / "instances.json",
        classification_dataset_name="imagenette",
        classification_split="validation",
        classification_labels=tmp_path / "labels.csv",
        num_images=2,
    )
    commands = []

    def fake_run(command, check):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    experiment.run_model_processes(["resnet18", "yolov8n"], args)
    assert len(commands) == 2
    assert all("--num-images" in command for command in commands)


def test_overwrite_clears_both_result_files(monkeypatch, tmp_path) -> None:
    paths = [tmp_path / "classification.csv", tmp_path / "detection.csv"]
    for path in paths:
        path.write_text("old", encoding="utf-8")

    monkeypatch.setattr(experiment, "CLASSIFICATION_RESULTS_CSV", paths[0])
    monkeypatch.setattr(experiment, "OBJECT_DETECTION_RESULTS_CSV", paths[1])
    monkeypatch.setattr(experiment, "validate_task_arguments", lambda *args: None)
    monkeypatch.setattr(experiment, "run_model_processes", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["vision_benchmark.py", "--model", "yolov8n", "resnet18", "-o"],
    )

    experiment.main()
    assert not any(path.exists() for path in paths)
