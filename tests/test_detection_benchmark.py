import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch

import experiments.detection.detection_benchmark as experiment
import experiments.detection.plot_detection_benchmark as plot_detection
import src.detection_benchmark.classification_benchmark as classification_module
import src.detection_benchmark.object_detection_benchmark as detection_module
from src.detection_benchmark.benchmark_result import DetectionBenchmarkResult
from src.detection_benchmark.classification_benchmark import (
    ClassificationBenchmark,
    calculate_macro_classification_metrics,
)
from src.detection_benchmark.classification_dataset import (
    ClassificationSample,
    load_classification_manifest,
)
from src.detection_benchmark.object_detection_benchmark import (
    GroundTruthBox,
    box_iou,
    build_category_mapping,
    calculate_coco_map,
    match_detections,
)
from src.detection_benchmark.object_detection_inference_adapter import PredictedBox
from src.detection_benchmark.utils import append_result_csv


def make_result(model_name: str = "model") -> DetectionBenchmarkResult:
    return DetectionBenchmarkResult(
        model_name=model_name,
        task="image_classification",
        dataset_name="test",
        split="validation",
        num_images=2,
        confidence_threshold=None,
        iou_threshold=None,
        map_50_95=None,
        map_50=None,
        map_75=None,
        precision=0.5,
        recall=0.5,
        f1_score=0.5,
        mean_iou=None,
        top1_accuracy=0.5,
        top5_accuracy=1.0,
    )


def write_tiny_coco_dataset(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "image.jpg").write_bytes(b"image")

    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [
                    {
                        "id": 1,
                        "file_name": "image.jpg",
                        "width": 100,
                        "height": 100,
                    }
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "object",
                        "supercategory": "object",
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return images_dir, annotation_path


def test_classification_manifest_metrics_benchmark_and_csv(
    monkeypatch,
    tmp_path,
) -> None:
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "image_path,class_id\nfirst.jpg,0\nsecond.jpg,1\n",
        encoding="utf-8",
    )

    samples = load_classification_manifest(manifest)
    assert samples == [
        ClassificationSample(first_image.resolve(), 0),
        ClassificationSample(second_image.resolve(), 1),
    ]
    assert load_classification_manifest(manifest, num_images=1) == samples[:1]

    precision, recall, f1_score = calculate_macro_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 2],
    )
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.5)
    assert f1_score == pytest.approx((2 / 3 + 0.5) / 2)

    class FakeAdapter:
        def __init__(self, model_name, model, device) -> None:
            self.predictions = iter(
                [
                    torch.tensor([3.0, 2.0, 1.0]),
                    torch.tensor([3.0, 2.0, 1.0]),
                ]
            )

        def predict(self, image_path: Path) -> torch.Tensor:
            return next(self.predictions)

    monkeypatch.setattr(
        classification_module,
        "ClassificationInferenceAdapter",
        FakeAdapter,
    )
    result = ClassificationBenchmark(
        model_name="mobilenet_v2",
        model=object(),
        samples=samples,
        device=torch.device("cpu"),
        dataset_name="tiny",
        split="validation",
    ).run()

    assert result.top1_accuracy == 0.5
    assert result.top5_accuracy == 1.0
    assert result.precision == pytest.approx(0.25)
    assert result.recall == pytest.approx(0.5)
    assert result.f1_score == pytest.approx(1 / 3)

    csv_path = tmp_path / "results" / "metrics.csv"
    append_result_csv(result, csv_path)
    append_result_csv(make_result("second"), csv_path)
    with csv_path.open(newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))
    assert [row["model_name"] for row in rows] == ["mobilenet_v2", "second"]
    assert list(rows[0]) == list(asdict(result))


def test_classification_manifest_rejects_invalid_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_classification_manifest(tmp_path / "missing.csv")

    invalid_manifest = tmp_path / "invalid.csv"
    invalid_manifest.write_text("path,label\nimage.jpg,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image_path and class_id"):
        load_classification_manifest(invalid_manifest)

    empty_manifest = tmp_path / "empty.csv"
    empty_manifest.write_text("image_path,class_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_classification_manifest(empty_manifest)


def test_detection_matching_iou_mapping_and_coco_map(tmp_path) -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert build_category_mapping(
        {0: "person", 1: "car"},
        [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
    ) == {0: 1, 1: 3}
    with pytest.raises(ValueError, match="do not match"):
        build_category_mapping({0: "unknown"}, [{"id": 1, "name": "person"}])

    ground_truth = {
        1: [
            GroundTruthBox(1, (0, 0, 10, 10), False),
            GroundTruthBox(1, (20, 20, 30, 30), True),
        ]
    }
    predictions = {
        1: [
            (1, PredictedBox(0, 0.9, (0, 0, 10, 10))),
            (1, PredictedBox(0, 0.8, (0, 0, 10, 10))),
            (1, PredictedBox(0, 0.7, (20, 20, 30, 30))),
            (2, PredictedBox(1, 0.6, (40, 40, 50, 50))),
            (1, PredictedBox(0, 0.1, (40, 40, 50, 50))),
        ]
    }
    counts = match_detections(ground_truth, predictions, 0.25, 0.5)
    assert counts.true_positive == 1
    assert counts.false_positive == 2
    assert counts.false_negative == 0
    assert counts.matched_ious == (1.0,)

    _, annotation_path = write_tiny_coco_dataset(tmp_path)
    map_50_95, map_50, map_75 = calculate_coco_map(
        annotation_path,
        [1],
        [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "score": 0.99,
            }
        ],
    )
    assert map_50_95 == pytest.approx(1.0)
    assert map_50 == pytest.approx(1.0)
    assert map_75 == pytest.approx(1.0)


def test_object_detection_benchmark_runs_with_fake_predictions(
    monkeypatch,
    tmp_path,
) -> None:
    images_dir, annotation_path = write_tiny_coco_dataset(tmp_path)

    class FakeModel:
        names = {0: "object"}

    class FakeAdapter:
        def __init__(self, model_name, model, device) -> None:
            pass

        def predict(self, image_path: Path) -> list[PredictedBox]:
            return [PredictedBox(0, 0.9, (10, 10, 30, 30))]

    monkeypatch.setattr(
        detection_module,
        "ObjectDetectionInferenceAdapter",
        FakeAdapter,
    )
    result = detection_module.ObjectDetectionBenchmark(
        model_name="yolov8n",
        model=FakeModel(),
        images_dir=images_dir,
        annotation_path=annotation_path,
        device=torch.device("cpu"),
        confidence_threshold=0.25,
        iou_threshold=0.5,
    ).run()

    assert result.task == "object_detection"
    assert result.num_images == 1
    assert result.map_50_95 == pytest.approx(1.0)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0
    assert result.mean_iou == 1.0


def test_detection_experiment_model_resolution_validation_and_routing(
    monkeypatch,
    tmp_path,
) -> None:
    parser = argparse.ArgumentParser()
    assert experiment.resolve_model_names(
        ["yolov8n", "yolov8n", "resnet18"],
        parser,
    ) == ["yolov8n", "resnet18"]

    monkeypatch.setattr(
        experiment,
        "get_downloaded_model_names",
        lambda: ["smolvlm_2b", "yolov8n", "resnet18"],
    )
    assert experiment.resolve_model_names(
        [experiment.ALL_LOADED],
        parser,
    ) == ["yolov8n", "resnet18"]
    with pytest.raises(SystemExit):
        experiment.validate_task_arguments(["resnet18"], None, parser)
    experiment.validate_task_arguments(["yolov8n"], None, parser)

    loaded_model = object()
    appended = []

    class FakeDetectionBenchmark:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model"] is loaded_model
            assert kwargs["num_images"] == 3

        def run(self) -> DetectionBenchmarkResult:
            return make_result("yolov8n")

    monkeypatch.setattr(experiment, "load_model", lambda model_name: loaded_model)
    monkeypatch.setattr(
        experiment,
        "ObjectDetectionBenchmark",
        FakeDetectionBenchmark,
    )
    monkeypatch.setattr(
        experiment,
        "append_result_csv",
        lambda result, path: appended.append((result, path)),
    )
    experiment.run_model_benchmark(
        model_name="yolov8n",
        device=torch.device("cpu"),
        num_images=3,
        classification_labels=None,
        classification_dataset_name="imagenet",
        classification_split="validation",
        coco_images_dir=tmp_path,
        coco_annotations=tmp_path / "instances.json",
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    assert appended == [(make_result("yolov8n"), experiment.RESULTS_CSV)]


def test_detection_plotting_creates_task_and_common_metric_plots(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    csv_path = tmp_path / "detection_benchmark_results.csv"
    pd.DataFrame(
        {
            "model_name": [
                "yolov8n",
                "mobilenet_v2",
                "mobilenet_v2",
            ],
            "task": [
                "object_detection",
                "image_classification",
                "image_classification",
            ],
            "map_50_95": [0.4, None, None],
            "map_50": [0.5, None, None],
            "map_75": [0.3, None, None],
            "precision": [0.7, 0.8, 0.9],
            "recall": [0.5, 0.6, 0.7],
            "f1_score": [0.58, 0.68, 0.78],
            "mean_iou": [0.85, None, None],
            "top1_accuracy": [None, 0.65, 0.7],
            "top5_accuracy": [None, 0.9, 0.92],
        }
    ).to_csv(csv_path, index=False)

    task_plots = []
    common_plots = []
    real_plot_bar_chart = plot_detection.plot_bar_chart

    def fake_task_plot(plot_data, metric_column, task_name, output_path) -> None:
        task_plots.append(
            (plot_data.copy(), metric_column, task_name, output_path)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("task plot", encoding="utf-8")

    def fake_common_plot(plot_data, metric_column, output_path) -> None:
        common_plots.append((plot_data.copy(), metric_column, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("common plot", encoding="utf-8")

    monkeypatch.setattr(plot_detection, "plot_task_metric", fake_task_plot)
    monkeypatch.setattr(plot_detection, "plot_common_metric", fake_common_plot)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_detection_benchmark.py", "--csv", str(csv_path)],
    )

    plot_detection.main()
    output_lines = capsys.readouterr().out.splitlines()

    assert "Created 15 plots:" in output_lines
    assert len(task_plots) == 12
    assert len(common_plots) == 3
    assert {plot[3].parent.name for plot in task_plots} == {
        "image_classification",
        "object_detection",
    }
    assert {plot[2].parent.name for plot in common_plots} == {"common_metrics"}
    assert task_plots[0][0]["model_name"].tolist() == [
        "mobilenet_v2 #1",
        "mobilenet_v2 #2",
    ]
    assert common_plots[0][0]["task"].tolist() == [
        "object_detection",
        "image_classification",
        "image_classification",
    ]
    assert plot_detection.format_metric_name("map_50_95") == "mAP 50-95"
    assert plot_detection.format_metric_name("top1_accuracy") == "Top-1 Accuracy"

    real_output = tmp_path / "real" / "metric_bar.png"
    real_plot_bar_chart(
        plot_data=pd.DataFrame(
            {
                "model_name": ["model_a", "model_b"],
                "f1_score": [0.5, 0.75],
            }
        ),
        metric_column="f1_score",
        output_path=real_output,
        colors=["#4C78A8", "#F58518"],
        title="F1 Score",
    )
    assert real_output.is_file()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        plot_detection.create_detection_plots(tmp_path / "missing.csv")

    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"model_name": ["model"]}).to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="task"):
        plot_detection.create_detection_plots(invalid_csv)

    stale_plot = tmp_path / "stale" / "old_bar.png"
    stale_plot.parent.mkdir()
    stale_plot.write_text("old", encoding="utf-8")
    plot_detection.remove_stale_plots(stale_plot.parent)
    assert stale_plot.exists() is False
