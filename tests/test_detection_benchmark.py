import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch

import experiments.detection.detection_benchmark as experiment
import experiments.detection.plot_detection_benchmark as plot_detection
import src.detection_benchmark.classification_benchmark as classification_module
import src.detection_benchmark.classification_inference_adapter as classification_adapter_module
import src.detection_benchmark.object_detection_benchmark as detection_module
import src.detection_benchmark.object_detection_inference_adapter as detection_adapter_module
from src.detection_benchmark.benchmark_result import DetectionBenchmarkResult
from src.detection_benchmark.classification_benchmark import (
    ClassificationBenchmark,
    calculate_macro_classification_metrics,
)
from src.detection_benchmark.classification_dataset import (
    ClassificationSample,
    load_classification_manifest,
)
from src.detection_benchmark.classification_inference_adapter import (
    ClassificationInferenceAdapter,
)
from src.detection_benchmark.object_detection_benchmark import (
    GroundTruthBox,
    box_iou,
    build_category_mapping,
    calculate_coco_map,
    match_detections,
)
from src.detection_benchmark.object_detection_inference_adapter import (
    ObjectDetectionInferenceAdapter,
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


class FakeClassificationModel:
    def __init__(self, logits: torch.Tensor | None = None) -> None:
        self.logits = logits if logits is not None else torch.tensor([[1.0, 2.0]])
        self.device = None
        self.eval_called = False
        self.calls = []
        self.config = SimpleNamespace(id2label={0: "zero", 1: "one"})

    def to(self, device: torch.device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.logits


class FakeMobileVitModel(FakeClassificationModel):
    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(logits=torch.tensor([[3.0, 4.0]]))


class FakeWeights:
    def transforms(self):
        return lambda image: torch.ones(3, 2, 2)


class FakeProcessor:
    def __call__(self, **kwargs):
        assert kwargs["return_tensors"] == "pt"
        return {"pixel_values": torch.ones(1, 3, 2, 2)}


def install_fake_classifier_modules(monkeypatch) -> None:
    torchvision = ModuleType("torchvision")
    torchvision_models = ModuleType("torchvision.models")
    torchvision_models.MobileNet_V2_Weights = SimpleNamespace(DEFAULT=FakeWeights())
    torchvision_models.MobileNet_V3_Small_Weights = SimpleNamespace(
        DEFAULT=FakeWeights()
    )
    torchvision_models.MobileNet_V3_Large_Weights = SimpleNamespace(
        DEFAULT=FakeWeights()
    )
    torchvision_models.EfficientNet_B0_Weights = SimpleNamespace(DEFAULT=FakeWeights())
    torchvision_models.ResNet18_Weights = SimpleNamespace(DEFAULT=FakeWeights())
    torchvision_models.Inception_V3_Weights = SimpleNamespace(DEFAULT=FakeWeights())
    torchvision.models = torchvision_models
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", torchvision_models)

    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeProcessor()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    timm = ModuleType("timm")
    timm.data = SimpleNamespace(
        resolve_model_data_config=lambda model: {"input_size": (3, 224, 224)},
        create_transform=lambda **kwargs: lambda image: torch.ones(3, 2, 2),
    )
    monkeypatch.setitem(sys.modules, "timm", timm)


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


def test_classification_inference_adapter_routes_all_supported_families(
    monkeypatch,
    tmp_path,
) -> None:
    install_fake_classifier_modules(monkeypatch)
    monkeypatch.setattr(
        classification_adapter_module,
        "load_rgb_image",
        lambda image_path: object(),
    )
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    for model_name in (
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "resnet18",
        "inception_v3",
        "efficientformer_l1",
    ):
        model = FakeClassificationModel()
        adapter = ClassificationInferenceAdapter(
            model_name,
            model,
            torch.device("cpu"),
        )
        logits = adapter.predict(image_path)

        assert logits.tolist() == [1.0, 2.0]
        assert model.device == torch.device("cpu")
        assert model.eval_called is True
        assert model.calls

    mobilevit_model = FakeMobileVitModel()
    adapter = ClassificationInferenceAdapter(
        "mobilevit_xxs",
        mobilevit_model,
        torch.device("cpu"),
    )
    assert adapter.predict(image_path).tolist() == [3.0, 4.0]
    assert mobilevit_model.calls


def test_classification_inference_adapter_validation_and_shape_errors(
    monkeypatch,
    tmp_path,
) -> None:
    install_fake_classifier_modules(monkeypatch)
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    with pytest.raises(ValueError, match="not an image classifier"):
        ClassificationInferenceAdapter("yolov8n", object(), torch.device("cpu"))

    model = FakeClassificationModel(torch.ones(1, 2, 3))
    adapter = ClassificationInferenceAdapter(
        "mobilenet_v2",
        model,
        torch.device("cpu"),
    )
    monkeypatch.setattr(
        classification_adapter_module,
        "load_rgb_image",
        lambda image_path: object(),
    )
    with pytest.raises(ValueError, match="Expected one logits vector"):
        adapter.predict(image_path)

    adapter = ClassificationInferenceAdapter.__new__(ClassificationInferenceAdapter)
    adapter.transform = None
    with pytest.raises(RuntimeError, match="transform"):
        adapter._predict_transformed_tensor(image_path)

    adapter.processor = None
    with pytest.raises(RuntimeError, match="processor"):
        adapter._predict_mobilevit(image_path)


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


def test_object_detection_inference_adapter_normalizes_ultralytics_boxes(
    tmp_path,
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    predict_calls = []

    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeBoxes:
        xyxy = FakeTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        conf = FakeTensor([0.9, 0.8])
        cls = FakeTensor([1.0, 2.0])

    class FakeYolo:
        def __init__(self, boxes) -> None:
            self.boxes = boxes

        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            return [SimpleNamespace(boxes=self.boxes)]

    adapter = ObjectDetectionInferenceAdapter(
        "yolov8n",
        FakeYolo(FakeBoxes()),
        torch.device("cpu"),
    )
    predictions = adapter.predict(image_path)

    assert predict_calls == [
        {
            "source": str(image_path),
            "device": "cpu",
            "conf": detection_adapter_module.PREDICTION_CONFIDENCE_FLOOR,
            "verbose": False,
        }
    ]
    assert predictions == [
        PredictedBox(1, 0.9, (1.0, 2.0, 3.0, 4.0)),
        PredictedBox(2, 0.8, (5.0, 6.0, 7.0, 8.0)),
    ]

    empty_adapter = ObjectDetectionInferenceAdapter(
        "yolov8n",
        FakeYolo(None),
        torch.device("cpu"),
    )
    assert empty_adapter.predict(image_path) == []

    with pytest.raises(ValueError, match="not an object detector"):
        ObjectDetectionInferenceAdapter("mobilenet_v2", object(), torch.device("cpu"))


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

    manifest = tmp_path / "labels.csv"
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    manifest.write_text(f"image_path,class_id\n{image_path.name},0\n", encoding="utf-8")
    appended.clear()

    class FakeClassificationBenchmark:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model"] is loaded_model
            assert kwargs["dataset_name"] == "imagenet"
            assert kwargs["split"] == "validation"
            assert kwargs["samples"] == [ClassificationSample(image_path.resolve(), 0)]

        def run(self) -> DetectionBenchmarkResult:
            return make_result("resnet18")

    monkeypatch.setattr(
        experiment,
        "ClassificationBenchmark",
        FakeClassificationBenchmark,
    )
    experiment.run_model_benchmark(
        model_name="resnet18",
        device=torch.device("cpu"),
        num_images=None,
        classification_labels=manifest,
        classification_dataset_name="imagenet",
        classification_split="validation",
        coco_images_dir=tmp_path,
        coco_annotations=tmp_path / "instances.json",
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    assert appended == [(make_result("resnet18"), experiment.RESULTS_CSV)]

    with pytest.raises(ValueError, match="classification_labels"):
        experiment.run_model_benchmark(
            model_name="resnet18",
            device=torch.device("cpu"),
            num_images=None,
            classification_labels=None,
            classification_dataset_name="imagenet",
            classification_split="validation",
            coco_images_dir=tmp_path,
            coco_annotations=tmp_path / "instances.json",
            confidence_threshold=0.25,
            iou_threshold=0.5,
        )


def test_detection_experiment_subprocesses_and_main_flow(
    monkeypatch,
    tmp_path,
) -> None:
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

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        commands.append((command, check))
        return_code = 1 if command[command.index("--model") + 1] == "bad" else 0
        return subprocess.CompletedProcess(command, returncode=return_code)

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    experiment.run_model_processes(["resnet18"], args)

    assert commands[0][1] is False
    assert commands[0][0][-4:] == [
        "--classification-labels",
        str(args.classification_labels),
        "--num-images",
        "2",
    ] or "--classification-labels" in commands[0][0]

    with pytest.raises(RuntimeError, match="bad"):
        experiment.run_model_processes(["resnet18", "bad"], args)

    output_path = tmp_path / "results.csv"
    output_path.write_text("old", encoding="utf-8")
    main_calls = []
    monkeypatch.setattr(experiment, "RESULTS_CSV", output_path)
    monkeypatch.setattr(experiment, "resolve_device", lambda name: torch.device(name))
    monkeypatch.setattr(
        experiment,
        "run_model_processes",
        lambda model_names, parsed_args: main_calls.append(
            (model_names, parsed_args.device)
        ),
    )
    monkeypatch.setattr(
        experiment,
        "validate_task_arguments",
        lambda model_names, classification_labels, parser: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detection_benchmark.py",
            "--model",
            "yolov8n",
            "resnet18",
            "--device",
            "cpu",
            "-o",
        ],
    )

    experiment.main()

    assert output_path.exists() is False
    assert main_calls == [(["yolov8n", "resnet18"], "cpu")]


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
