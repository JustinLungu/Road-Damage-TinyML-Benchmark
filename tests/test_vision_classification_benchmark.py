import csv
import sys
from dataclasses import asdict, replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

import src.vision_benchmark.classification_benchmark as classification_module
import src.vision_benchmark.classification_inference_adapter as adapter_module
from src.vision_benchmark.classification_benchmark import (
    ClassificationBenchmark,
    calculate_macro_classification_metrics,
)
from src.vision_benchmark.classification_dataset import (
    ClassificationSample,
    load_classification_manifest,
)
from src.vision_benchmark.classification_inference_adapter import (
    ClassificationInferenceAdapter,
)
from src.vision_benchmark.constants import IMAGE_CLASSIFICATION_MODELS
from src.vision_benchmark.results import ClassificationBenchmarkResult
from src.vision_benchmark.utils import save_result_csv


class FakeModel:
    def __init__(self, logits: torch.Tensor | None = None) -> None:
        self.logits = logits if logits is not None else torch.tensor([[1.0, 2.0]])
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.logits


class FakeWeights:
    def transforms(self):
        return lambda image: torch.ones(3, 2, 2)


def install_fake_model_libraries(monkeypatch) -> None:
    torchvision = ModuleType("torchvision")
    torchvision_models = ModuleType("torchvision.models")
    for name in (
        "MobileNet_V2_Weights",
        "MobileNet_V3_Small_Weights",
        "MobileNet_V3_Large_Weights",
        "EfficientNet_B0_Weights",
        "ResNet18_Weights",
        "ShuffleNet_V2_X0_5_Weights",
        "Inception_V3_Weights",
    ):
        setattr(torchvision_models, name, SimpleNamespace(DEFAULT=FakeWeights()))
    torchvision.models = torchvision_models
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", torchvision_models)

    class FakeProcessor:
        def __call__(self, **kwargs):
            return {"pixel_values": torch.ones(1, 3, 2, 2)}

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


def test_manifest_metrics_and_benchmark(monkeypatch, tmp_path) -> None:
    images = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for image in images:
        image.write_bytes(b"image")
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "image_path,class_id\nfirst.jpg,0\nsecond.jpg,1\n",
        encoding="utf-8",
    )

    samples = load_classification_manifest(manifest)
    assert samples == [
        ClassificationSample(images[0].resolve(), 0),
        ClassificationSample(images[1].resolve(), 1),
    ]
    assert load_classification_manifest(manifest, num_images=1) == samples[:1]

    precision, recall, f1_score = calculate_macro_classification_metrics(
        [0, 0, 1, 1], [0, 1, 1, 2]
    )
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.5)
    assert f1_score == pytest.approx((2 / 3 + 0.5) / 2)

    class FakeAdapter:
        def __init__(self, *args) -> None:
            self.predictions = iter(
                [torch.tensor([3.0, 2.0, 1.0]), torch.tensor([3.0, 2.0, 1.0])]
            )

        def predict(self, image_path):
            return next(self.predictions)

    monkeypatch.setattr(
        classification_module, "ClassificationInferenceAdapter", FakeAdapter
    )
    result = ClassificationBenchmark(
        "mobilenet_v2", object(), samples, torch.device("cpu"), "tiny", "validation"
    ).run()

    assert result.top1_accuracy == 0.5
    assert result.top5_accuracy == 1.0
    assert result.f1_score == pytest.approx(1 / 3)


def test_result_csv_replaces_matching_run(tmp_path) -> None:
    first = ClassificationBenchmarkResult(
        "mobilenet_v2", "imagenette", "validation", 10, 0.5, 0.5, 0.5, 0.5, 1.0
    )
    second = replace(first, model_name="resnet18")
    updated = replace(first, top1_accuracy=0.75)
    output_path = tmp_path / "classification.csv"
    keys = ("model_name", "dataset_name", "split", "num_images")

    save_result_csv(first, output_path, keys)
    save_result_csv(second, output_path, keys)
    save_result_csv(updated, output_path, keys)

    with output_path.open(newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(rows) == 2
    assert rows[-1]["model_name"] == "mobilenet_v2"
    assert float(rows[-1]["top1_accuracy"]) == 0.75
    assert list(rows[-1]) == list(asdict(first))


def test_manifest_rejects_invalid_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_classification_manifest(tmp_path / "missing.csv")

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("path,label\nimage.jpg,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image_path and class_id"):
        load_classification_manifest(invalid)

    empty = tmp_path / "empty.csv"
    empty.write_text("image_path,class_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_classification_manifest(empty)


def test_adapter_routes_every_pretrained_classifier(monkeypatch, tmp_path) -> None:
    install_fake_model_libraries(monkeypatch)
    monkeypatch.setattr(adapter_module, "load_rgb_image", lambda path: object())
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    for model_name in sorted(
        IMAGE_CLASSIFICATION_MODELS - {"mobilevit_xxs", "mobilevit_xs", "mobilevit_s"}
    ):
        model = FakeModel()
        assert ClassificationInferenceAdapter(
            model_name, model, torch.device("cpu")
        ).predict(image_path).tolist() == [1.0, 2.0]

    class FakeMobileVit(FakeModel):
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[3.0, 4.0]]))

    assert ClassificationInferenceAdapter(
        "mobilevit_xxs", FakeMobileVit(), torch.device("cpu")
    ).predict(image_path).tolist() == [3.0, 4.0]


def test_adapter_rejects_detectors_custom_models_and_invalid_logits(
    monkeypatch, tmp_path
) -> None:
    install_fake_model_libraries(monkeypatch)
    assert "tiny_cnn" not in IMAGE_CLASSIFICATION_MODELS
    for model_name in ("yolov8n", "tiny_cnn"):
        with pytest.raises(ValueError, match="not an image classifier"):
            ClassificationInferenceAdapter(model_name, object(), torch.device("cpu"))

    monkeypatch.setattr(adapter_module, "load_rgb_image", lambda path: object())
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    adapter = ClassificationInferenceAdapter(
        "mobilenet_v2", FakeModel(torch.ones(1, 2, 3)), torch.device("cpu")
    )
    with pytest.raises(ValueError, match="Expected one logits vector"):
        adapter.predict(image_path)
