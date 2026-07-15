import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

import src.load_model as load_model_module
from src.constants import (
    CUSTOM_IMAGE_CLASSIFICATION_MODELS,
    EFFICIENTNET_MODEL_CHECKPOINTS,
    EFFICIENTFORMER_MODEL_IDS,
    INCEPTION_MODEL_CHECKPOINTS,
    MOBILENET_MODEL_CHECKPOINTS,
    MOBILEVIT_MODEL_IDS,
    RESNET_MODEL_CHECKPOINTS,
    SHUFFLENET_MODEL_CHECKPOINTS,
    SMOLVLM_MODEL_IDS,
    YOLO_MODEL_CHECKPOINTS,
)
from src.performance_benchmark.model_inference_adapter import ModelInferenceAdapter


class FakeModel:
    def __init__(self, name: str = "model") -> None:
        self.name = name
        self.calls = []
        self.device = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeWeights:
    def transforms(self):
        return lambda image: torch.ones(3, 2, 2)


class FakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt: bool):
        assert add_generation_prompt is True
        assert messages[0]["role"] == "user"
        return "Describe the image briefly."

    def __call__(self, **kwargs):
        assert kwargs["return_tensors"] == "pt"
        return {"input_ids": torch.ones(1)}


def install_fake_model_modules(monkeypatch) -> None:
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = lambda checkpoint: FakeModel(checkpoint)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

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
    torchvision_models.ShuffleNet_V2_X0_5_Weights = SimpleNamespace(
        DEFAULT=FakeWeights()
    )
    torchvision_models.Inception_V3_Weights = SimpleNamespace(DEFAULT=FakeWeights())
    torchvision_models.mobilenet_v2 = lambda weights: FakeModel("v2")
    torchvision_models.mobilenet_v3_small = lambda weights: FakeModel("small")
    torchvision_models.mobilenet_v3_large = lambda weights: FakeModel("large")
    torchvision_models.efficientnet_b0 = lambda weights: FakeModel("efficientnet_b0")
    torchvision_models.resnet18 = lambda weights: FakeModel("resnet18")
    torchvision_models.shufflenet_v2_x0_5 = lambda weights: FakeModel(
        "shufflenet_v2_x0_5"
    )
    torchvision_models.inception_v3 = lambda weights: FakeModel("inception_v3")
    torchvision.models = torchvision_models
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", torchvision_models)

    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeProcessor()
    )
    transformers.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeProcessor()
    )
    transformers.AutoModelForImageClassification = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeModel("mobilevit")
    )
    transformers.AutoModelForImageTextToText = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeModel("smolvlm")
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    timm = ModuleType("timm")
    timm.create_model = lambda *args, **kwargs: FakeModel("efficientformer")
    timm.data = SimpleNamespace(
        resolve_model_data_config=lambda model: {"input_size": (3, 224, 224)},
        create_transform=lambda **kwargs: lambda image: torch.ones(3, 2, 2),
    )
    monkeypatch.setitem(sys.modules, "timm", timm)


def make_image(path: Path) -> None:
    Image.new("RGB", (2, 2), color="white").save(path)


def test_all_registered_models_route_through_load_model(monkeypatch) -> None:
    install_fake_model_modules(monkeypatch)

    expected_model_names = (
        set(YOLO_MODEL_CHECKPOINTS)
        | set(MOBILENET_MODEL_CHECKPOINTS)
        | set(EFFICIENTNET_MODEL_CHECKPOINTS)
        | set(RESNET_MODEL_CHECKPOINTS)
        | set(SHUFFLENET_MODEL_CHECKPOINTS)
        | set(INCEPTION_MODEL_CHECKPOINTS)
        | set(MOBILEVIT_MODEL_IDS)
        | set(EFFICIENTFORMER_MODEL_IDS)
        | set(CUSTOM_IMAGE_CLASSIFICATION_MODELS)
        | set(SMOLVLM_MODEL_IDS)
    )

    assert set(load_model_module.MODEL_LOADERS) == expected_model_names

    for model_name in sorted(expected_model_names):
        model = load_model_module.load_model(model_name)
        if model_name in CUSTOM_IMAGE_CLASSIFICATION_MODELS:
            assert isinstance(model, torch.nn.Module)
        else:
            assert isinstance(model, FakeModel)


def test_model_registry_cache_detection_and_cli(monkeypatch, tmp_path, capsys) -> None:
    loaded_model = object()
    monkeypatch.setitem(
        load_model_module.MODEL_LOADERS, "fake_model", lambda: loaded_model
    )
    assert load_model_module.load_model("fake_model") is loaded_model
    with pytest.raises(ValueError, match="Unknown model"):
        load_model_module.load_model("missing_model")

    checkpoint_path = tmp_path / "model.pt"
    monkeypatch.setitem(
        load_model_module.MODEL_CHECKPOINT_PATHS,
        "checkpoint_model",
        checkpoint_path,
    )
    assert load_model_module.is_model_downloaded("checkpoint_model") is False
    checkpoint_path.write_bytes(b"weights")
    assert load_model_module.is_model_downloaded("checkpoint_model") is True

    cache_dir = tmp_path / "cache"
    (cache_dir / ".no_exist" / "model").mkdir(parents=True)
    (cache_dir / ".no_exist" / "model" / "ignored.safetensors").write_bytes(b"fake")
    monkeypatch.delitem(
        load_model_module.MODEL_CHECKPOINT_PATHS,
        "cache_model",
        raising=False,
    )
    monkeypatch.setitem(load_model_module.MODEL_CACHE_DIRS, "cache_model", cache_dir)
    assert load_model_module.is_model_downloaded("cache_model") is False
    (cache_dir / "snapshots" / "123").mkdir(parents=True)
    (cache_dir / "snapshots" / "123" / "model.safetensors").write_bytes(b"weights")
    assert load_model_module.is_model_downloaded("cache_model") is True
    assert load_model_module.is_model_downloaded("unknown_model") is False

    monkeypatch.setattr(
        load_model_module,
        "MODEL_LOADERS",
        {"missing_model": lambda: None, "downloaded_model": lambda: None},
    )
    monkeypatch.setattr(
        load_model_module,
        "is_model_downloaded",
        lambda model_name: model_name == "downloaded_model",
    )
    assert load_model_module.get_downloaded_model_names() == ["downloaded_model"]

    def fail_loader():
        raise RuntimeError("broken")

    monkeypatch.setattr(
        load_model_module,
        "MODEL_LOADERS",
        {"ok_model": lambda: "loaded", "bad_model": fail_loader},
    )
    assert load_model_module.load_all_models() == {"ok_model": "loaded"}
    assert "Failed to load bad_model" in capsys.readouterr().out

    class FakeCliModel:
        pass

    monkeypatch.setattr(
        load_model_module, "MODEL_LOADERS", {"fake_model": FakeCliModel}
    )
    monkeypatch.setattr(
        load_model_module, "load_model", lambda model_name: FakeCliModel()
    )
    monkeypatch.setattr(sys, "argv", ["load_model.py", "--model", "fake_model"])
    load_model_module.main()
    assert "Loaded fake_model: FakeCliModel" in capsys.readouterr().out

    monkeypatch.setattr(
        load_model_module, "load_all_models", lambda: {"first": object()}
    )
    monkeypatch.setattr(sys, "argv", ["load_model.py", "--model", "all"])
    load_model_module.main()
    assert "Loaded 1 models successfully" in capsys.readouterr().out


def test_loader_functions_use_external_factories(monkeypatch) -> None:
    install_fake_model_modules(monkeypatch)

    assert load_model_module.load_ultralytics_model("yolov5nu.pt").name.endswith(
        "yolov5nu.pt"
    )
    assert load_model_module.load_mobilenet_v2().eval_called is True
    assert load_model_module.load_mobilenet_v3_small().eval_called is True
    assert load_model_module.load_mobilenet_v3_large().eval_called is True
    assert load_model_module.load_efficientnet_b0().eval_called is True
    assert load_model_module.load_resnet18().eval_called is True
    assert load_model_module.load_shufflenet_v2_x0_5().eval_called is True
    assert load_model_module.load_inception_v3().eval_called is True
    assert isinstance(load_model_module.load_tiny_cnn(), torch.nn.Module)
    assert isinstance(load_model_module.load_resnet8(), torch.nn.Module)
    assert isinstance(load_model_module.load_ds_cnn_small(), torch.nn.Module)
    assert isinstance(load_model_module.load_mobilenet_v1_025(), torch.nn.Module)
    assert (
        load_model_module.load_mobilevit("apple/mobilevit-small", "mobilevit_s").name
        == "mobilevit"
    )
    assert (
        load_model_module.load_efficientformer("efficientformer_l1", "ef_l1").name
        == "efficientformer"
    )
    assert (
        load_model_module.load_smolvlm("HuggingFaceTB/SmolVLM-Instruct", "smolvlm").name
        == "smolvlm"
    )


def test_adapter_prepare_paths_run_without_real_models(monkeypatch, tmp_path) -> None:
    install_fake_model_modules(monkeypatch)
    image_path = tmp_path / "image.jpg"
    make_image(image_path)

    yolo_calls = []

    class FakeYolo:
        def predict(self, **kwargs) -> None:
            yolo_calls.append(kwargs)

    yolo = ModelInferenceAdapter("yolov5nu", FakeYolo(), torch.device("cpu"))
    yolo.infer(image_path)
    assert yolo_calls == [
        {"source": str(image_path), "device": "cpu", "verbose": False}
    ]

    for model_name in (
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "resnet18",
        "inception_v3",
        "mobilevit_xxs",
        "efficientformer_l1",
        "smolvlm_256m",
    ):
        fake_model = FakeModel(model_name)
        adapter = ModelInferenceAdapter(model_name, fake_model, torch.device("cpu"))
        adapter.infer(image_path)
        assert fake_model.device == torch.device("cpu")
        assert fake_model.eval_called is True
        assert fake_model.calls

    with pytest.raises(ValueError, match="No inference adapter"):
        ModelInferenceAdapter("unknown", object(), torch.device("cpu"))


def test_adapter_private_infer_helpers_validate_initialization(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    make_image(image_path)

    adapter = ModelInferenceAdapter.__new__(ModelInferenceAdapter)
    adapter.transform = None
    with pytest.raises(RuntimeError, match="transform"):
        adapter._infer_transformed_tensor(image_path)

    adapter.processor = None
    with pytest.raises(RuntimeError, match="processor"):
        adapter._infer_mobilevit(image_path)

    adapter.prompt = None
    with pytest.raises(RuntimeError, match="SmolVLM processor"):
        adapter._infer_smolvlm(image_path)
