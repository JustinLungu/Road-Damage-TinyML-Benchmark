import argparse
import os
from typing import Any, Callable

from src.constants import (
    CHECKPOINT_PATTERNS,
    CNN_DIR,
    EFFICIENTFORMER_MODEL_IDS,
    MOBILEVIT_MODEL_IDS,
    MODEL_CACHE_DIRS,
    MODEL_CHECKPOINT_PATHS,
    MODEL_STORAGE_DIRS,
    SMOLVLM_MODEL_IDS,
    VIT_DIR,
    VLM_DIR,
    YOLO_MODEL_CHECKPOINTS,
)

# Guarantees that the folders exist before downloading or loading anything
for directory in MODEL_STORAGE_DIRS:
    directory.mkdir(parents=True, exist_ok=True)


def load_ultralytics_model(checkpoint_name: str) -> Any:
    # Heavy ML libraries are imported lazily so simple CLI/help/test paths stay fast.
    from ultralytics import YOLO

    checkpoint_path = CNN_DIR / checkpoint_name
    return YOLO(str(checkpoint_path))


def load_mobilenet_v2() -> Any:
    # Torchvision reads TORCH_HOME when deciding where pretrained weights live.
    os.environ["TORCH_HOME"] = str(CNN_DIR)

    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()


def load_mobilenet_v3_small() -> Any:
    # Torchvision reads TORCH_HOME when deciding where pretrained weights live.
    os.environ["TORCH_HOME"] = str(CNN_DIR)

    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()


def load_mobilenet_v3_large() -> Any:
    # Torchvision reads TORCH_HOME when deciding where pretrained weights live.
    os.environ["TORCH_HOME"] = str(CNN_DIR)

    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    return mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()


def load_efficientnet_b0() -> Any:
    # Torchvision reads TORCH_HOME when deciding where pretrained weights live.
    os.environ["TORCH_HOME"] = str(CNN_DIR)

    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).eval()


def load_mobilevit(model_id: str, local_name: str) -> Any:
    from transformers import AutoModelForImageClassification

    return AutoModelForImageClassification.from_pretrained(
        model_id,
        cache_dir=str(VIT_DIR / local_name),
    ).eval()


def load_efficientformer(model_name: str, local_name: str) -> Any:
    import timm

    # timm may use Hugging Face caches internally for pretrained weights.
    os.environ["HF_HOME"] = str(VIT_DIR / local_name)
    os.environ["TIMM_HOME"] = str(VIT_DIR / local_name)

    return timm.create_model(
        model_name,
        pretrained=True,
    ).eval()


def load_smolvlm(model_id: str, local_name: str) -> Any:
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        model_id,
        cache_dir=str(VLM_DIR / local_name),
    ).eval()


def make_ultralytics_loader(checkpoint_name: str) -> Callable[[], Any]:
    return lambda: load_ultralytics_model(checkpoint_name)


def make_mobilevit_loader(model_name: str, model_id: str) -> Callable[[], Any]:
    return lambda: load_mobilevit(model_id, model_name)


def make_efficientformer_loader(model_name: str, model_id: str) -> Callable[[], Any]:
    return lambda: load_efficientformer(model_id, model_name)


def make_smolvlm_loader(model_name: str, model_id: str) -> Callable[[], Any]:
    return lambda: load_smolvlm(model_id, model_name)


MODEL_LOADERS: dict[str, Callable[[], Any]] = {
    # CNN / object detection
    **{
        model_name: make_ultralytics_loader(checkpoint_name)
        for model_name, checkpoint_name in YOLO_MODEL_CHECKPOINTS.items()
    },
    # CNN / image classification
    "mobilenet_v2": load_mobilenet_v2,
    "mobilenet_v3_small": load_mobilenet_v3_small,
    "mobilenet_v3_large": load_mobilenet_v3_large,
    "efficientnet_b0": load_efficientnet_b0,
    # Lightweight vision transformers
    **{
        model_name: make_mobilevit_loader(model_name, model_id)
        for model_name, model_id in MOBILEVIT_MODEL_IDS.items()
    },
    **{
        model_name: make_efficientformer_loader(model_name, model_id)
        for model_name, model_id in EFFICIENTFORMER_MODEL_IDS.items()
    },
    # Tiny VLMs
    **{
        model_name: make_smolvlm_loader(model_name, model_id)
        for model_name, model_id in SMOLVLM_MODEL_IDS.items()
    },
}


def get_downloaded_model_names() -> list[str]:
    return [
        model_name for model_name in MODEL_LOADERS if is_model_downloaded(model_name)
    ]


def is_model_downloaded(model_name: str) -> bool:
    checkpoint_path = MODEL_CHECKPOINT_PATHS.get(model_name)
    if checkpoint_path is not None:
        return checkpoint_path.is_file()

    cache_dir = MODEL_CACHE_DIRS.get(model_name)
    if cache_dir is None:
        return False

    return any(
        # Hugging Face can create .no_exist cache markers for failed lookups.
        ".no_exist" not in checkpoint_path.parts and checkpoint_path.is_file()
        for pattern in CHECKPOINT_PATTERNS
        for checkpoint_path in cache_dir.rglob(pattern)
    )


# It checks that the name is valid, then calls the correct loader.
def load_model(model_name: str) -> Any:
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unknown model: {model_name}")

    return MODEL_LOADERS[model_name]()


def load_all_models() -> dict[str, Any]:
    loaded_models = {}

    for model_name, loader in MODEL_LOADERS.items():
        try:
            print(f"\nLoading {model_name}...")
            loaded_models[model_name] = loader()
            print(f"[OK] Loaded {model_name}")
        except Exception as exc:
            # If one fails, it does not crash the full script.
            print(f"[FAILED] Failed to load {model_name}")
            print(f"  Error: {exc}")

    return loaded_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a selected pretrained model.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_LOADERS.keys()) + ["all"],
    )

    args = parser.parse_args()

    if args.model == "all":
        models = load_all_models()

        print("\nSummary")
        print("=" * 40)

        for model_name in models:
            print(f"[OK] {model_name}")

        print(f"\nLoaded {len(models)} models successfully.")

    else:
        model = load_model(args.model)

        print(f"Loaded {args.model}: {type(model).__name__}")

    print(f"CNN models directory: {CNN_DIR}")
    print(f"ViT models directory: {VIT_DIR}")
    print(f"VLM models directory: {VLM_DIR}")


if __name__ == "__main__":
    main()
