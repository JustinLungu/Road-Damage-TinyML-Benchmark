import argparse
import os
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

CNN_DIR = MODELS_DIR / "cnn"
VIT_DIR = MODELS_DIR / "vit"
VLM_DIR = MODELS_DIR / "vlm"

for directory in (CNN_DIR, VIT_DIR, VLM_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_ultralytics_model(checkpoint_name: str) -> Any:
    from ultralytics import YOLO

    checkpoint_path = CNN_DIR / checkpoint_name
    return YOLO(str(checkpoint_path))


def load_mobilenet_v3_small() -> Any:
    os.environ["TORCH_HOME"] = str(CNN_DIR)

    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    return mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT
    ).eval()


def load_mobilenet_v3_large() -> Any:
    os.environ["TORCH_HOME"] = str(CNN_DIR)

    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    return mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.DEFAULT
    ).eval()


def load_mobilevit(model_id: str, local_name: str) -> Any:
    from transformers import AutoModelForImageClassification

    return AutoModelForImageClassification.from_pretrained(
        model_id,
        cache_dir=str(VIT_DIR / local_name),
    ).eval()


def load_efficientformer(model_name: str, local_name: str) -> Any:
    import timm

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


MODEL_LOADERS: dict[str, Callable[[], Any]] = {
    # CNN / object detection
    "yolov5nu": lambda: load_ultralytics_model("yolov5nu.pt"),
    "yolov8n": lambda: load_ultralytics_model("yolov8n.pt"),

    # CNN / image classification
    "mobilenet_v3_small": load_mobilenet_v3_small,
    "mobilenet_v3_large": load_mobilenet_v3_large,

    # Lightweight vision transformers
    "mobilevit_xxs": lambda: load_mobilevit(
        "apple/mobilevit-xx-small",
        "mobilevit_xxs",
    ),
    "mobilevit_xs": lambda: load_mobilevit(
        "apple/mobilevit-x-small",
        "mobilevit_xs",
    ),
    "mobilevit_s": lambda: load_mobilevit(
        "apple/mobilevit-small",
        "mobilevit_s",
    ),
    "efficientformer_l1": lambda: load_efficientformer(
        "efficientformer_l1.snap_dist_in1k",
        "efficientformer_l1",
    ),
    "efficientformer_l3": lambda: load_efficientformer(
        "efficientformer_l3.snap_dist_in1k",
        "efficientformer_l3",
    ),
    "efficientformer_l7": lambda: load_efficientformer(
        "efficientformer_l7.snap_dist_in1k",
        "efficientformer_l7",
    ),

    # Tiny VLMs
    "smolvlm_256m": lambda: load_smolvlm(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        "smolvlm_256m",
    ),
    "smolvlm_500m": lambda: load_smolvlm(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "smolvlm_500m",
    ),
    "smolvlm_2b": lambda: load_smolvlm(
        "HuggingFaceTB/SmolVLM-Instruct",
        "smolvlm_2b",
    ),
}


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
            print(f"✓ Loaded {model_name}")
        except Exception as exc:
            print(f"✗ Failed to load {model_name}")
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
            print(f"✓ {model_name}")

        print(f"\nLoaded {len(models)} models successfully.")

    else:
        model = load_model(args.model)

        print(f"Loaded {args.model}: {type(model).__name__}")

    print(f"CNN models directory: {CNN_DIR}")
    print(f"ViT models directory: {VIT_DIR}")
    print(f"VLM models directory: {VLM_DIR}")


if __name__ == "__main__":
    main()