from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.constants import SHUFFLENET_MODEL_CHECKPOINTS  # noqa: E402
from src.load_model import load_model  # noqa: E402
from src.performance_benchmark.utils import load_rgb_image, resolve_device  # noqa: E402


SUPPORTED_MODEL_NAMES = tuple(SHUFFLENET_MODEL_CHECKPOINTS)
MODEL_NAMES = ["shufflenet_v2_x0_5"]

IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"
RANDOM_SEED: int | None = None

DEVICE_NAME = "cpu"
TOP_K = 5

SAVE_RESULTS_JSON = True
SAVE_INPUT_IMAGE = True
OUTPUT_DIR = MODEL_DOCS_DIR / "outputs"


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def select_image_path() -> Path:
    if not USE_RANDOM_IMAGE:
        return resolve_repo_path(IMAGE_PATH)

    image_dir = resolve_repo_path(RANDOM_IMAGE_DIR)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Random image directory does not exist: {image_dir}")

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in: {image_dir}")

    return random.Random(RANDOM_SEED).choice(image_paths)


def validate_config() -> None:
    if not MODEL_NAMES:
        raise ValueError("MODEL_NAMES must contain at least one model name.")
    if TOP_K <= 0:
        raise ValueError(f"TOP_K must be greater than zero, got {TOP_K}.")

    unknown_model_names = [
        model_name
        for model_name in MODEL_NAMES
        if model_name not in SUPPORTED_MODEL_NAMES
    ]
    if unknown_model_names:
        raise ValueError(
            f"MODEL_NAMES must only contain {SUPPORTED_MODEL_NAMES}, "
            f"got {unknown_model_names}"
        )


def topk_rows(logits: torch.Tensor, categories: list[str]) -> list[dict[str, Any]]:
    probabilities = logits.softmax(dim=1)[0]
    scores, class_ids = probabilities.topk(TOP_K)

    rows = []
    for rank, (score, class_id) in enumerate(zip(scores, class_ids), start=1):
        class_id_int = int(class_id.item())
        rows.append(
            {
                "rank": rank,
                "label": categories[class_id_int],
                "probability": float(score.item()),
                "class_id": class_id_int,
            }
        )
    return rows


def print_classification_table(rows: list[dict[str, Any]]) -> None:
    headers = ("rank", "label", "probability", "class_id")
    formatted_rows = [
        (
            str(row["rank"]),
            row["label"],
            f"{row['probability']:.4f}",
            str(row["class_id"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in formatted_rows))
        for column in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in formatted_rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def save_results_json(
    model_name: str,
    image_path: Path,
    rows: list[dict[str, Any]],
    input_shape: tuple[int, ...],
) -> None:
    output_dir = resolve_repo_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_topk.json"
    output = {
        "model_name": model_name,
        "checkpoint": SHUFFLENET_MODEL_CHECKPOINTS[model_name],
        "image_path": str(display_path(image_path)),
        "top_k": TOP_K,
        "input_shape": list(input_shape),
        "predictions": rows,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved JSON: {display_path(output_path)}")


def save_input_image(image_path: Path) -> None:
    output_dir = resolve_repo_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_image.jpg"
    image = load_rgb_image(image_path)
    image.save(output_path)
    print(f"Saved input image: {display_path(output_path)}")


def run_model(model_name: str, image_path: Path, device: torch.device) -> None:
    from torchvision.models import ShuffleNet_V2_X0_5_Weights

    print(f"\nLoading {model_name} on {device}...")
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
    model = load_model(model_name).to(device).eval()
    transform = weights.transforms()
    categories = weights.meta["categories"]

    image = load_rgb_image(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)

    rows = topk_rows(logits, categories)
    input_shape = tuple(input_tensor.shape)
    print(f"Model: {model_name}")
    print(f"Image: {display_path(image_path)}")
    print(f"Input tensor shape: {input_shape}")
    print_classification_table(rows)

    if SAVE_RESULTS_JSON:
        save_results_json(model_name, image_path, rows, input_shape)


def main() -> None:
    validate_config()

    image_path = select_image_path()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    device = resolve_device(DEVICE_NAME)
    print(f"Selected image: {display_path(image_path)}")

    if SAVE_INPUT_IMAGE:
        save_input_image(image_path)

    for model_name in MODEL_NAMES:
        run_model(model_name, image_path, device)


if __name__ == "__main__":
    main()
