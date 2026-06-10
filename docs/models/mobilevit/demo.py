from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoImageProcessor


# Resolve paths from this file so the demo can be run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DOCS_DIR = Path(__file__).resolve().parent

# Allow imports from src/ when this file is executed directly.
sys.path.insert(0, str(REPO_ROOT))

from src.constants import MOBILEVIT_MODEL_IDS, VIT_DIR  # noqa: E402
from src.load_model import load_model  # noqa: E402
from src.performance_benchmark.utils import (  # noqa: E402
    load_rgb_image,
    move_inputs_to_device,
    resolve_device,
)


#############
# Demo Config
#############

SUPPORTED_MODEL_NAMES = tuple(MOBILEVIT_MODEL_IDS)
MODEL_NAMES = ["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"]

# Manual-image mode: used when USE_RANDOM_IMAGE is False.
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"

# Random-image mode: set USE_RANDOM_IMAGE to True to sample from this directory.
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"

# Set to an integer for reproducible random image selection.
RANDOM_SEED: int | None = None

DEVICE_NAME = "cpu"  # or "cuda:0" if CUDA is available
TOP_K = 5

# Keep this True for local demos so Transformers uses the repo's cached files
# instead of trying to reach Hugging Face.
LOCAL_FILES_ONLY = True

SAVE_RESULTS_JSON = True
SAVE_INPUT_IMAGE = True
OUTPUT_DIR = MODEL_DOCS_DIR / "outputs"


def resolve_repo_path(path: Path) -> Path:
    """Accept absolute paths or paths relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> Path:
    """Print short repo-relative paths when possible."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def select_image_path() -> Path:
    """Choose either the configured image or a random COCO image."""
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
        if model_name not in MOBILEVIT_MODEL_IDS
    ]
    if unknown_model_names:
        raise ValueError(
            f"MODEL_NAMES must only contain {SUPPORTED_MODEL_NAMES}, "
            f"got {unknown_model_names}"
        )


def configure_transformers_cache_mode() -> None:
    """Force Hugging Face calls to use cached files when LOCAL_FILES_ONLY is set."""
    if LOCAL_FILES_ONLY:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_processor(model_name: str) -> Any:
    """Load the matching image processor from the same local cache."""
    model_id = MOBILEVIT_MODEL_IDS[model_name]
    return AutoImageProcessor.from_pretrained(
        model_id,
        cache_dir=str(VIT_DIR / model_name),
        local_files_only=LOCAL_FILES_ONLY,
        use_fast=False,
    )


def tensor_shapes(inputs: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Record tensor shapes without saving large tensor values to JSON."""
    return {
        key: tuple(value.shape)
        for key, value in inputs.items()
        if hasattr(value, "shape")
    }


def get_label(id2label: dict[Any, str], class_id: int) -> str:
    """Handle configs that store ImageNet label keys as ints or strings."""
    return id2label.get(class_id) or id2label.get(str(class_id), str(class_id))


def topk_rows(
    logits: torch.Tensor,
    id2label: dict[Any, str],
) -> list[dict[str, Any]]:
    """Convert model logits into top-k probability rows."""
    probabilities = logits.softmax(dim=-1)[0]
    scores, class_ids = probabilities.topk(TOP_K)

    rows = []
    for rank, (score, class_id) in enumerate(zip(scores, class_ids), start=1):
        class_id_int = int(class_id.item())
        rows.append(
            {
                "rank": rank,
                "label": get_label(id2label, class_id_int),
                "probability": float(score.item()),
                "class_id": class_id_int,
            }
        )
    return rows


def print_classification_table(rows: list[dict[str, Any]]) -> None:
    """Print top-k classification predictions as a fixed-width table."""
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
    header_line = "  ".join(
        header.ljust(widths[column]) for column, header in enumerate(headers)
    )
    divider = "  ".join("-" * width for width in widths)
    print(header_line)
    print(divider)
    for row in formatted_rows:
        print(
            "  ".join(
                value.ljust(widths[column]) for column, value in enumerate(row)
            )
        )


def save_results_json(
    model_name: str,
    image_path: Path,
    rows: list[dict[str, Any]],
    input_shapes: dict[str, tuple[int, ...]],
) -> None:
    output_dir = resolve_repo_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_topk.json"
    output = {
        "model_name": model_name,
        "model_id": MOBILEVIT_MODEL_IDS[model_name],
        "image_path": str(display_path(image_path)),
        "top_k": TOP_K,
        "input_shapes": {
            key: list(shape) for key, shape in input_shapes.items()
        },
        "predictions": rows,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved JSON: {display_path(output_path)}")


def save_input_image(image_path: Path) -> None:
    """Save the selected image so the classification input is easy to inspect."""
    output_dir = resolve_repo_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_image.jpg"
    image = load_rgb_image(image_path)
    image.save(output_path)
    print(f"Saved input image: {display_path(output_path)}")


def run_model(model_name: str, image_path: Path, device: torch.device) -> None:
    print(f"\nLoading {model_name} on {device}...")
    processor = load_processor(model_name)
    model = load_model(model_name).to(device).eval()

    # The processor applies the checkpoint's resize, center crop, channel order,
    # and tensor conversion before the image reaches the model.
    image = load_rgb_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    input_shapes = tensor_shapes(inputs)

    with torch.inference_mode():
        outputs = model(**move_inputs_to_device(inputs, device))

    rows = topk_rows(outputs.logits, model.config.id2label)
    print(f"Model: {model_name}")
    print(f"Model ID: {MOBILEVIT_MODEL_IDS[model_name]}")
    print(f"Image: {display_path(image_path)}")
    print(f"Input tensor shapes: {input_shapes}")
    print_classification_table(rows)

    if SAVE_RESULTS_JSON:
        save_results_json(model_name, image_path, rows, input_shapes)


def main() -> None:
    validate_config()
    configure_transformers_cache_mode()

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
