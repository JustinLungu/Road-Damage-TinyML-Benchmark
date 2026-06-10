from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor


# Resolve paths from this file so the demo can be run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DOCS_DIR = Path(__file__).resolve().parent

# Allow imports from src/ when this file is executed directly.
sys.path.insert(0, str(REPO_ROOT))

from src.constants import SMOLVLM_MODEL_IDS, VLM_DIR  # noqa: E402
from src.load_model import load_model  # noqa: E402
from src.performance_benchmark.utils import (  # noqa: E402
    load_rgb_image,
    move_inputs_to_device,
    resolve_device,
)


#############
# Demo Config
#############

SUPPORTED_MODEL_NAMES = tuple(SMOLVLM_MODEL_IDS)
MODEL_NAMES = ["smolvlm_256m", "smolvlm_500m"]#, "smolvlm_2b"]

# Manual-image mode: used when USE_RANDOM_IMAGE is False.
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"

# Random-image mode: set USE_RANDOM_IMAGE to True to sample from this directory.
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"

# Set to an integer for reproducible random image selection.
RANDOM_SEED: int | None = None

DEVICE_NAME = "cuda:0"  # "cpu" or "cuda:0" if CUDA is available

# This text is the instruction given to the VLM together with the image.
PROMPT = "Describe the image in one sentence."

# Generation controls. Larger MAX_NEW_TOKENS allows longer answers, but it also
# makes inference slower because each new token requires another decoding step.
MAX_NEW_TOKENS = 80
DO_SAMPLE = False
TEMPERATURE = 0.7

# Keep this True for local demos so Transformers uses the repo's cached files
# instead of trying to reach Hugging Face.
LOCAL_FILES_ONLY = True

SAVE_RESPONSE_JSON = True
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
    if MAX_NEW_TOKENS <= 0:
        raise ValueError(
            f"MAX_NEW_TOKENS must be greater than zero, got {MAX_NEW_TOKENS}."
        )
    if DO_SAMPLE and TEMPERATURE <= 0:
        raise ValueError(f"TEMPERATURE must be greater than zero, got {TEMPERATURE}.")

    unknown_model_names = [
        model_name for model_name in MODEL_NAMES if model_name not in SMOLVLM_MODEL_IDS
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
    """Load the matching image/text processor from the same local cache."""
    model_id = SMOLVLM_MODEL_IDS[model_name]
    return AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(VLM_DIR / model_name),
        local_files_only=LOCAL_FILES_ONLY,
    )


def build_messages() -> list[dict[str, Any]]:
    """Build the chat-style user message expected by SmolVLM instruct models."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def tensor_shapes(inputs: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Record tensor shapes without saving large tensor values to JSON."""
    return {
        key: tuple(value.shape)
        for key, value in inputs.items()
        if hasattr(value, "shape")
    }


def save_input_image(image_path: Path) -> None:
    """Save the selected image so the VLM input is easy to inspect."""
    output_dir = resolve_repo_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selected_image.jpg"
    image = load_rgb_image(image_path)
    image.save(output_path)
    print(f"Saved input image: {display_path(output_path)}")


def save_response_json(
    model_name: str,
    image_path: Path,
    response: str,
    input_shapes: dict[str, tuple[int, ...]],
) -> None:
    output_dir = resolve_repo_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_response.json"
    output = {
        "model_name": model_name,
        "model_id": SMOLVLM_MODEL_IDS[model_name],
        "image_path": str(display_path(image_path)),
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE if DO_SAMPLE else None,
        "input_shapes": {
            key: list(shape) for key, shape in input_shapes.items()
        },
        "response": response,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved JSON: {display_path(output_path)}")


def run_model(model_name: str, image_path: Path, device: torch.device) -> None:
    print(f"\nLoading {model_name} on {device}...")
    processor = load_processor(model_name)
    model = load_model(model_name).to(device).eval()

    # The chat template inserts the special text and image tokens expected by
    # the instruct checkpoint. The image itself is supplied separately below.
    prompt_text = processor.apply_chat_template(
        build_messages(),
        add_generation_prompt=True,
    )

    image = load_rgb_image(image_path)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    input_shapes = tensor_shapes(inputs)
    prompt_token_count = inputs["input_ids"].shape[1]
    inputs = move_inputs_to_device(inputs, device)

    # Greedy decoding is deterministic. If DO_SAMPLE is True, temperature makes
    # the next-token choices more or less random.
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
    }
    if DO_SAMPLE:
        generate_kwargs["temperature"] = TEMPERATURE

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    # Decode only the newly generated tokens. The full generated sequence also
    # contains the prompt/image tokens that were supplied as input.
    new_token_ids = generated_ids[:, prompt_token_count:]
    response = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
    )[0].strip()

    print(f"Model: {model_name}")
    print(f"Model ID: {SMOLVLM_MODEL_IDS[model_name]}")
    print(f"Image: {display_path(image_path)}")
    print(f"Prompt: {PROMPT}")
    print(f"Input tensor shapes: {input_shapes}")
    print("Response:")
    print(response)

    if SAVE_RESPONSE_JSON:
        save_response_json(model_name, image_path, response, input_shapes)


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
