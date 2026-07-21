from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.performance_benchmark.utils import load_rgb_image, resolve_device  # noqa: E402
from src.rdd_benchmark.constants import ID_TO_LABEL  # noqa: E402
from src.rdd_benchmark.training.utils import (  # noqa: E402
    extract_logits,
    load_and_adapt_model_for_binary_pothole,
    make_image_transform,
)


CHECKPOINT_PATH = (
    REPO_ROOT
    / "results"
    / "rdd_trained_models"
    / "G_clean400_full_image_downsample_1to5_standard_aug"
    / "tiny_cnn"
    / "best.pt"
)

# Set this to a specific image to test it directly. When None, the first image
# in MANIFEST_PATH is used.
IMAGE_PATH: Path | None = None
MANIFEST_PATH = REPO_ROOT / "datasets" / "rdd2022" / "binary_pothole" / "test.csv"

DEVICE_NAME = "cpu"


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def select_image_path() -> Path:
    if IMAGE_PATH is not None:
        return resolve_repo_path(IMAGE_PATH)

    with resolve_repo_path(MANIFEST_PATH).open(newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))

    return resolve_repo_path(Path(row["image_path"]))


def main() -> None:
    checkpoint_path = resolve_repo_path(CHECKPOINT_PATH)
    image_path = select_image_path()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    device = resolve_device(DEVICE_NAME)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_name = checkpoint["model_name"]

    model = load_and_adapt_model_for_binary_pothole(model_name).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])

    image = load_rgb_image(image_path)
    transform = make_image_transform(model_name, is_train=False)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = extract_logits(model(input_tensor))
        probabilities = logits.softmax(dim=1)[0]
        predicted_label_id = int(probabilities.argmax().item())

    print(f"checkpoint: {display_path(checkpoint_path)}")
    print(f"model: {model_name}")
    print(f"image: {display_path(image_path)}")
    print(f"prediction: {ID_TO_LABEL[predicted_label_id]}")
    print(f"non_pothole_probability: {probabilities[0].item():.4f}")
    print(f"pothole_probability: {probabilities[1].item():.4f}")


if __name__ == "__main__":
    main()
