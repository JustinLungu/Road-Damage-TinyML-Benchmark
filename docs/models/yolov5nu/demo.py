from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import torch


# Resolve paths from this file so the demo can be run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DOCS_DIR = Path(__file__).resolve().parent

# Allow imports from src/ when this file is executed directly.
sys.path.insert(0, str(REPO_ROOT))

from src.load_model import load_model  # noqa: E402
from src.performance_benchmark.utils import resolve_device  # noqa: E402


#############
# Demo Config
#############

MODEL_NAME = "yolov5nu"

# Manual-image mode: used when USE_RANDOM_IMAGE is False.
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"

# Random-image mode: set USE_RANDOM_IMAGE to True to sample from this directory.
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"

# Set to an integer for reproducible random image selection.
RANDOM_SEED: int | None = None

DEVICE_NAME = "cpu"       # or "cuda:0" if CUDA is available
# Ultralytics resizes/letterboxes the source image to this inference size.
# Output boxes are still mapped back to the original image pixel coordinates.
# 320 faster & less memory, 640 more accurate, 1280 slower & more memory.
# Must be multiple of 32 due to YOLOv5 architecture.
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.6
NMS_IOU_THRESHOLD = 0.7

SAVE_ANNOTATED_IMAGE = True
OUTPUT_PATH = MODEL_DOCS_DIR / "outputs" / "annotated.jpg"


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


def detection_rows(result: Any) -> list[dict[str, Any]]:
    """Convert Ultralytics boxes into plain Python values for printing."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    rows = []
    for index, (xyxy, confidence, class_id) in enumerate(
        zip(boxes.xyxy, boxes.conf, boxes.cls),
        start=1,
    ):
        class_id_int = int(class_id.item())
        # xyxy is already scaled back to original image pixel coordinates.
        x1, y1, x2, y2 = [float(value) for value in xyxy.tolist()]
        rows.append(
            {
                "index": index,
                "label": result.names[class_id_int],
                "confidence": float(confidence.item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )
    return rows


def print_detection_table(rows: list[dict[str, Any]]) -> None:
    """Print detections as a small fixed-width table."""
    if not rows:
        print("No detections above the confidence threshold.")
        return

    headers = ("#", "label", "conf", "x1", "y1", "x2", "y2", "w", "h")
    formatted_rows = [
        (
            str(row["index"]),
            row["label"],
            f"{row['confidence']:.3f}",
            f"{row['x1']:.1f}",
            f"{row['y1']:.1f}",
            f"{row['x2']:.1f}",
            f"{row['y2']:.1f}",
            f"{row['width']:.1f}",
            f"{row['height']:.1f}",
        )
        for row in rows
    ]
    # Compute column widths from both headers and values so labels align cleanly.
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


def main() -> None:
    image_path = select_image_path()
    output_path = resolve_repo_path(OUTPUT_PATH)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    device = resolve_device(DEVICE_NAME)
    print(f"Loading {MODEL_NAME} on {device}...")
    model = load_model(MODEL_NAME)

    # Ultralytics handles image loading, resizing, tensor conversion, forward
    # pass, confidence filtering, and non-maximum suppression inside predict().
    with torch.inference_mode():
        result = model.predict(
            source=str(image_path),
            device=str(device),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
        )[0]

    rows = detection_rows(result)
    print(f"Image: {display_path(image_path)}")
    print(f"Original shape: {result.orig_shape}")
    print(f"Detections: {len(rows)}")
    print_detection_table(rows)

    if SAVE_ANNOTATED_IMAGE:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(filename=str(output_path))
        print(f"Annotated image: {display_path(output_path)}")


if __name__ == "__main__":
    main()
