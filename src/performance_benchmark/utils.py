from __future__ import annotations

import csv
from dataclasses import asdict
import platform
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from PIL import Image

from src.performance_benchmark.benchmark_result import BenchmarkResult


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device."
        )

    return device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "CPU"


def model_precision(model: Any) -> str:
    module = getattr(model, "model", model)
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return "unknown"

    try:
        return str(next(parameters()).dtype).removeprefix("torch.")
    except StopIteration:
        return "unknown"


def list_coco_images(images_dir: Path, num_images: int | None = None) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO images directory does not exist: {images_dir}")

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in: {images_dir}")

    if num_images is not None:
        if num_images <= 0:
            raise ValueError("num_images must be greater than zero.")
        image_paths = image_paths[:num_images]

    return image_paths


def append_result_csv(result: BenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Append rows across runs, but create the header for a new/empty file.
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(asdict(result).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def move_inputs_to_device(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def synchronize_device(device: torch.device) -> None:
    # CUDA work is asynchronous; synchronize before reading timing boundaries.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def average_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def percentile(values: list[float], percentile_value: float) -> float:
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile_value / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index

    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )
