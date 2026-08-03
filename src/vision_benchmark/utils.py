from __future__ import annotations

from typing import Sequence


def validate_unit_interval(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def print_progress(completed: int, total: int) -> None:
    from src.vision_benchmark.constants import PROGRESS_INTERVAL_IMAGES

    if completed % PROGRESS_INTERVAL_IMAGES == 0 or completed == total:
        print(f"Progress: {completed}/{total} images")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
