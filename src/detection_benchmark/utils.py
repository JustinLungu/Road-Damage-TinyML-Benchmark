from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from src.detection_benchmark.benchmark_result import DetectionBenchmarkResult


def append_result_csv(
    result: DetectionBenchmarkResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(asdict(result)))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def validate_unit_interval(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def print_progress(completed: int, total: int) -> None:
    from src.detection_benchmark.constants import PROGRESS_INTERVAL_IMAGES

    if completed % PROGRESS_INTERVAL_IMAGES == 0 or completed == total:
        print(f"Progress: {completed}/{total} images")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
