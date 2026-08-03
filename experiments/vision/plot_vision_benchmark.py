from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow direct execution from the repository root.
REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from experiments.vision.constants import (  # noqa: E402
    CLASSIFICATION_RESULTS_CSV,
    IMAGE_CLASSIFICATION_METRICS,
    OBJECT_DETECTION_METRICS,
    OBJECT_DETECTION_RESULTS_CSV,
)


TASKS = (
    (
        "image_classification",
        "Image classification",
        "#4C78A8",
        IMAGE_CLASSIFICATION_METRICS,
    ),
    (
        "object_detection",
        "Object detection",
        "#F58518",
        OBJECT_DETECTION_METRICS,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create task-specific vision benchmark plots."
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=CLASSIFICATION_RESULTS_CSV,
    )
    parser.add_argument(
        "--object-detection-csv",
        type=Path,
        default=OBJECT_DETECTION_RESULTS_CSV,
    )
    args = parser.parse_args()

    created_plots = create_vision_plots(
        args.classification_csv,
        args.object_detection_csv,
    )
    print(f"Created {len(created_plots)} plots:")
    for plot_path in created_plots:
        print(f"- {plot_path}")


def create_vision_plots(
    classification_csv: Path,
    object_detection_csv: Path,
) -> list[Path]:
    created_plots = []
    csv_paths = (classification_csv, object_detection_csv)

    for (output_name, title, color, metrics), csv_path in zip(TASKS, csv_paths):
        if not csv_path.is_file():
            print(f"Skipping {title.lower()}: {csv_path} does not exist.")
            continue

        results = pd.read_csv(csv_path)
        if "model_name" not in results.columns:
            raise ValueError(f"Results CSV is missing model_name: {csv_path}")

        output_dir = csv_path.parent / output_name
        remove_stale_plots(output_dir)
        created_plots.extend(
            create_task_plots(results, metrics, output_dir, title, color)
        )

    return created_plots


def create_task_plots(
    results: pd.DataFrame,
    metrics: tuple[str, ...],
    output_dir: Path,
    task_title: str,
    color: str,
) -> list[Path]:
    created_plots = []

    for metric in metrics:
        if metric not in results.columns:
            continue

        values = pd.to_numeric(results[metric], errors="coerce")
        plot_data = results.loc[values.notna(), ["model_name"]].copy()
        plot_data[metric] = values[values.notna()]
        if plot_data.empty:
            continue

        output_path = output_dir / f"{metric}_bar.png"
        plot_metric(plot_data, metric, task_title, color, output_path)
        created_plots.append(output_path)

    return created_plots


def plot_metric(
    results: pd.DataFrame,
    metric: str,
    task_title: str,
    color: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(8, len(results) * 0.85), 5.5))
    bars = axis.bar(results["model_name"], results[metric], color=color)

    axis.set_title(f"{task_title}: {format_metric_name(metric)}")
    axis.set_xlabel("Model")
    axis.set_ylabel(format_metric_name(metric))
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def remove_stale_plots(output_dir: Path) -> None:
    if output_dir.is_dir():
        for plot_path in output_dir.glob("*_bar.png"):
            plot_path.unlink()


def format_metric_name(metric: str) -> str:
    labels = {
        "map_50_95": "mAP 50-95",
        "map_50": "mAP 50",
        "map_75": "mAP 75",
        "mean_iou": "Mean IoU",
        "f1_score": "F1 Score",
        "top1_accuracy": "Top-1 Accuracy",
        "top5_accuracy": "Top-5 Accuracy",
    }
    return labels.get(metric, metric.replace("_", " ").title())


if __name__ == "__main__":
    main()
