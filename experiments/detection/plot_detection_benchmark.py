from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Allow direct execution from the repository root.
REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from experiments.detection.constants import (  # noqa: E402
    COMMON_METRICS,
    COMMON_METRICS_DIR,
    IMAGE_CLASSIFICATION_DIR,
    IMAGE_CLASSIFICATION_METRICS,
    OBJECT_DETECTION_DIR,
    OBJECT_DETECTION_METRICS,
    RESULTS_CSV,
    TASK_COLORS,
)


TASK_LABELS = {
    IMAGE_CLASSIFICATION_DIR: "Image classification",
    OBJECT_DETECTION_DIR: "Object detection",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create task-specific and common detection benchmark plots."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_CSV,
        help="CSV file to plot. Defaults to the detection benchmark results.",
    )
    args = parser.parse_args()

    created_plots = create_detection_plots(args.csv)

    print(f"Created {len(created_plots)} plots:")
    for plot_path in created_plots:
        print(f"- {plot_path}")


def create_detection_plots(csv_path: Path) -> list[Path]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results CSV does not exist: {csv_path}")

    results = pd.read_csv(csv_path)
    required_columns = {"model_name", "task"}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(
            "Results CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    output_root = csv_path.parent
    created_plots = []

    task_groups = (
        (
            IMAGE_CLASSIFICATION_DIR,
            IMAGE_CLASSIFICATION_METRICS,
            results[results["task"] == IMAGE_CLASSIFICATION_DIR].copy(),
        ),
        (
            OBJECT_DETECTION_DIR,
            OBJECT_DETECTION_METRICS,
            results[results["task"] == OBJECT_DETECTION_DIR].copy(),
        ),
    )
    for task_name, metric_columns, task_results in task_groups:
        output_dir = output_root / task_name
        remove_stale_plots(output_dir)
        created_plots.extend(
            create_task_plots(
                task_results,
                metric_columns,
                output_dir,
                task_name,
            )
        )

    common_output_dir = output_root / COMMON_METRICS_DIR
    remove_stale_plots(common_output_dir)
    supported_results = results[
        results["task"].isin(TASK_COLORS)
    ].copy()
    created_plots.extend(
        create_common_plots(
            supported_results,
            COMMON_METRICS,
            common_output_dir,
        )
    )

    unknown_tasks = sorted(set(results["task"].astype(str)) - set(TASK_COLORS))
    if unknown_tasks:
        print(
            "Rows skipped because their task has no plot style: "
            + ", ".join(unknown_tasks)
        )

    return created_plots


def create_task_plots(
    results: pd.DataFrame,
    metric_columns: tuple[str, ...],
    output_dir: Path,
    task_name: str,
) -> list[Path]:
    if results.empty:
        print(f"Skipping {task_name}: no rows to plot.")
        return []

    labels = make_plot_labels(results["model_name"].astype(str).tolist())
    created_plots = []

    for metric_column in metric_columns:
        plot_data = make_metric_data(results, labels, metric_column)
        if plot_data.empty:
            print(f"Skipping {task_name}/{metric_column}: no numeric values.")
            continue

        output_path = output_dir / f"{safe_filename(metric_column)}_bar.png"
        plot_task_metric(plot_data, metric_column, task_name, output_path)
        created_plots.append(output_path)

    return created_plots


def create_common_plots(
    results: pd.DataFrame,
    metric_columns: tuple[str, ...],
    output_dir: Path,
) -> list[Path]:
    if results.empty:
        print("Skipping common_metrics: no supported rows to plot.")
        return []

    labels = make_plot_labels(results["model_name"].astype(str).tolist())
    created_plots = []

    for metric_column in metric_columns:
        plot_data = make_metric_data(results, labels, metric_column)
        if plot_data.empty:
            print(f"Skipping common_metrics/{metric_column}: no numeric values.")
            continue

        plot_data["task"] = results.loc[plot_data.index, "task"].astype(str)
        output_path = output_dir / f"{safe_filename(metric_column)}_bar.png"
        plot_common_metric(plot_data, metric_column, output_path)
        created_plots.append(output_path)

    return created_plots


def make_metric_data(
    results: pd.DataFrame,
    labels: list[str],
    metric_column: str,
) -> pd.DataFrame:
    if metric_column not in results.columns:
        return pd.DataFrame(columns=["model_name", metric_column])

    metric_values = pd.to_numeric(results[metric_column], errors="coerce")
    return pd.DataFrame(
        {
            "model_name": labels,
            metric_column: metric_values,
        },
        index=results.index,
    ).dropna(subset=[metric_column])


def plot_task_metric(
    plot_data: pd.DataFrame,
    metric_column: str,
    task_name: str,
    output_path: Path,
) -> None:
    color = TASK_COLORS[task_name]
    title = f"{TASK_LABELS[task_name]}: {format_metric_name(metric_column)}"
    plot_bar_chart(
        plot_data=plot_data,
        metric_column=metric_column,
        output_path=output_path,
        colors=[color] * len(plot_data),
        title=title,
    )


def plot_common_metric(
    plot_data: pd.DataFrame,
    metric_column: str,
    output_path: Path,
) -> None:
    colors = [TASK_COLORS[task] for task in plot_data["task"]]
    legend_handles = [
        Patch(color=color, label=TASK_LABELS[task])
        for task, color in TASK_COLORS.items()
    ]
    plot_bar_chart(
        plot_data=plot_data,
        metric_column=metric_column,
        output_path=output_path,
        colors=colors,
        title=f"All Models: {format_metric_name(metric_column)}",
        subtitle=(
            "Task-specific definitions and datasets; use for context, "
            "not a single cross-task leaderboard."
        ),
        legend_handles=legend_handles,
    )


def plot_bar_chart(
    plot_data: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    colors: list[str],
    title: str,
    subtitle: str | None = None,
    legend_handles: list[Patch] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(8, len(plot_data) * 0.85)
    figure, axis = plt.subplots(figsize=(figure_width, 5.5))

    bars = axis.bar(
        plot_data["model_name"],
        plot_data[metric_column],
        color=colors,
    )
    axis.set_title(title, pad=22 if subtitle else 10)
    if subtitle is not None:
        axis.text(
            0.5,
            1.01,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xlabel("Model")
    axis.set_ylabel(format_metric_name(metric_column))
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)

    for tick_label in axis.get_xticklabels():
        tick_label.set_horizontalalignment("right")
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    if legend_handles is not None:
        axis.legend(handles=legend_handles, loc="lower right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def remove_stale_plots(output_dir: Path) -> None:
    if not output_dir.is_dir():
        return

    for plot_path in output_dir.glob("*_bar.png"):
        plot_path.unlink()


def make_plot_labels(model_names: list[str]) -> list[str]:
    total_counts = Counter(model_names)
    seen_counts: dict[str, int] = {}
    labels = []

    for model_name in model_names:
        seen_counts[model_name] = seen_counts.get(model_name, 0) + 1
        if total_counts[model_name] == 1:
            labels.append(model_name)
        else:
            labels.append(f"{model_name} #{seen_counts[model_name]}")

    return labels


def format_metric_name(metric_column: str) -> str:
    labels = {
        "map_50_95": "mAP 50-95",
        "map_50": "mAP 50",
        "map_75": "mAP 75",
        "mean_iou": "Mean IoU",
        "f1_score": "F1 Score",
        "top1_accuracy": "Top-1 Accuracy",
        "top5_accuracy": "Top-5 Accuracy",
    }
    return labels.get(metric_column, metric_column.replace("_", " ").title())


def safe_filename(metric_column: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", metric_column).strip("_").lower()


if __name__ == "__main__":
    main()
