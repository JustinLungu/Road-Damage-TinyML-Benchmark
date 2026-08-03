import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow direct execution from the repository root.
REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from experiments.performance.constants import (  # noqa: E402
    PERFORMANCE_METRICS,
    PERFORMANCE_TASKS,
    RESULTS_CSV,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create task-specific system performance plots."
    )
    parser.add_argument("--csv", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    created_plots = create_metric_plots(args.csv)
    print(f"Created {len(created_plots)} plots:")
    for plot_path in created_plots:
        print(f"- {plot_path}")


def create_metric_plots(csv_path: Path) -> list[Path]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results CSV does not exist: {csv_path}")

    results = pd.read_csv(csv_path)
    required_columns = {"model_name", "task", "device_name", "num_images"}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(
            "Results CSV is missing columns: " + ", ".join(sorted(missing_columns))
        )

    created_plots = []
    for task in PERFORMANCE_TASKS:
        task_results = results[results["task"] == task].copy()
        output_dir = csv_path.parent / task
        remove_stale_plots(output_dir)
        created_plots.extend(create_task_plots(task_results, output_dir, task))

    unknown_tasks = sorted(set(results["task"].astype(str)) - set(PERFORMANCE_TASKS))
    if unknown_tasks:
        print("Rows skipped for unknown tasks: " + ", ".join(unknown_tasks))

    return created_plots


def create_task_plots(
    results: pd.DataFrame,
    output_dir: Path,
    task: str,
) -> list[Path]:
    if results.empty:
        print(f"Skipping {task}: no rows to plot.")
        return []

    labels = make_plot_labels(results)
    created_plots = []

    for metric in PERFORMANCE_METRICS:
        if metric not in results.columns:
            continue

        values = pd.to_numeric(results[metric], errors="coerce")
        plot_data = pd.DataFrame({"model_name": labels, metric: values}).dropna(
            subset=[metric]
        )
        if plot_data.empty:
            continue

        output_path = output_dir / f"{metric}_bar.png"
        plot_metric(plot_data, metric, output_path)
        created_plots.append(output_path)

    return created_plots


def make_plot_labels(results: pd.DataFrame) -> list[str]:
    model_names = results["model_name"].astype(str)
    duplicate_models = model_names.duplicated(keep=False)
    labels = model_names.copy()
    labels.loc[duplicate_models] = (
        model_names[duplicate_models]
        + "\n"
        + results.loc[duplicate_models, "device_name"].astype(str)
        + ", n="
        + results.loc[duplicate_models, "num_images"].astype(str)
    )
    return labels.tolist()


def plot_metric(
    plot_data: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(8, len(plot_data) * 0.8), 5))
    bars = axis.bar(plot_data["model_name"], plot_data[metric])
    axis.set_title(format_metric_name(metric))
    axis.set_xlabel("Model")
    axis.set_ylabel(format_metric_name(metric))
    axis.tick_params(axis="x", rotation=45)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def remove_stale_plots(output_dir: Path) -> None:
    if output_dir.is_dir():
        for plot_path in output_dir.glob("*_bar.png"):
            plot_path.unlink()


def format_metric_name(metric: str) -> str:
    labels = {
        "fps": "Images per Second (Batch Size 1)",
        "avg_latency_ms": "Average Latency (ms)",
        "p95_latency_ms": "P95 Latency (ms)",
        "avg_cpu_ram_mb": "Average CPU RAM (MB)",
        "peak_cpu_ram_mb": "Peak CPU RAM (MB)",
        "avg_gpu_ram_mb": "Average GPU RAM (MB)",
        "peak_gpu_ram_mb": "Peak GPU RAM (MB)",
        "avg_gpu_utilization_pct": "Average GPU Utilization (%)",
        "avg_power_w": "Average Power (W)",
        "energy_per_inference_j": "Energy per Inference (J)",
    }
    return labels[metric]


if __name__ == "__main__":
    main()
