import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from experiments.performance.constants import (  # noqa: E402
    DEFAULT_CSV,
    WITH_VLMS_DIR,
    WITHOUT_VLMS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one bar plot per system performance metric."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV file to plot. Defaults to the system metrics results CSV.",
    )
    args = parser.parse_args()

    created_plots = create_metric_plots(args.csv)

    print(f"Created {len(created_plots)} plots:")
    for plot_path in created_plots:
        print(f"- {plot_path}")


def create_metric_plots(csv_path: Path) -> list[Path]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results CSV does not exist: {csv_path}")

    results = pd.read_csv(csv_path)
    if "model_name" not in results.columns:
        raise ValueError("Results CSV must contain a model_name column.")

    output_root = csv_path.parent
    model_names = results["model_name"].astype(str)
    non_vlm_results = results[~model_names.map(is_vlm_model)].copy()

    plot_groups = [
        (WITH_VLMS_DIR, results),
        (WITHOUT_VLMS_DIR, non_vlm_results),
    ]

    metric_columns = [column for column in results.columns if column != "model_name"]
    created_plots = []

    for group_name, group_results in plot_groups:
        created_plots.extend(
            create_metric_plots_for_group(
                results=group_results,
                metric_columns=metric_columns,
                output_dir=output_root / group_name,
                group_name=group_name,
            )
        )

    return created_plots


def create_metric_plots_for_group(
    results: pd.DataFrame,
    metric_columns: list[str],
    output_dir: Path,
    group_name: str,
) -> list[Path]:
    if results.empty:
        print(f"Skipping {group_name}: no rows to plot.")
        return []

    plot_labels = make_plot_labels(results["model_name"].astype(str).tolist())
    created_plots = []

    for metric_column in metric_columns:
        metric_values = pd.to_numeric(results[metric_column], errors="coerce")
        plot_data = pd.DataFrame(
            {
                "model_name": plot_labels,
                metric_column: metric_values,
            }
        ).dropna(subset=[metric_column])

        if plot_data.empty:
            print(f"Skipping {group_name}/{metric_column}: no numeric values found.")
            continue

        output_path = output_dir / f"{safe_filename(metric_column)}_bar.png"
        plot_metric(plot_data, metric_column, output_path)
        created_plots.append(output_path)

    return created_plots


def is_vlm_model(model_name: str) -> bool:
    return "vlm" in model_name.lower()


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


def plot_metric(plot_data: pd.DataFrame, metric_column: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(8, len(plot_data) * 0.8)
    figure, axis = plt.subplots(figsize=(figure_width, 5))

    axis.bar(plot_data["model_name"], plot_data[metric_column])
    axis.set_title(format_metric_name(metric_column))
    axis.set_xlabel("Model")
    axis.set_ylabel(metric_column)
    axis.tick_params(axis="x", rotation=45)

    for tick_label in axis.get_xticklabels():
        tick_label.set_horizontalalignment("right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def format_metric_name(metric_column: str) -> str:
    return metric_column.replace("_", " ").title()


def safe_filename(metric_column: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", metric_column).strip("_").lower()


if __name__ == "__main__":
    main()
