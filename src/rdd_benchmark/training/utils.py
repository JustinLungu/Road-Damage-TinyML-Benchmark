from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import torch

from src.rdd_benchmark.constants import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RDD_COMPARISON_RANKING_METRIC,
)
from src.rdd_benchmark.data_preprocessing.augmentation import make_rdd_image_transform
from src.rdd_benchmark.training.constants import DEFAULT_IMAGE_SIZE, MODEL_IMAGE_SIZES


def make_image_transform(
    model_name: str,
    is_train: bool,
    augmentation_strategy: str = "standard",
):
    image_size = MODEL_IMAGE_SIZES.get(model_name, DEFAULT_IMAGE_SIZE)
    return make_rdd_image_transform(
        image_size=image_size,
        is_train=is_train,
        augmentation_strategy=augmentation_strategy,
    )


def extract_logits(model_output: Any) -> torch.Tensor:
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, tuple | list):
        return model_output[0]
    return model_output


def compute_binary_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    predictions = predictions.detach().cpu().long()
    targets = targets.detach().cpu().long()

    true_positive = int(
        ((predictions == POSITIVE_LABEL) & (targets == POSITIVE_LABEL)).sum().item()
    )
    true_negative = int(
        ((predictions == NEGATIVE_LABEL) & (targets == NEGATIVE_LABEL)).sum().item()
    )
    false_positive = int(
        ((predictions == POSITIVE_LABEL) & (targets == NEGATIVE_LABEL)).sum().item()
    )
    false_negative = int(
        ((predictions == NEGATIVE_LABEL) & (targets == POSITIVE_LABEL)).sum().item()
    )

    total = true_positive + true_negative + false_positive + false_negative
    accuracy = safe_divide(true_positive + true_negative, total)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    balanced_accuracy = (recall + specificity) / 2
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(true_positive),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


def compute_binary_roc_auc(
    positive_scores: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    scores = positive_scores.detach().cpu().float().tolist()
    labels = targets.detach().cpu().long().tolist()
    positive_count = sum(1 for label in labels if label == POSITIVE_LABEL)
    negative_count = sum(1 for label in labels if label == NEGATIVE_LABEL)

    if positive_count == 0 or negative_count == 0:
        return float("nan")

    ranked_scores = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    current_index = 0

    while current_index < len(ranked_scores):
        next_index = current_index + 1
        while (
            next_index < len(ranked_scores)
            and ranked_scores[next_index][1] == ranked_scores[current_index][1]
        ):
            next_index += 1

        average_rank = (current_index + 1 + next_index) / 2
        for rank_index in range(current_index, next_index):
            original_index = ranked_scores[rank_index][0]
            ranks[original_index] = average_rank

        current_index = next_index

    positive_rank_sum = sum(
        rank
        for rank, label in zip(ranks, labels, strict=True)
        if label == POSITIVE_LABEL
    )
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def compute_binary_roc_curve(
    positive_scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[float], list[float]]:
    scores = positive_scores.detach().cpu().float()
    labels = targets.detach().cpu().long()
    thresholds = torch.cat(
        [
            torch.tensor([float("inf")]),
            torch.sort(torch.unique(scores, dim=0), descending=True).values,
            torch.tensor([float("-inf")]),
        ],
        dim=0,
    )
    false_positive_rates = []
    true_positive_rates = []

    positive_count = int((labels == POSITIVE_LABEL).sum().item())
    negative_count = int((labels == NEGATIVE_LABEL).sum().item())
    if positive_count == 0 or negative_count == 0:
        return [0.0, 1.0], [0.0, 1.0]

    for threshold in thresholds:
        predictions = (scores >= threshold).long()
        true_positive = int(
            ((predictions == POSITIVE_LABEL) & (labels == POSITIVE_LABEL)).sum().item()
        )
        false_positive = int(
            ((predictions == POSITIVE_LABEL) & (labels == NEGATIVE_LABEL)).sum().item()
        )
        true_positive_rates.append(true_positive / positive_count)
        false_positive_rates.append(false_positive / negative_count)

    return false_positive_rates, true_positive_rates


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def write_training_curves(
    plot_path: Path,
    history: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (metric_name, title) in zip(
        axes,
        (("loss", "Loss"), ("f1", "F1-score"), ("accuracy", "Accuracy")),
        strict=True,
    ):
        axis.plot(
            epochs,
            [float(row[f"train_{metric_name}"]) for row in history],
            marker="o",
            label="train",
        )
        axis.plot(
            epochs,
            [float(row[f"validation_{metric_name}"]) for row in history],
            marker="o",
            label="validation",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.legend()

    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def write_failure_log(
    log_path: Path,
    message: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(message, encoding="utf-8")


def write_evaluation_metrics_csv(
    metrics_path: Path,
    model_name: str,
    split: str,
    metrics: dict[str, Any],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model_name": model_name, "split": split, **metrics}

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_confusion_matrix_plot(
    plot_path: Path,
    metrics: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = [
        [int(metrics["true_negative"]), int(metrics["false_positive"])],
        [int(metrics["false_negative"]), int(metrics["true_positive"])],
    ]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["non_pothole", "pothole"],
        yticklabels=["non_pothole", "pothole"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("RDD Binary Pothole Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def write_roc_curve_plot(
    plot_path: Path,
    false_positive_rates: list[float],
    true_positive_rates: list[float],
    roc_auc: float,
) -> None:
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(false_positive_rates, true_positive_rates, label=f"ROC-AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RDD Binary Pothole ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def write_model_comparison_csv(
    comparison_path: Path,
    evaluation_rows: list[dict[str, Any]],
    ranking_metric: str = RDD_COMPARISON_RANKING_METRIC,
) -> list[dict[str, Any]]:
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_rows = rank_evaluation_rows(evaluation_rows, ranking_metric)

    with comparison_path.open("w", newline="", encoding="utf-8") as comparison_file:
        writer = csv.DictWriter(comparison_file, fieldnames=list(ranked_rows[0]))
        writer.writeheader()
        writer.writerows(ranked_rows)

    return ranked_rows


def load_evaluation_rows_from_metrics_csv(
    model_names: Iterable[str],
    output_dir: Path,
    skip_missing: bool = False,
) -> list[dict[str, Any]]:
    evaluation_rows = []

    for model_name in model_names:
        metrics_path = output_dir / model_name / "test_metrics.csv"
        if not metrics_path.is_file():
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"Evaluation metrics do not exist for {model_name}: {metrics_path}"
            )

        with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
            reader = csv.DictReader(metrics_file)
            row = next(reader)

        evaluation_rows.append(
            {
                "model_name": model_name,
                "checkpoint_path": str(output_dir / model_name / "best.pt"),
                **{
                    key: parse_metric_value(value)
                    for key, value in row.items()
                    if key not in {"model_name", "split"}
                },
            }
        )

    return evaluation_rows


def rank_evaluation_rows(
    evaluation_rows: list[dict[str, Any]],
    ranking_metric: str,
) -> list[dict[str, Any]]:
    if not evaluation_rows:
        raise ValueError("Cannot rank an empty evaluation result list.")

    ranked_rows = sorted(
        evaluation_rows,
        key=lambda row: row[ranking_metric],
        reverse=True,
    )

    return [
        {
            "rank": rank,
            **row,
        }
        for rank, row in enumerate(ranked_rows, start=1)
    ]


def parse_metric_value(value: str) -> float | str:
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value
