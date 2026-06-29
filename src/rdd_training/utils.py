from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from PIL import Image

from src.constants import PROJECT_ROOT
from src.rdd_training.constants import (
    DEFAULT_IMAGE_SIZE,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MODEL_IMAGE_SIZES,
    NEGATIVE_LABEL,
    NUM_BINARY_CLASSES,
    POSITIVE_LABEL,
    RDD_EVALUATION_RANKING_METRIC,
    RDD_EVALUATION_TOP_K,
    RDD_MODEL_MODE,
    RDD_MODEL_NAMES,
    RDD_SINGLE_MODEL,
    REQUIRED_MANIFEST_COLUMNS,
)


########### Manifest Validation ###########


def validate_manifest_columns(
    fieldnames: list[str] | None,
    manifest_path: Path,
) -> None:
    if fieldnames is None:
        raise ValueError(f"Manifest has no header: {manifest_path}")

    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(fieldnames)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {', '.join(sorted(missing_columns))}"
        )


def parse_binary_label(raw_label: str, row_number: int) -> int:
    try:
        label = int(raw_label)
    except ValueError as exc:
        raise ValueError(f"Invalid label on row {row_number}: {raw_label}") from exc

    if label not in {NEGATIVE_LABEL, POSITIVE_LABEL}:
        raise ValueError(
            f"Binary pothole label must be {NEGATIVE_LABEL} or {POSITIVE_LABEL} "
            f"on row {row_number}."
        )
    return label


########### Manifest Paths ###########


def expected_label_name(label: int) -> str:
    return "pothole" if label == POSITIVE_LABEL else "non_pothole"


def resolve_manifest_path(
    raw_path: str,
    project_root: Path,
    manifest_path: Path,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    project_relative_path = project_root / path
    if project_relative_path.exists():
        return project_relative_path

    return manifest_path.parent / path


def validate_existing_file(path: Path, column_name: str, row_number: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{column_name} on row {row_number} does not exist: {path}"
        )


########### Images ###########


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def make_image_transform(model_name: str, is_train: bool):
    from torchvision import transforms

    image_size = MODEL_IMAGE_SIZES.get(model_name, DEFAULT_IMAGE_SIZE)
    augmentation = (
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if is_train
        else [
            transforms.Resize((image_size, image_size)),
        ]
    )

    return transforms.Compose(
        [
            *augmentation,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def collate_binary_pothole_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.as_tensor(
            [int(item["label"]) for item in batch],
            dtype=torch.long,
        ),
        "image_path": [item["image_path"] for item in batch],
        "label_name": [item["label_name"] for item in batch],
        "country": [item["country"] for item in batch],
        "split": [item["split"] for item in batch],
    }


########### Manifest Rows ###########


def parse_manifest_row(
    row: dict[str, str],
    row_number: int,
    manifest_path: Path,
    sample_class,
    project_root: Path = PROJECT_ROOT,
    expected_split: str | None = None,
):
    split = row["split"].strip()
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"Expected split {expected_split}, got {split} on row {row_number}."
        )

    label = parse_binary_label(row["label"], row_number)
    label_name = row["label_name"].strip()
    expected_name = expected_label_name(label)
    if label_name != expected_name:
        raise ValueError(
            f"label_name must be {expected_name} for label {label} "
            f"on row {row_number}."
        )

    image_path = resolve_manifest_path(row["image_path"], project_root, manifest_path)
    annotation_path = resolve_manifest_path(
        row["annotation_path"],
        project_root,
        manifest_path,
    )
    validate_existing_file(image_path, "image_path", row_number)
    validate_existing_file(annotation_path, "annotation_path", row_number)

    country = row["country"].strip()
    if not country:
        raise ValueError(f"Missing country on row {row_number}.")

    return sample_class(
        image_path=image_path,
        annotation_path=annotation_path,
        label=label,
        label_name=label_name,
        country=country,
        split=split,
    )


def validate_manifest_split(
    samples,
    expected_split: str,
    manifest_path: Path,
) -> None:
    bad_splits = sorted(
        {sample.split for sample in samples if sample.split != expected_split}
    )
    if bad_splits:
        raise ValueError(
            f"Manifest {manifest_path} contains split values other than "
            f"{expected_split}: {', '.join(bad_splits)}"
        )


########### Model Adaptation ###########


def require_attribute(model: Any, attribute_name: str) -> Any:
    if not hasattr(model, attribute_name):
        raise ValueError(f"Model has no {attribute_name} attribute.")
    return getattr(model, attribute_name)


def require_linear_attribute(model: Any, attribute_name: str) -> nn.Linear:
    layer = require_attribute(model, attribute_name)
    if not isinstance(layer, nn.Linear):
        raise ValueError(f"Expected {attribute_name} to be nn.Linear.")
    return layer


def make_replacement_linear(layer: nn.Linear, num_classes: int) -> nn.Linear:
    return nn.Linear(
        in_features=layer.in_features,
        out_features=num_classes,
        bias=layer.bias is not None,
    )


def replace_linear_attribute(
    model: Any,
    attribute_name: str,
    num_classes: int,
) -> None:
    layer = require_linear_attribute(model, attribute_name)
    setattr(model, attribute_name, make_replacement_linear(layer, num_classes))


def replace_last_linear(module: Any, num_classes: int) -> None:
    if isinstance(module, nn.Linear):
        raise ValueError("Expected a classifier container, received nn.Linear.")
    if not hasattr(module, "__len__") or not hasattr(module, "__getitem__"):
        raise ValueError("Classifier does not support indexed layer replacement.")

    for index in reversed(range(len(module))):
        layer = module[index]
        if isinstance(layer, nn.Linear):
            module[index] = make_replacement_linear(layer, num_classes)
            return

    raise ValueError("Classifier contains no nn.Linear layer to replace.")


def update_hugging_face_label_config(model: Any, num_classes: int) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return

    config.num_labels = num_classes
    config.id2label = dict(ID_TO_LABEL)
    config.label2id = dict(LABEL_TO_ID)


def adapt_model_for_binary_pothole(model_name: str, model: Any) -> Any:
    from src.rdd_training.model_adapter import BinaryPotholeModelAdapter

    return BinaryPotholeModelAdapter(model_name).adapt(model)


def load_and_adapt_model_for_binary_pothole(model_name: str) -> Any:
    from src.load_model import load_model

    model = load_model(model_name)
    return adapt_model_for_binary_pothole(model_name, model)


def load_and_adapt_all_binary_pothole_models(
    model_names: Iterable[str] = RDD_MODEL_NAMES,
) -> dict[str, Any]:
    adapted_models = {}

    for model_name in sorted(model_names):
        adapted_models[model_name] = load_and_adapt_model_for_binary_pothole(model_name)

    return adapted_models


########### Training Helpers ###########


def select_rdd_model_names(
    mode: str = RDD_MODEL_MODE,
    model_names: Iterable[str] = RDD_MODEL_NAMES,
    single_model: str = RDD_SINGLE_MODEL,
) -> tuple[str, ...]:
    if mode == "single":
        return (single_model,)
    if mode == "all":
        return tuple(model_names)

    raise ValueError("RDD_MODEL_MODE must be 'single' or 'all'.")


def calculate_class_weights(class_counts: dict[int, int]) -> torch.Tensor:
    total_count = sum(class_counts.get(label, 0) for label in range(NUM_BINARY_CLASSES))
    if total_count == 0:
        raise ValueError("Cannot calculate class weights for an empty dataset.")

    weights = []
    for label in range(NUM_BINARY_CLASSES):
        label_count = class_counts.get(label, 0)
        if label_count == 0:
            raise ValueError(f"Cannot calculate class weight for missing label: {label}")
        weights.append(total_count / (NUM_BINARY_CLASSES * label_count))

    return torch.tensor(weights, dtype=torch.float32)


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
        rank for rank, label in zip(ranks, labels, strict=True) if label == POSITIVE_LABEL
    )
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def compute_binary_roc_curve(
    positive_scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[float], list[float]]:
    scores = positive_scores.detach().cpu().float()
    labels = targets.detach().cpu().long()
    thresholds = torch.cat(
        [
            torch.tensor([float("inf")]),
            torch.sort(torch.unique(scores), descending=True).values,
            torch.tensor([float("-inf")]),
        ]
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


########### Training Outputs ###########


def write_training_loss_plot(
    plot_path: Path,
    history: list[dict[str, Any]],
) -> None:
    write_training_metric_plot(
        plot_path=plot_path,
        history=history,
        metric_name="loss",
        y_label="Loss",
        title="RDD Binary Pothole Training Loss",
    )


def write_training_metric_plot(
    plot_path: Path,
    history: list[dict[str, Any]],
    metric_name: str,
    y_label: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    train_values = [float(row[f"train_{metric_name}"]) for row in history]
    validation_values = [
        float(row[f"validation_{metric_name}"]) for row in history
    ]

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_values, marker="o", label=f"train {metric_name}")
    plt.plot(
        epochs,
        validation_values,
        marker="o",
        label=f"validation {metric_name}",
    )
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


########### Evaluation Outputs ###########


def write_evaluation_metrics_json(
    metrics_path: Path,
    metrics: dict[str, float],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(sanitize_metrics_for_json(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_evaluation_metrics_csv(
    metrics_path: Path,
    model_name: str,
    split: str,
    metrics: dict[str, float],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model_name": model_name, "split": split, **metrics}

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_confusion_matrix_csv(
    confusion_matrix_path: Path,
    metrics: dict[str, float],
) -> None:
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "actual": "non_pothole",
            "predicted_non_pothole": int(metrics["true_negative"]),
            "predicted_pothole": int(metrics["false_positive"]),
        },
        {
            "actual": "pothole",
            "predicted_non_pothole": int(metrics["false_negative"]),
            "predicted_pothole": int(metrics["true_positive"]),
        },
    ]

    with confusion_matrix_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as confusion_file:
        writer = csv.DictWriter(confusion_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def write_metric_bar_plot(
    plot_path: Path,
    metrics: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    metric_names = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_values = [metrics[metric_name] for metric_name in metric_names]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=metric_names, y=metric_values, hue=metric_names, palette="viridis")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("RDD Binary Pothole Test Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def write_model_comparison_csv(
    comparison_path: Path,
    evaluation_rows: list[dict[str, Any]],
    ranking_metric: str = RDD_EVALUATION_RANKING_METRIC,
    top_k: int = RDD_EVALUATION_TOP_K,
) -> list[dict[str, Any]]:
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_rows = rank_evaluation_rows(evaluation_rows, ranking_metric)

    with comparison_path.open("w", newline="", encoding="utf-8") as comparison_file:
        writer = csv.DictWriter(comparison_file, fieldnames=list(ranked_rows[0]))
        writer.writeheader()
        writer.writerows(ranked_rows)

    top_models_path = comparison_path.with_name("top_models.csv")
    with top_models_path.open("w", newline="", encoding="utf-8") as top_models_file:
        writer = csv.DictWriter(top_models_file, fieldnames=list(ranked_rows[0]))
        writer.writeheader()
        writer.writerows(ranked_rows[:top_k])

    return ranked_rows


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


def sanitize_metrics_for_json(metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        key: None if isinstance(value, float) and math.isnan(value) else value
        for key, value in metrics.items()
    }
