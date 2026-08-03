from __future__ import annotations

from typing import Any

import torch

from src.vision_benchmark.classification_dataset import ClassificationSample
from src.vision_benchmark.classification_inference_adapter import (
    ClassificationInferenceAdapter,
)
from src.vision_benchmark.results import ClassificationBenchmarkResult
from src.vision_benchmark.utils import mean, print_progress


class ClassificationBenchmark:
    """Evaluate ImageNet-style classifiers against zero-based class IDs."""

    def __init__(
        self,
        model_name: str,
        model: Any,
        samples: list[ClassificationSample],
        device: torch.device,
        dataset_name: str,
        split: str,
    ) -> None:
        if not samples:
            raise ValueError("At least one classification sample is required.")

        self.model_name = model_name
        self.samples = samples
        self.dataset_name = dataset_name
        self.split = split
        self.adapter = ClassificationInferenceAdapter(model_name, model, device)

    def run(self) -> ClassificationBenchmarkResult:
        true_labels = []
        predicted_labels = []
        top5_correct = 0

        with torch.inference_mode():
            for completed, sample in enumerate(self.samples, start=1):
                logits = self.adapter.predict(sample.image_path)
                if sample.class_id >= logits.numel():
                    raise ValueError(
                        f"class_id {sample.class_id} is outside the model's "
                        f"{logits.numel()} output classes."
                    )

                top_k = min(5, logits.numel())
                top_predictions = torch.topk(logits, k=top_k).indices.tolist()
                true_labels.append(sample.class_id)
                predicted_labels.append(top_predictions[0])
                top5_correct += int(sample.class_id in top_predictions)
                print_progress(completed, len(self.samples))

        top1_accuracy = sum(
            predicted == target
            for predicted, target in zip(predicted_labels, true_labels)
        ) / len(true_labels)
        precision, recall, f1_score = calculate_macro_classification_metrics(
            true_labels,
            predicted_labels,
        )

        return ClassificationBenchmarkResult(
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            split=self.split,
            num_images=len(self.samples),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            top1_accuracy=top1_accuracy,
            top5_accuracy=top5_correct / len(true_labels),
        )


def calculate_macro_classification_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
) -> tuple[float, float, float]:
    if not true_labels or len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted labels must have equal non-zero length.")

    precisions = []
    recalls = []
    f1_scores = []

    # Average over classes represented in the ground truth subset.
    for class_id in sorted(set(true_labels)):
        true_positive = sum(
            target == class_id and prediction == class_id
            for target, prediction in zip(true_labels, predicted_labels)
        )
        false_positive = sum(
            target != class_id and prediction == class_id
            for target, prediction in zip(true_labels, predicted_labels)
        )
        false_negative = sum(
            target == class_id and prediction != class_id
            for target, prediction in zip(true_labels, predicted_labels)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return mean(precisions), mean(recalls), mean(f1_scores)
