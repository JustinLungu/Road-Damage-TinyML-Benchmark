from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import RDD2022_BINARY_POTHOLE_DIR, RDD_TRAINING_RESULTS_DIR
from src.rdd_benchmark.constants import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RDD_EVALUATION_BATCH_SIZE,
    RDD_EVALUATION_NUM_WORKERS,
    RDD_EVALUATION_PROGRESS_INTERVAL,
    RDD_EVALUATION_SAVE_PLOTS,
)
from src.rdd_benchmark.data_loader.dataset import (
    BinaryPotholeDataset,
    BinaryPotholeManifest,
)
from src.rdd_benchmark.data_loader.utils import collate_binary_pothole_batch
from src.rdd_benchmark.training.utils import (
    compute_binary_classification_metrics,
    compute_binary_roc_auc,
    compute_binary_roc_curve,
    extract_logits,
    load_and_adapt_model_for_binary_pothole,
    make_image_transform,
    write_confusion_matrix_csv,
    write_confusion_matrix_plot,
    write_evaluation_metrics_csv,
    write_evaluation_metrics_json,
    write_metric_bar_plot,
    write_roc_curve_plot,
)


@dataclass(frozen=True)
class RDDEvaluationConfig:
    model_name: str
    experiment_name: str = "default"
    test_manifest_path: Path = RDD2022_BINARY_POTHOLE_DIR / "test.csv"
    output_dir: Path = RDD_TRAINING_RESULTS_DIR
    batch_size: int = RDD_EVALUATION_BATCH_SIZE
    num_workers: int = RDD_EVALUATION_NUM_WORKERS
    progress_interval: int = RDD_EVALUATION_PROGRESS_INTERVAL
    save_plots: bool = RDD_EVALUATION_SAVE_PLOTS
    save_balanced_test_metrics: bool = True
    balanced_test_random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_output_dir(self) -> Path:
        return self.output_dir / self.experiment_name / self.model_name

    @property
    def checkpoint_path(self) -> Path:
        return self.model_output_dir / "best.pt"


@dataclass(frozen=True)
class RDDEvaluationResult:
    model_name: str
    checkpoint_path: Path
    metrics: dict[str, float | str]
    metrics_json_path: Path
    metrics_csv_path: Path
    confusion_matrix_path: Path
    confusion_matrix_plot_path: Path | None
    roc_curve_plot_path: Path | None
    metric_bar_plot_path: Path | None
    evaluation_timing_path: Path

    def comparison_row(self) -> dict[str, float | str]:
        return {
            "model_name": self.model_name,
            "checkpoint_path": str(self.checkpoint_path),
            **self.metrics,
        }


class BinaryPotholeEvaluator:
    """Evaluate one trained classifier on realistic and balanced RDD test views."""

    def __init__(
        self,
        config: RDDEvaluationConfig,
        model: nn.Module | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model

    def evaluate(self) -> RDDEvaluationResult:
        if not self.config.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained checkpoint does not exist: {self.config.checkpoint_path}"
            )

        manifest = BinaryPotholeManifest.from_csv(
            self.config.test_manifest_path,
            expected_split="test",
        )
        data_loader = self._make_data_loader(manifest)
        model = self._load_model().to(self.device)
        model.eval()

        print(
            "  evaluation setup: "
            f"device={self.device}, test_samples={len(manifest)}, "
            f"test_batches={len(data_loader)}, checkpoint={self.config.checkpoint_path}"
        )

        predictions = []
        targets = []
        positive_scores = []
        self._synchronize_device()
        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_index, batch in enumerate(data_loader, start=1):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = extract_logits(model(images))
                predictions.append(logits.argmax(dim=1))
                targets.append(labels)
                positive_scores.append(torch.softmax(logits, dim=1)[:, POSITIVE_LABEL])
                self._synchronize_device()

                if self._should_print_progress(batch_index, len(data_loader)):
                    print(f"  evaluated batches: {batch_index}/{len(data_loader)}")

        all_predictions = torch.cat(predictions)
        all_targets = torch.cat(targets)
        all_scores = torch.cat(positive_scores)
        total_seconds = time.perf_counter() - start_time
        metrics = self._calculate_metrics(all_predictions, all_targets, all_scores)
        evaluation_timing = self._make_evaluation_timing(len(manifest), total_seconds)
        metrics.update(
            {"experiment_name": self.config.experiment_name, **evaluation_timing}
        )

        evaluation_timing_path = self.config.model_output_dir / "evaluation_timing.json"
        write_evaluation_metrics_json(evaluation_timing_path, evaluation_timing)
        self._write_balanced_test_outputs(
            manifest,
            all_predictions,
            all_targets,
            all_scores,
        )
        return self._write_evaluation_outputs(metrics, all_scores, all_targets)

    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        positive_scores: torch.Tensor,
    ) -> dict[str, float]:
        metrics = compute_binary_classification_metrics(predictions, targets)
        metrics["roc_auc"] = compute_binary_roc_auc(positive_scores, targets)
        return metrics

    def _write_evaluation_outputs(
        self,
        metrics: dict[str, float | str],
        positive_scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> RDDEvaluationResult:
        output_dir = self.config.model_output_dir
        metrics_json_path = output_dir / "test_metrics.json"
        metrics_csv_path = output_dir / "test_metrics.csv"
        confusion_matrix_path = output_dir / "confusion_matrix.csv"
        confusion_matrix_plot_path = output_dir / "confusion_matrix.png"
        roc_curve_plot_path = output_dir / "roc_curve.png"
        metric_bar_plot_path = output_dir / "test_metric_bars.png"

        write_evaluation_metrics_json(metrics_json_path, metrics)
        write_evaluation_metrics_csv(
            metrics_csv_path,
            self.config.model_name,
            "test",
            metrics,
        )
        write_confusion_matrix_csv(confusion_matrix_path, metrics)
        if self.config.save_plots:
            false_positive_rates, true_positive_rates = compute_binary_roc_curve(
                positive_scores,
                targets,
            )
            write_confusion_matrix_plot(confusion_matrix_plot_path, metrics)
            write_roc_curve_plot(
                roc_curve_plot_path,
                false_positive_rates,
                true_positive_rates,
                float(metrics["roc_auc"]),
            )
            write_metric_bar_plot(metric_bar_plot_path, metrics)
        else:
            confusion_matrix_plot_path = None
            roc_curve_plot_path = None
            metric_bar_plot_path = None

        print(
            "  test metrics: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

        return RDDEvaluationResult(
            model_name=self.config.model_name,
            checkpoint_path=self.config.checkpoint_path,
            metrics=metrics,
            metrics_json_path=metrics_json_path,
            metrics_csv_path=metrics_csv_path,
            confusion_matrix_path=confusion_matrix_path,
            confusion_matrix_plot_path=confusion_matrix_plot_path,
            roc_curve_plot_path=roc_curve_plot_path,
            metric_bar_plot_path=metric_bar_plot_path,
            evaluation_timing_path=output_dir / "evaluation_timing.json",
        )

    def _write_balanced_test_outputs(
        self,
        manifest: BinaryPotholeManifest,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        positive_scores: torch.Tensor,
    ) -> None:
        if not self.config.save_balanced_test_metrics:
            return

        indices = select_balanced_test_indices(
            manifest,
            self.config.balanced_test_random_seed,
        )
        if not indices:
            return

        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=predictions.device
        )
        balanced_predictions = predictions.index_select(0, index_tensor)
        balanced_targets = targets.index_select(0, index_tensor)
        balanced_scores = positive_scores.index_select(0, index_tensor)
        metrics = self._calculate_metrics(
            balanced_predictions,
            balanced_targets,
            balanced_scores,
        )
        metrics.update(
            {
                "experiment_name": self.config.experiment_name,
                "test_view": "balanced",
                "num_images": float(len(indices)),
            }
        )

        output_dir = self.config.model_output_dir
        write_evaluation_metrics_json(
            output_dir / "balanced_test_metrics.json", metrics
        )
        write_evaluation_metrics_csv(
            output_dir / "balanced_test_metrics.csv",
            self.config.model_name,
            "balanced_test",
            metrics,
        )
        write_confusion_matrix_csv(
            output_dir / "balanced_confusion_matrix.csv", metrics
        )
        if self.config.save_plots:
            false_positive_rates, true_positive_rates = compute_binary_roc_curve(
                balanced_scores,
                balanced_targets,
            )
            write_confusion_matrix_plot(
                output_dir / "balanced_confusion_matrix.png",
                metrics,
            )
            write_roc_curve_plot(
                output_dir / "balanced_roc_curve.png",
                false_positive_rates,
                true_positive_rates,
                float(metrics["roc_auc"]),
            )
            write_metric_bar_plot(output_dir / "balanced_test_metric_bars.png", metrics)

        print(
            "  balanced test metrics: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    def _make_evaluation_timing(
        self,
        image_count: int,
        total_seconds: float,
    ) -> dict[str, float]:
        return {
            "num_images": float(image_count),
            "total_evaluation_seconds": total_seconds,
            "avg_image_processing_ms": (
                total_seconds * 1000 / image_count if image_count else 0.0
            ),
            "evaluation_images_per_second": (
                image_count / total_seconds if total_seconds else 0.0
            ),
        }

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _load_model(self) -> nn.Module:
        model = self.model or load_and_adapt_model_for_binary_pothole(
            self.config.model_name
        )
        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _make_data_loader(self, manifest: BinaryPotholeManifest) -> DataLoader:
        dataset = BinaryPotholeDataset(
            manifest,
            transform=make_image_transform(self.config.model_name, is_train=False),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_binary_pothole_batch,
        )

    def _should_print_progress(self, batch_index: int, total_batches: int) -> bool:
        return batch_index == total_batches or (
            self.config.progress_interval > 0
            and batch_index % self.config.progress_interval == 0
        )


def select_balanced_test_indices(
    manifest: BinaryPotholeManifest,
    random_seed: int = 42,
) -> list[int]:
    positive_indices = [
        index
        for index, sample in enumerate(manifest.samples)
        if sample.label == POSITIVE_LABEL
    ]
    negative_indices = [
        index
        for index, sample in enumerate(manifest.samples)
        if sample.label == NEGATIVE_LABEL
    ]
    balanced_count = min(len(positive_indices), len(negative_indices))
    if balanced_count == 0:
        return []

    rng = random.Random(f"{random_seed}:balanced_test")
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)
    return sorted(positive_indices[:balanced_count] + negative_indices[:balanced_count])
