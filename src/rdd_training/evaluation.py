from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import RDD2022_BINARY_POTHOLE_DIR, RDD_TRAINING_RESULTS_DIR
from src.rdd_training.constants import (
    POSITIVE_LABEL,
    RDD_EVALUATION_BATCH_SIZE,
    RDD_EVALUATION_NUM_WORKERS,
    RDD_EVALUATION_PROGRESS_INTERVAL,
    RDD_EVALUATION_SAVE_PLOTS,
    RDD_GRID_OVERLAP,
    RDD_GRID_SIZE,
    RDD_PATCH_DECISION_THRESHOLD,
)
from src.rdd_training.dataset import BinaryPotholeDataset, BinaryPotholeManifest
from src.rdd_training.utils import (
    collate_binary_pothole_batch,
    compute_binary_classification_metrics,
    compute_binary_roc_auc,
    compute_binary_roc_curve,
    extract_logits,
    load_rgb_image,
    load_and_adapt_model_for_binary_pothole,
    make_image_transform,
    write_confusion_matrix_csv,
    write_confusion_matrix_plot,
    write_evaluation_metrics_csv,
    write_evaluation_metrics_json,
    write_metric_bar_plot,
    write_roc_curve_plot,
)


PatchBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class GridImagePrediction:
    image_path: Path
    patch_boxes: tuple[PatchBox, ...]
    patch_scores: tuple[float, ...]
    image_score: float
    prediction: int


class GridPatchGenerator:
    """Generate fixed grid crop boxes covering a full image."""

    def __init__(
        self,
        grid_size: int = RDD_GRID_SIZE,
        overlap: float = RDD_GRID_OVERLAP,
    ) -> None:
        if grid_size < 1:
            raise ValueError("grid_size must be at least one.")
        if overlap != 0.0:
            raise NotImplementedError("Only non-overlapping grid patches are supported.")

        self.grid_size = grid_size
        self.overlap = overlap

    def generate(self, image_width: int, image_height: int) -> tuple[PatchBox, ...]:
        if image_width < 1 or image_height < 1:
            raise ValueError("image_width and image_height must be positive.")

        x_edges = self._make_edges(image_width)
        y_edges = self._make_edges(image_height)

        return tuple(
            (
                x_edges[column],
                y_edges[row],
                x_edges[column + 1],
                y_edges[row + 1],
            )
            for row in range(self.grid_size)
            for column in range(self.grid_size)
        )

    def _make_edges(self, size: int) -> list[int]:
        return [
            round(index * size / self.grid_size)
            for index in range(self.grid_size + 1)
        ]


class GridImageInferenceRunner:
    """Run a binary patch classifier over grid patches from a full image."""

    def __init__(
        self,
        model: nn.Module,
        transform,
        device: torch.device | str,
        patch_generator: GridPatchGenerator | None = None,
        decision_threshold: float = RDD_PATCH_DECISION_THRESHOLD,
    ) -> None:
        self.model = model
        self.transform = transform
        self.device = torch.device(device)
        self.patch_generator = patch_generator or GridPatchGenerator()
        self.decision_threshold = decision_threshold

    def predict_image(self, image_path: Path) -> GridImagePrediction:
        image = load_rgb_image(image_path)
        patch_boxes = self.patch_generator.generate(*image.size)
        patch_tensors = [
            self.transform(image.crop(patch_box))
            for patch_box in patch_boxes
        ]
        images = torch.stack(patch_tensors).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = extract_logits(self.model(images))
            probabilities = torch.softmax(logits, dim=1)
            positive_scores = probabilities[:, POSITIVE_LABEL].detach().cpu()

        image_score = float(positive_scores.max().item())
        prediction = int(image_score >= self.decision_threshold)

        return GridImagePrediction(
            image_path=image_path,
            patch_boxes=patch_boxes,
            patch_scores=tuple(float(score) for score in positive_scores.tolist()),
            image_score=image_score,
            prediction=prediction,
        )


@dataclass(frozen=True)
class RDDEvaluationConfig:
    model_name: str
    test_manifest_path: Path = RDD2022_BINARY_POTHOLE_DIR / "test.csv"
    output_dir: Path = RDD_TRAINING_RESULTS_DIR
    batch_size: int = RDD_EVALUATION_BATCH_SIZE
    num_workers: int = RDD_EVALUATION_NUM_WORKERS
    progress_interval: int = RDD_EVALUATION_PROGRESS_INTERVAL
    save_plots: bool = RDD_EVALUATION_SAVE_PLOTS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_output_dir(self) -> Path:
        return self.output_dir / self.model_name

    @property
    def checkpoint_path(self) -> Path:
        return self.model_output_dir / "best.pt"


@dataclass(frozen=True)
class RDDEvaluationResult:
    model_name: str
    checkpoint_path: Path
    metrics: dict[str, float]
    metrics_json_path: Path
    metrics_csv_path: Path
    confusion_matrix_path: Path
    confusion_matrix_plot_path: Path | None
    roc_curve_plot_path: Path | None
    metric_bar_plot_path: Path | None

    def comparison_row(self) -> dict[str, float | str]:
        return {
            "model_name": self.model_name,
            "checkpoint_path": str(self.checkpoint_path),
            **self.metrics,
        }


class BinaryPotholeEvaluator:
    """Evaluate one trained binary pothole classifier on the RDD test split."""

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

        test_manifest = BinaryPotholeManifest.from_csv(
            self.config.test_manifest_path,
            expected_split="test",
        )
        test_loader = self._make_data_loader(test_manifest)
        model = self._load_model().to(self.device)
        model.eval()

        print(
            "  evaluation setup: "
            f"device={self.device}, test_samples={len(test_manifest)}, "
            f"test_batches={len(test_loader)}, checkpoint={self.config.checkpoint_path}"
        )

        predictions = []
        targets = []
        positive_scores = []

        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader, start=1):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = extract_logits(model(images))
                probabilities = torch.softmax(logits, dim=1)

                predictions.append(logits.argmax(dim=1))
                targets.append(labels)
                positive_scores.append(probabilities[:, POSITIVE_LABEL])

                if self._should_print_batch_progress(batch_index, len(test_loader)):
                    print(f"  evaluated batches: {batch_index}/{len(test_loader)}")

        all_predictions = torch.cat(predictions)
        all_targets = torch.cat(targets)
        all_positive_scores = torch.cat(positive_scores)
        metrics = compute_binary_classification_metrics(
            predictions=all_predictions,
            targets=all_targets,
        )
        metrics["roc_auc"] = compute_binary_roc_auc(
            positive_scores=all_positive_scores,
            targets=all_targets,
        )
        false_positive_rates, true_positive_rates = compute_binary_roc_curve(
            positive_scores=all_positive_scores,
            targets=all_targets,
        )

        metrics_json_path = self.config.model_output_dir / "test_metrics.json"
        metrics_csv_path = self.config.model_output_dir / "test_metrics.csv"
        confusion_matrix_path = self.config.model_output_dir / "confusion_matrix.csv"
        confusion_matrix_plot_path = (
            self.config.model_output_dir / "confusion_matrix.png"
        )
        roc_curve_plot_path = self.config.model_output_dir / "roc_curve.png"
        metric_bar_plot_path = self.config.model_output_dir / "test_metric_bars.png"
        write_evaluation_metrics_json(metrics_json_path, metrics)
        write_evaluation_metrics_csv(
            metrics_csv_path,
            model_name=self.config.model_name,
            split="test",
            metrics=metrics,
        )
        write_confusion_matrix_csv(confusion_matrix_path, metrics)
        if self.config.save_plots:
            write_confusion_matrix_plot(confusion_matrix_plot_path, metrics)
            write_roc_curve_plot(
                roc_curve_plot_path,
                false_positive_rates=false_positive_rates,
                true_positive_rates=true_positive_rates,
                roc_auc=metrics["roc_auc"],
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
        )

    def _load_model(self) -> nn.Module:
        model = self.model
        if model is None:
            model = load_and_adapt_model_for_binary_pothole(self.config.model_name)

        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _make_data_loader(self, manifest: BinaryPotholeManifest) -> DataLoader:
        dataset = BinaryPotholeDataset(
            manifest=manifest,
            transform=make_image_transform(self.config.model_name, is_train=False),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_binary_pothole_batch,
        )

    def _should_print_batch_progress(
        self,
        batch_index: int,
        total_batches: int,
    ) -> bool:
        if batch_index == total_batches:
            return True
        if self.config.progress_interval <= 0:
            return False
        return batch_index % self.config.progress_interval == 0
