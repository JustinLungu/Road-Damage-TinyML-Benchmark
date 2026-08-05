from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import RDD2022_BINARY_POTHOLE_DIR, RDD_TRAINING_RESULTS_DIR
from src.rdd_benchmark.constants import (
    ID_TO_LABEL,
    RDD_EVALUATION_BATCH_SIZE,
    RDD_EVALUATION_NUM_WORKERS,
    RDD_EVALUATION_PROGRESS_INTERVAL,
    RDD_PREDICTION_MODEL_DISPLAY_NAMES,
)
from src.rdd_benchmark.data_loader.dataset import (
    BinaryPotholeDataset,
    BinaryPotholeManifest,
)
from src.rdd_benchmark.data_loader.utils import (
    collate_binary_pothole_batch,
    load_rgb_image,
)
from src.rdd_benchmark.experiments.experiment_registry import (
    get_rdd_experiment_config,
)
from src.rdd_benchmark.training.model_adapter import (
    load_and_adapt_model_for_binary_pothole,
)
from src.rdd_benchmark.training.utils import extract_logits, make_image_transform


PREDICTION_COLUMNS = (
    "image_id",
    "non_pothole_softmax",
    "pothole_softmax",
    "predicted_class",
    "actual_class",
)


@dataclass(frozen=True)
class RDDPredictionExportConfig:
    model_name: str
    experiment_id: str
    test_manifest_path: Path = RDD2022_BINARY_POTHOLE_DIR / "test.csv"
    output_dir: Path = RDD_TRAINING_RESULTS_DIR
    batch_size: int = RDD_EVALUATION_BATCH_SIZE
    num_workers: int = RDD_EVALUATION_NUM_WORKERS
    progress_interval: int = RDD_EVALUATION_PROGRESS_INTERVAL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def experiment_name(self) -> str:
        return get_rdd_experiment_config(self.experiment_id).experiment_name

    @property
    def model_output_dir(self) -> Path:
        return self.output_dir / self.experiment_name / self.model_name

    @property
    def checkpoint_path(self) -> Path:
        return self.model_output_dir / "best.pt"

    @property
    def predictions_path(self) -> Path:
        return self.model_output_dir / "test_predictions.csv"


class RDDPredictionExporter:
    """Run one checkpoint over every test image and export its predictions."""

    def __init__(
        self,
        config: RDDPredictionExportConfig,
        model: nn.Module | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model

    def export(self) -> Path:
        if not self.config.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained checkpoint does not exist: {self.config.checkpoint_path}"
            )

        manifest = BinaryPotholeManifest.from_csv(
            self.config.test_manifest_path,
            expected_split="test",
        )
        data_loader = self._make_data_loader(manifest)
        model = self._load_model().to(self.device).eval()

        probabilities = []
        targets = []
        print(
            f"Evaluating {self.config.model_name} ({self.config.experiment_id}) "
            f"on all {len(manifest)} test images using {self.device}"
        )
        with torch.inference_mode():
            for batch_index, batch in enumerate(data_loader, start=1):
                logits = extract_logits(model(batch["image"].to(self.device)))
                probabilities.append(torch.softmax(logits, dim=1).cpu())
                targets.append(batch["label"].cpu())

                if self._should_print_progress(batch_index, len(data_loader)):
                    print(f"  evaluated batches: {batch_index}/{len(data_loader)}")

        all_probabilities = torch.cat(probabilities)
        all_targets = torch.cat(targets)
        if len(all_probabilities) != len(manifest):
            raise RuntimeError(
                f"Expected {len(manifest)} predictions, "
                f"received {len(all_probabilities)}."
            )

        self._write_predictions(manifest, all_probabilities, all_targets)
        print(f"  predictions: {self.config.predictions_path}")
        return self.config.predictions_path

    def _load_model(self) -> nn.Module:
        model = self.model or load_and_adapt_model_for_binary_pothole(
            self.config.model_name
        )
        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name not in {None, self.config.model_name}:
            raise ValueError(
                f"Checkpoint contains {checkpoint_model_name}, expected "
                f"{self.config.model_name}."
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

    def _write_predictions(
        self,
        manifest: BinaryPotholeManifest,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self.config.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions = probabilities.argmax(dim=1)
        with self.config.predictions_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for sample, scores, prediction, target in zip(
                manifest.samples,
                probabilities,
                predictions,
                targets,
                strict=True,
            ):
                writer.writerow(
                    {
                        "image_id": sample.image_path.stem,
                        "non_pothole_softmax": f"{scores[0].item():.8f}",
                        "pothole_softmax": f"{scores[1].item():.8f}",
                        "predicted_class": ID_TO_LABEL[int(prediction.item())],
                        "actual_class": ID_TO_LABEL[int(target.item())],
                    }
                )

    def _should_print_progress(self, batch_index: int, total_batches: int) -> bool:
        return batch_index == total_batches or (
            self.config.progress_interval > 0
            and batch_index % self.config.progress_interval == 0
        )


@dataclass(frozen=True)
class RDDPredictionPreviewConfig:
    model_experiments: dict[str, str]
    sample_count: int
    test_manifest_path: Path = RDD2022_BINARY_POTHOLE_DIR / "test.csv"
    output_dir: Path = RDD_TRAINING_RESULTS_DIR

    @property
    def preview_path(self) -> Path:
        return self.output_dir / "prediction_preview.png"


class RDDPredictionPreviewer:
    """Create a random visual comparison from previously exported CSV rows."""

    def __init__(self, config: RDDPredictionPreviewConfig) -> None:
        self.config = config

    def create(self) -> Path:
        import matplotlib.pyplot as plt

        if not self.config.model_experiments:
            raise ValueError("At least one prediction model is required.")

        manifest = BinaryPotholeManifest.from_csv(
            self.config.test_manifest_path,
            expected_split="test",
        )
        if not 1 <= self.config.sample_count <= len(manifest):
            raise ValueError(
                f"sample_count must be between 1 and {len(manifest)}, "
                f"received {self.config.sample_count}."
            )

        image_ids = [sample.image_path.stem for sample in manifest.samples]
        prediction_rows = {
            model_name: self._read_predictions(
                model_name,
                experiment_id,
                image_ids,
            )
            for model_name, experiment_id in self.config.model_experiments.items()
        }
        actual_classes = [
            [row["actual_class"] for row in rows] for rows in prediction_rows.values()
        ]
        if any(classes != actual_classes[0] for classes in actual_classes[1:]):
            raise ValueError(
                "Prediction CSV actual classes do not match row-for-row. "
                "Run export-predictions again."
            )

        sample_indices = random.SystemRandom().sample(
            range(len(manifest)),
            self.config.sample_count,
        )
        columns = min(4, self.config.sample_count)
        rows = math.ceil(self.config.sample_count / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows))
        axes = [axes] if not hasattr(axes, "flat") else list(axes.flat)

        for axis, index in zip(axes, sample_indices, strict=False):
            sample = manifest.samples[index]
            axis.imshow(load_rgb_image(sample.image_path))
            title_lines = [
                f"Image ID: {image_ids[index]}",
                f"Actual class: {actual_classes[0][index]}",
            ]
            for model_name, model_rows in prediction_rows.items():
                prediction = model_rows[index]
                predicted_class = prediction["predicted_class"]
                confidence = float(prediction[f"{predicted_class}_softmax"])
                display_name = RDD_PREDICTION_MODEL_DISPLAY_NAMES.get(
                    model_name,
                    model_name,
                )
                title_lines.append(
                    f"{display_name} Prediction: {predicted_class} ({confidence:.2%})"
                )
            axis.set_title("\n".join(title_lines), fontsize=8.5)
            axis.axis("off")

        for axis in axes[len(sample_indices) :]:
            axis.axis("off")

        figure.suptitle("RDD Test Prediction Samples", fontsize=16, y=0.98)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        self.config.preview_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.config.preview_path, dpi=160)
        plt.close(figure)
        print(
            f"Random preview ({self.config.sample_count} images): {self.config.preview_path}"
        )
        return self.config.preview_path

    def _read_predictions(
        self,
        model_name: str,
        experiment_id: str,
        expected_image_ids: list[str],
    ) -> list[dict[str, str]]:
        predictions_path = RDDPredictionExportConfig(
            model_name=model_name,
            experiment_id=experiment_id,
            output_dir=self.config.output_dir,
        ).predictions_path
        if not predictions_path.is_file():
            raise FileNotFoundError(
                f"Prediction CSV does not exist: {predictions_path}. "
                "Run export-predictions first."
            )

        with predictions_path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {predictions_path}: {reader.fieldnames}"
                )
            rows = list(reader)

        if [row["image_id"] for row in rows] != expected_image_ids:
            raise ValueError(
                f"{model_name} CSV image IDs do not match the test manifest. "
                "Run export-predictions again."
            )
        return rows
