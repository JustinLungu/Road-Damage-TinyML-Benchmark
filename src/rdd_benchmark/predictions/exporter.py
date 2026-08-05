from __future__ import annotations

import csv
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
)
from src.rdd_benchmark.data_loader.dataset import (
    BinaryPotholeDataset,
    BinaryPotholeManifest,
)
from src.rdd_benchmark.data_loader.utils import collate_binary_pothole_batch
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
