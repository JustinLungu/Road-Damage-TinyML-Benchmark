from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from src.constants import RDD2022_BINARY_POTHOLE_DIR, RDD_TRAINING_RESULTS_DIR
from src.rdd_benchmark.constants import RDD_PREDICTION_MODEL_DISPLAY_NAMES
from src.rdd_benchmark.data_loader.dataset import BinaryPotholeManifest
from src.rdd_benchmark.data_loader.utils import load_rgb_image
from src.rdd_benchmark.training.prediction_export import (
    PREDICTION_COLUMNS,
    RDDPredictionExportConfig,
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

        manifest = BinaryPotholeManifest.from_csv(
            self.config.test_manifest_path,
            expected_split="test",
        )
        if not 1 <= self.config.sample_count <= len(manifest):
            raise ValueError(
                f"sample_count must be between 1 and {len(manifest)}, "
                f"received {self.config.sample_count}."
            )

        prediction_rows = {
            model_name: self._read_predictions(model_name, experiment_id)
            for model_name, experiment_id in self.config.model_experiments.items()
        }
        for model_name, rows in prediction_rows.items():
            if len(rows) != len(manifest):
                raise ValueError(
                    f"{model_name} CSV has {len(rows)} rows, but the test manifest "
                    f"has {len(manifest)}. Run export-predictions again."
                )
        actual_classes = [
            [row["actual_class"] for row in rows] for rows in prediction_rows.values()
        ]
        if any(classes != actual_classes[0] for classes in actual_classes[1:]):
            raise ValueError(
                "Prediction CSV actual classes do not match row-for-row. "
                "Run export-predictions again."
            )
        manifest_image_ids = [sample.image_path.stem for sample in manifest.samples]
        for model_name, rows in prediction_rows.items():
            csv_image_ids = [row["image_id"] for row in rows]
            if csv_image_ids != manifest_image_ids:
                raise ValueError(
                    f"{model_name} CSV image IDs do not match the test manifest. "
                    "Run export-predictions again."
                )

        sample_indices = random.sample(range(len(manifest)), self.config.sample_count)
        columns = min(4, self.config.sample_count)
        rows = math.ceil(self.config.sample_count / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows))
        axes = [axes] if not hasattr(axes, "flat") else list(axes.flat)

        for axis, index in zip(axes, sample_indices, strict=False):
            sample = manifest.samples[index]
            axis.imshow(load_rgb_image(sample.image_path))
            title_lines = [
                f"Image ID: {manifest_image_ids[index]}",
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
            return list(reader)
