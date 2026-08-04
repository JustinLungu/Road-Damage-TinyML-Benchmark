import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.rdd_benchmark.training.prediction_export import (
    PREDICTION_COLUMNS,
    RDDPredictionExportConfig,
    RDDPredictionExporter,
)
from src.rdd_benchmark.training.prediction_preview import (
    RDDPredictionPreviewConfig,
    RDDPredictionPreviewer,
)


class BrightnessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, images):
        scores = images.mean(dim=(1, 2, 3)) + self.bias
        return torch.stack((-scores, scores), dim=1)


def write_test_manifest(tmp_path: Path) -> Path:
    rows = []
    for index, label in enumerate((0, 0, 0, 1, 1, 1)):
        image_path = tmp_path / "images" / f"{index}.jpg"
        annotation_path = tmp_path / "annotations" / f"{index}.xml"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=30 + index * 35).save(image_path)
        annotation_path.write_text("<annotation />", encoding="utf-8")
        rows.append(
            f"{image_path},{annotation_path},India,test,{label},"
            f"{'pothole' if label else 'non_pothole'}"
        )

    manifest_path = tmp_path / "test.csv"
    manifest_path.write_text(
        "image_path,annotation_path,country,split,label,label_name\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def export_test_predictions(
    tmp_path: Path,
    manifest_path: Path,
    model_name: str,
    experiment_id: str,
) -> Path:
    model = BrightnessModel()
    config = RDDPredictionExportConfig(
        model_name=model_name,
        experiment_id=experiment_id,
        test_manifest_path=manifest_path,
        output_dir=tmp_path / "results",
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    config.model_output_dir.mkdir(parents=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
        },
        config.checkpoint_path,
    )
    return RDDPredictionExporter(config, model=model).export()


def test_prediction_export_writes_every_test_row_with_four_columns(tmp_path):
    manifest_path = write_test_manifest(tmp_path)
    predictions_path = export_test_predictions(
        tmp_path,
        manifest_path,
        model_name="resnet18",
        experiment_id="A",
    )

    with predictions_path.open(newline="", encoding="utf-8") as predictions_file:
        reader = csv.DictReader(predictions_file)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == PREDICTION_COLUMNS
    assert len(rows) == 6
    assert [row["image_id"] for row in rows] == [str(index) for index in range(6)]
    assert all(
        float(row["non_pothole_softmax"]) + float(row["pothole_softmax"])
        == pytest.approx(1.0, abs=1e-7)
        for row in rows
    )
    assert {row["predicted_class"] for row in rows} <= {
        "non_pothole",
        "pothole",
    }
    assert {row["actual_class"] for row in rows} == {
        "non_pothole",
        "pothole",
    }


def test_prediction_preview_uses_both_exported_model_csvs(tmp_path):
    manifest_path = write_test_manifest(tmp_path)
    export_test_predictions(tmp_path, manifest_path, "ds_cnn_small", "A")
    export_test_predictions(tmp_path, manifest_path, "mobilevit_xs", "G")
    config = RDDPredictionPreviewConfig(
        model_experiments={"ds_cnn_small": "A", "mobilevit_xs": "G"},
        sample_count=4,
        test_manifest_path=manifest_path,
        output_dir=tmp_path / "results",
    )

    preview_path = RDDPredictionPreviewer(config).create()

    assert preview_path.is_file()


def test_prediction_preview_rejects_more_samples_than_test_images(tmp_path):
    config = RDDPredictionPreviewConfig(
        model_experiments={"ds_cnn_small": "A"},
        sample_count=7,
        test_manifest_path=write_test_manifest(tmp_path),
        output_dir=tmp_path / "results",
    )

    with pytest.raises(ValueError, match="between 1 and 6"):
        RDDPredictionPreviewer(config).create()
