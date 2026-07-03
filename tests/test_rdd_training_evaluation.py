from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import src.rdd_training.evaluation as evaluation_module
from src.rdd_training.evaluation import (
    BinaryPotholeEvaluator,
    GridImageInferenceRunner,
    GridPatchGenerator,
    RDDEvaluationConfig,
)
from src.rdd_training.utils import (
    collate_binary_pothole_batch,
    compute_binary_roc_auc,
    load_evaluation_rows_from_metrics_csv,
    rank_evaluation_rows,
    write_evaluation_metrics_csv,
    write_model_comparison_csv,
)


class FixedClassifier(nn.Module):
    def forward(self, images):
        scores = images.flatten(start_dim=1).mean(dim=1)
        return torch.stack([-scores, scores], dim=1)


class PositiveIfBrightClassifier(nn.Module):
    def forward(self, images):
        scores = images.flatten(start_dim=1).mean(dim=1) * 6.0
        return torch.stack([-scores, scores], dim=1)


class FakeManifest:
    @classmethod
    def from_csv(cls, manifest_path: Path, expected_split: str):
        return cls(expected_split)

    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self):
        return 4


def make_loader() -> DataLoader:
    return DataLoader(
        [
            make_sample(label=0),
            make_sample(label=1),
            make_sample(label=0),
            make_sample(label=1),
        ],
        batch_size=2,
        collate_fn=collate_binary_pothole_batch,
    )


def make_sample(label: int) -> dict:
    return {
        "image": torch.ones(3, 4, 4) * label,
        "label": label,
        "image_path": Path(f"image_{label}.jpg"),
        "label_name": "pothole" if label == 1 else "non_pothole",
        "country": "Japan",
        "split": "test",
    }


def test_binary_roc_auc_handles_ties_and_missing_classes() -> None:
    auc = compute_binary_roc_auc(
        positive_scores=torch.tensor([0.1, 0.4, 0.4, 0.9]),
        targets=torch.tensor([0, 1, 0, 1]),
    )
    assert auc == pytest.approx(0.875)

    missing_class_auc = compute_binary_roc_auc(
        positive_scores=torch.tensor([0.1, 0.2]),
        targets=torch.tensor([0, 0]),
    )
    assert torch.isnan(torch.tensor(missing_class_auc))


def test_grid_patch_generator_splits_full_image_into_fixed_grid() -> None:
    patch_boxes = GridPatchGenerator(grid_size=3).generate(
        image_width=9,
        image_height=6,
    )

    assert patch_boxes == (
        (0, 0, 3, 2),
        (3, 0, 6, 2),
        (6, 0, 9, 2),
        (0, 2, 3, 4),
        (3, 2, 6, 4),
        (6, 2, 9, 4),
        (0, 4, 3, 6),
        (3, 4, 6, 6),
        (6, 4, 9, 6),
    )


def test_grid_patch_generator_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        GridPatchGenerator(grid_size=0)

    with pytest.raises(NotImplementedError, match="non-overlapping"):
        GridPatchGenerator(grid_size=3, overlap=0.25)

    with pytest.raises(ValueError, match="positive"):
        GridPatchGenerator(grid_size=3).generate(image_width=0, image_height=6)


def test_grid_image_inference_uses_max_patch_score_for_image_prediction(
    tmp_path,
) -> None:
    image_path = tmp_path / "grid.jpg"
    image = Image.new("RGB", (9, 6), color=(0, 0, 0))
    for x_position in range(3, 6):
        for y_position in range(2, 4):
            image.putpixel((x_position, y_position), (255, 255, 255))
    image.save(image_path)

    def transform(patch: Image.Image) -> torch.Tensor:
        value = 1.0 if patch.getextrema()[0][1] > 0 else 0.0
        return torch.full((3, 4, 4), value)

    runner = GridImageInferenceRunner(
        model=PositiveIfBrightClassifier(),
        transform=transform,
        device="cpu",
        patch_generator=GridPatchGenerator(grid_size=3),
        decision_threshold=0.75,
    )

    prediction = runner.predict_image(image_path)

    assert len(prediction.patch_boxes) == 9
    assert len(prediction.patch_scores) == 9
    assert prediction.patch_boxes[4] == (3, 2, 6, 4)
    assert prediction.patch_scores[4] == pytest.approx(0.9999938, rel=1e-5)
    assert prediction.image_score == pytest.approx(max(prediction.patch_scores))
    assert prediction.prediction == 1


def test_binary_pothole_evaluator_writes_metrics_and_confusion_matrix(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr(evaluation_module, "BinaryPotholeManifest", FakeManifest)

    config = RDDEvaluationConfig(
        model_name="tiny",
        test_manifest_path=tmp_path / "test.csv",
        output_dir=tmp_path / "results",
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    config.model_output_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": FixedClassifier().state_dict()},
        config.checkpoint_path,
    )

    evaluator = BinaryPotholeEvaluator(config=config, model=FixedClassifier())
    monkeypatch.setattr(evaluator, "_make_data_loader", lambda manifest: make_loader())

    result = evaluator.evaluate()
    output = capsys.readouterr().out

    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert result.metrics["precision"] == pytest.approx(1.0)
    assert result.metrics["recall"] == pytest.approx(1.0)
    assert result.metrics["f1"] == pytest.approx(1.0)
    assert result.metrics["roc_auc"] == pytest.approx(1.0)
    assert result.metrics_json_path.is_file()
    assert result.metrics_csv_path.is_file()
    assert result.confusion_matrix_path.is_file()
    assert result.confusion_matrix_plot_path is not None
    assert result.confusion_matrix_plot_path.is_file()
    assert result.roc_curve_plot_path is not None
    assert result.roc_curve_plot_path.is_file()
    assert result.metric_bar_plot_path is not None
    assert result.metric_bar_plot_path.is_file()
    assert "test metrics: accuracy=1.0000" in output


def test_model_comparison_ranking_and_top_models_csv(tmp_path) -> None:
    rows = [
        {"model_name": "a", "f1": 0.3, "recall": 0.5},
        {"model_name": "b", "f1": 0.9, "recall": 0.7},
    ]

    ranked = rank_evaluation_rows(rows, ranking_metric="f1")
    assert [row["model_name"] for row in ranked] == ["b", "a"]
    assert [row["rank"] for row in ranked] == [1, 2]

    comparison_path = tmp_path / "model_comparison.csv"
    written = write_model_comparison_csv(
        comparison_path,
        evaluation_rows=rows,
        ranking_metric="f1",
        top_k=1,
    )

    assert written[0]["model_name"] == "b"
    assert comparison_path.is_file()
    assert (tmp_path / "top_models.csv").is_file()


def test_load_evaluation_rows_from_existing_metrics_csv(tmp_path) -> None:
    model_dir = tmp_path / "mobilenet_v3_small"
    metrics_path = model_dir / "test_metrics.csv"
    write_evaluation_metrics_csv(
        metrics_path,
        model_name="mobilenet_v3_small",
        split="test",
        metrics={
            "accuracy": 0.8,
            "balanced_accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.55,
            "roc_auc": 0.75,
        },
    )

    rows = load_evaluation_rows_from_metrics_csv(
        model_names=("mobilenet_v3_small",),
        output_dir=tmp_path,
    )

    assert rows == [
        {
            "model_name": "mobilenet_v3_small",
            "checkpoint_path": str(model_dir / "best.pt"),
            "accuracy": 0.8,
            "balanced_accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.55,
            "roc_auc": 0.75,
        }
    ]
