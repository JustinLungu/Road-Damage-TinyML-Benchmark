from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import src.rdd_training.evaluation as evaluation_module
from src.rdd_training.evaluation import BinaryPotholeEvaluator, RDDEvaluationConfig
from src.rdd_training.utils import (
    collate_binary_pothole_batch,
    compute_binary_roc_auc,
    rank_evaluation_rows,
    write_model_comparison_csv,
)


class FixedClassifier(nn.Module):
    def forward(self, images):
        scores = images.flatten(start_dim=1).mean(dim=1)
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
