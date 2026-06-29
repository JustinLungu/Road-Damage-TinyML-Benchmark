from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import src.rdd_training.trainer as trainer_module
from src.rdd_training.trainer import BinaryPotholeTrainer, RDDTrainingConfig
from src.rdd_training.utils import (
    calculate_class_weights,
    collate_binary_pothole_batch,
    compute_binary_classification_metrics,
    extract_logits,
    select_rdd_model_names,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 2),
        )

    def forward(self, images):
        return self.classifier(images)


class FakeManifest:
    @classmethod
    def from_csv(cls, manifest_path: Path, expected_split: str):
        return cls(expected_split)

    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self):
        return 4

    def class_counts(self):
        return {0: 3, 1: 1}


def make_loader() -> DataLoader:
    samples = [
        make_sample(label=0),
        make_sample(label=1),
        make_sample(label=0),
        make_sample(label=0),
    ]
    return DataLoader(
        samples,
        batch_size=2,
        collate_fn=collate_binary_pothole_batch,
    )


def make_sample(label: int) -> dict:
    return {
        "image": torch.ones(3, 4, 4) * label,
        "label": label,
        "image_path": Path(f"image_{label}.jpg"),
        "label_name": "pothole" if label == 1 else "non_pothole",
        "country": "India",
        "split": "train",
    }


def test_training_utils_select_models_weights_metrics_and_logits() -> None:
    assert select_rdd_model_names(
        mode="single",
        single_model="resnet18",
    ) == ("resnet18",)
    assert select_rdd_model_names(
        mode="all",
        model_names=("b", "a"),
    ) == ("b", "a")
    with pytest.raises(ValueError, match="single' or 'all"):
        select_rdd_model_names(mode="bad")

    weights = calculate_class_weights({0: 3, 1: 1})
    assert weights.tolist() == pytest.approx([2 / 3, 2.0])

    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([0, 1, 0, 1])
    metrics = compute_binary_classification_metrics(predictions, targets)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)

    output = type("Output", (), {"logits": torch.ones(2, 2)})()
    assert torch.equal(extract_logits(output), output.logits)


def test_binary_pothole_trainer_runs_and_saves_best_checkpoint(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr(trainer_module, "BinaryPotholeManifest", FakeManifest)

    config = RDDTrainingConfig(
        model_name="tiny",
        train_manifest_path=tmp_path / "train.csv",
        validation_manifest_path=tmp_path / "validation.csv",
        output_dir=tmp_path / "results",
        batch_size=2,
        num_workers=0,
        epochs=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        best_metric="f1",
        progress_interval=1,
        device="cpu",
    )
    trainer = BinaryPotholeTrainer(config=config, model=TinyClassifier())

    def fake_make_data_loader(manifest, is_train: bool):
        assert manifest.split == ("train" if is_train else "validation")
        return make_loader()

    monkeypatch.setattr(trainer, "_make_data_loader", fake_make_data_loader)

    result = trainer.train()
    output = capsys.readouterr().out

    assert result.model_name == "tiny"
    assert result.best_epoch in {1, 2}
    assert result.best_metric_name == "f1"
    assert result.best_checkpoint_path.is_file()
    assert result.history_path.is_file()
    assert result.loss_curve_path is not None
    assert result.loss_curve_path.is_file()
    assert result.f1_curve_path is not None
    assert result.f1_curve_path.is_file()
    assert result.accuracy_curve_path is not None
    assert result.accuracy_curve_path.is_file()
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    assert checkpoint["model_name"] == "tiny"
    assert "model_state_dict" in checkpoint
    assert "setup: device=cpu, epochs=2, batch_size=2" in output
    assert "epoch 1/2: training" in output
    assert "batch 1/2 epoch=1/2" in output
    assert "epoch 1/2: validating" in output
    assert "saved new best checkpoint" in output
    assert "loss_curve:" in output
    assert "f1_curve:" in output
    assert "accuracy_curve:" in output
