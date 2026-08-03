from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import src.rdd_benchmark.training.trainer as trainer_module
from src.rdd_benchmark.data_loader.sampling import calculate_class_weights
from src.rdd_benchmark.data_loader.utils import collate_binary_pothole_batch
from src.rdd_benchmark.training.trainer import BinaryPotholeTrainer, RDDTrainingConfig
from src.rdd_benchmark.training.utils import (
    compute_binary_classification_metrics,
    extract_logits,
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
    def __init__(self, split: str) -> None:
        self.split = split
        self.samples = [
            type("Sample", (), {"label": label})() for label in (0, 1, 0, 0)
        ]

    def __len__(self):
        return 4

    def class_counts(self):
        return {0: 3, 1: 1}


class FakeDataset:
    def __init__(self, split: str) -> None:
        self.split = split
        self.manifest = FakeManifest(split)

    def __len__(self):
        return 4


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


def test_training_utils_compute_weights_metrics_and_logits() -> None:
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
    config = RDDTrainingConfig(
        model_name="tiny",
        experiment_name="full_image_baseline",
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

    def fake_make_dataset(split: str, is_train: bool):
        assert split == ("train" if is_train else "validation")
        return FakeDataset(split)

    def fake_make_data_loader(dataset, is_train: bool):
        assert dataset.split == ("train" if is_train else "validation")
        return make_loader()

    monkeypatch.setattr(trainer, "_make_dataset", fake_make_dataset)
    monkeypatch.setattr(trainer, "_make_data_loader", fake_make_data_loader)

    result = trainer.train()
    output = capsys.readouterr().out

    assert result.model_name == "tiny"
    assert result.experiment_name == "full_image_baseline"
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
    assert result.best_checkpoint_path.parent == (
        tmp_path / "results" / "full_image_baseline" / "tiny"
    )
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    assert checkpoint["model_name"] == "tiny"
    assert "model_state_dict" in checkpoint
    assert (
        "setup: device=cpu, epochs=2, batch_size=2, "
        "sampler_strategy=none, "
        "augmentation_strategy=standard"
    ) in output
    assert "epoch 1/2: training" in output
    assert "batch 1/2 epoch=1/2" in output
    assert "epoch 1/2: validating" in output
    assert "saved new best checkpoint" in output
    assert "loss_curve:" in output
    assert "f1_curve:" in output
    assert "accuracy_curve:" in output


def test_binary_pothole_trainer_forwards_augmentation_strategy(
    monkeypatch,
) -> None:
    captured = {}
    trainer = BinaryPotholeTrainer(
        config=RDDTrainingConfig(
            model_name="tiny",
            augmentation_strategy="strong",
            device="cpu",
        ),
        model=TinyClassifier(),
    )

    def fake_make_image_transform(model_name, is_train, augmentation_strategy):
        captured["transform"] = (model_name, is_train, augmentation_strategy)
        return "transform"

    def fake_make_rdd_dataset(split, transform):
        captured["dataset"] = (split, transform)
        return FakeDataset(split)

    monkeypatch.setattr(
        trainer_module,
        "make_image_transform",
        fake_make_image_transform,
    )
    monkeypatch.setattr(trainer_module, "make_rdd_dataset", fake_make_rdd_dataset)

    dataset = trainer._make_dataset(split="train", is_train=True)

    assert isinstance(dataset, FakeDataset)
    assert captured["transform"] == ("tiny", True, "strong")


def test_binary_pothole_trainer_uses_weighted_sampler_for_training() -> None:
    trainer = BinaryPotholeTrainer(
        config=RDDTrainingConfig(
            model_name="tiny",
            batch_size=2,
            num_workers=0,
            sampler_strategy="weighted_sampler",
            target_pothole_fraction=0.5,
            device="cpu",
        ),
        model=TinyClassifier(),
    )

    data_loader = trainer._make_data_loader(FakeDataset("train"), is_train=True)

    assert isinstance(data_loader.sampler, WeightedRandomSampler)


def test_binary_pothole_trainer_disables_sampler_for_validation() -> None:
    trainer = BinaryPotholeTrainer(
        config=RDDTrainingConfig(
            model_name="tiny",
            batch_size=2,
            num_workers=0,
            sampler_strategy="weighted_sampler",
            target_pothole_fraction=0.5,
            device="cpu",
        ),
        model=TinyClassifier(),
    )

    data_loader = trainer._make_data_loader(FakeDataset("validation"), is_train=False)

    assert not isinstance(data_loader.sampler, WeightedRandomSampler)


def test_binary_pothole_trainer_rejects_invalid_early_stopping_patience() -> None:
    config = RDDTrainingConfig(
        model_name="tiny",
        early_stopping_patience=-1,
    )

    with pytest.raises(ValueError, match="patience cannot be negative"):
        BinaryPotholeTrainer(config=config, model=TinyClassifier())


def test_binary_pothole_trainer_rejects_invalid_sampler_strategy() -> None:
    config = RDDTrainingConfig(
        model_name="tiny",
        sampler_strategy="balanced_magic",
    )

    with pytest.raises(ValueError, match="sampler_strategy must be one of"):
        BinaryPotholeTrainer(config=config, model=TinyClassifier())
