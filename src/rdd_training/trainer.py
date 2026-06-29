from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.constants import RDD2022_BINARY_POTHOLE_DIR, RDD_TRAINING_RESULTS_DIR
from src.rdd_training.constants import (
    RDD_TRAINING_BATCH_SIZE,
    RDD_TRAINING_BEST_METRIC,
    RDD_TRAINING_EPOCHS,
    RDD_TRAINING_LEARNING_RATE,
    RDD_TRAINING_NUM_WORKERS,
    RDD_TRAINING_PROGRESS_INTERVAL,
    RDD_TRAINING_SAVE_PLOTS,
    RDD_TRAINING_USE_WEIGHTED_LOSS,
    RDD_TRAINING_WEIGHT_DECAY,
)
from src.rdd_training.dataset import BinaryPotholeDataset, BinaryPotholeManifest
from src.rdd_training.utils import (
    calculate_class_weights,
    collate_binary_pothole_batch,
    compute_binary_classification_metrics,
    extract_logits,
    load_and_adapt_model_for_binary_pothole,
    make_image_transform,
    write_training_loss_plot,
)


@dataclass(frozen=True)
class RDDTrainingConfig:
    model_name: str
    train_manifest_path: Path = RDD2022_BINARY_POTHOLE_DIR / "train.csv"
    validation_manifest_path: Path = RDD2022_BINARY_POTHOLE_DIR / "validation.csv"
    output_dir: Path = RDD_TRAINING_RESULTS_DIR
    batch_size: int = RDD_TRAINING_BATCH_SIZE
    num_workers: int = RDD_TRAINING_NUM_WORKERS
    epochs: int = RDD_TRAINING_EPOCHS
    learning_rate: float = RDD_TRAINING_LEARNING_RATE
    weight_decay: float = RDD_TRAINING_WEIGHT_DECAY
    best_metric: str = RDD_TRAINING_BEST_METRIC
    use_weighted_loss: bool = RDD_TRAINING_USE_WEIGHTED_LOSS
    progress_interval: int = RDD_TRAINING_PROGRESS_INTERVAL
    save_plots: bool = RDD_TRAINING_SAVE_PLOTS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class RDDTrainingResult:
    model_name: str
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    best_checkpoint_path: Path
    history_path: Path
    loss_curve_path: Path | None


class BinaryPotholeTrainer:
    """Fine-tune one adapted image classifier on binary RDD pothole labels."""

    def __init__(
        self,
        config: RDDTrainingConfig,
        model: nn.Module | None = None,
    ) -> None:
        if config.epochs < 1:
            raise ValueError("epochs must be at least one.")
        if config.batch_size < 1:
            raise ValueError("batch_size must be at least one.")

        self.config = config
        self.device = torch.device(config.device)
        self.model = model

    def train(self) -> RDDTrainingResult:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = self.config.output_dir / self.config.model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        train_manifest = BinaryPotholeManifest.from_csv(
            self.config.train_manifest_path,
            expected_split="train",
        )
        validation_manifest = BinaryPotholeManifest.from_csv(
            self.config.validation_manifest_path,
            expected_split="validation",
        )
        train_loader = self._make_data_loader(train_manifest, is_train=True)
        validation_loader = self._make_data_loader(validation_manifest, is_train=False)

        print(
            "  setup: "
            f"device={self.device}, epochs={self.config.epochs}, "
            f"batch_size={self.config.batch_size}"
        )
        print(
            "  data: "
            f"train_samples={len(train_manifest)}, "
            f"validation_samples={len(validation_manifest)}, "
            f"train_batches={len(train_loader)}, "
            f"validation_batches={len(validation_loader)}"
        )
        print(f"  loading/adapting model: {self.config.model_name}")
        model = self._load_model().to(self.device)
        criterion = self._make_criterion(train_manifest)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_metric_value = float("-inf")
        best_epoch = 0
        history = []
        best_checkpoint_path = model_output_dir / "best.pt"

        for epoch in range(1, self.config.epochs + 1):
            print()
            print(f"  epoch {epoch}/{self.config.epochs}: training")
            train_loss = self._run_training_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                epoch,
            )
            print(f"  epoch {epoch}/{self.config.epochs}: validating")
            validation_metrics = self.evaluate(model, validation_loader, criterion)
            selected_metric = validation_metrics[self.config.best_metric]

            epoch_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
            history.append(epoch_row)

            print(
                "  epoch "
                f"{epoch}/{self.config.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={validation_metrics['loss']:.4f}, "
                f"val_f1={validation_metrics['f1']:.4f}, "
                f"val_balanced_accuracy="
                f"{validation_metrics['balanced_accuracy']:.4f}, "
                f"val_recall={validation_metrics['recall']:.4f}"
            )

            if selected_metric > best_metric_value:
                best_metric_value = selected_metric
                best_epoch = epoch
                self._save_checkpoint(
                    model=model,
                    checkpoint_path=best_checkpoint_path,
                    epoch=epoch,
                    metric_value=selected_metric,
                    validation_metrics=validation_metrics,
                )
                print(
                    "  saved new best checkpoint: "
                    f"{best_checkpoint_path} "
                    f"({self.config.best_metric}={selected_metric:.4f})"
                )

        history_path = model_output_dir / "history.csv"
        self._write_history(history_path, history)
        loss_curve_path = model_output_dir / "loss_curve.png"
        if self.config.save_plots:
            write_training_loss_plot(loss_curve_path, history)
            print(f"  loss_curve: {loss_curve_path}")
        else:
            loss_curve_path = None

        return RDDTrainingResult(
            model_name=self.config.model_name,
            best_epoch=best_epoch,
            best_metric_name=self.config.best_metric,
            best_metric_value=best_metric_value,
            best_checkpoint_path=best_checkpoint_path,
            history_path=history_path,
            loss_curve_path=loss_curve_path,
        )

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        sample_count = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = extract_logits(model(images))
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                sample_count += batch_size
                predictions.append(logits.argmax(dim=1))
                targets.append(labels)

        metrics = compute_binary_classification_metrics(
            predictions=torch.cat(predictions),
            targets=torch.cat(targets),
        )
        metrics["loss"] = total_loss / sample_count
        return metrics

    def _load_model(self) -> nn.Module:
        if self.model is not None:
            return self.model
        return load_and_adapt_model_for_binary_pothole(self.config.model_name)

    def _make_data_loader(
        self,
        manifest: BinaryPotholeManifest,
        is_train: bool,
    ) -> DataLoader:
        dataset = BinaryPotholeDataset(
            manifest=manifest,
            transform=make_image_transform(self.config.model_name, is_train=is_train),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_train,
            num_workers=self.config.num_workers,
            collate_fn=collate_binary_pothole_batch,
        )

    def _make_criterion(self, train_manifest: BinaryPotholeManifest) -> nn.Module:
        if not self.config.use_weighted_loss:
            return nn.CrossEntropyLoss()

        class_weights = calculate_class_weights(train_manifest.class_counts()).to(
            self.device
        )
        return nn.CrossEntropyLoss(weight=class_weights)

    def _run_training_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        sample_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = extract_logits(model(images))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size

            if self._should_print_batch_progress(batch_index, len(train_loader)):
                average_loss = total_loss / sample_count
                print(
                    "    batch "
                    f"{batch_index}/{len(train_loader)} "
                    f"epoch={epoch}/{self.config.epochs} "
                    f"samples={sample_count}/{len(train_loader.dataset)} "
                    f"loss={average_loss:.4f}"
                )

        return total_loss / sample_count

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

    def _save_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        epoch: int,
        metric_value: float,
        validation_metrics: dict[str, float],
    ) -> None:
        torch.save(
            {
                "model_name": self.config.model_name,
                "epoch": epoch,
                "best_metric": self.config.best_metric,
                "best_metric_value": metric_value,
                "validation_metrics": validation_metrics,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )

    def _write_history(
        self,
        history_path: Path,
        history: list[dict[str, Any]],
    ) -> None:
        if not history:
            return

        with history_path.open("w", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
