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
    GridImagePrediction,
    GridPatchGenerator,
    GridThresholdTuner,
    RDDEvaluationConfig,
    load_grid_decision_threshold,
)
from src.rdd_training.dataset import BinaryPotholeManifest, BinaryPotholeSample
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


class FakeGridRunner:
    def __init__(self, scores_by_path: dict[Path, float]) -> None:
        self.scores_by_path = scores_by_path

    def predict_image(self, image_path: Path) -> GridImagePrediction:
        score = self.scores_by_path[image_path]
        return GridImagePrediction(
            image_path=image_path,
            patch_boxes=((0, 0, 1, 1),),
            patch_scores=(score,),
            image_score=score,
            prediction=int(score >= 0.5),
        )


class FakeGridInferenceRunner:
    scores_by_path: dict[Path, float] = {}

    def __init__(self, *args, decision_threshold: float = 0.5, **kwargs) -> None:
        self.decision_threshold = decision_threshold

    def predict_image(self, image_path: Path) -> GridImagePrediction:
        score = self.scores_by_path[image_path]
        return GridImagePrediction(
            image_path=image_path,
            patch_boxes=((0, 0, 1, 1),),
            patch_scores=(score,),
            image_score=score,
            prediction=int(score >= self.decision_threshold),
        )


class IterableManifest:
    def __init__(
        self,
        split: str,
        labels_by_path: dict[Path, int],
    ) -> None:
        self.split = split
        self.samples = [
            BinaryPotholeSample(
                image_path=image_path,
                annotation_path=Path("annotation.xml"),
                label=label,
                label_name="pothole" if label == 1 else "non_pothole",
                country="Japan",
                split=split,
            )
            for image_path, label in labels_by_path.items()
        ]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


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


def test_grid_threshold_tuner_selects_best_validation_f1_and_writes_json(
    tmp_path,
) -> None:
    image_paths = [tmp_path / f"image_{index}.jpg" for index in range(4)]
    annotation_path = tmp_path / "annotation.xml"
    threshold_path = tmp_path / "threshold.json"
    manifest = BinaryPotholeManifest(
        samples=[
            BinaryPotholeSample(
                image_path=image_paths[0],
                annotation_path=annotation_path,
                label=0,
                label_name="non_pothole",
                country="United_States",
                split="validation",
            ),
            BinaryPotholeSample(
                image_path=image_paths[1],
                annotation_path=annotation_path,
                label=1,
                label_name="pothole",
                country="United_States",
                split="validation",
            ),
            BinaryPotholeSample(
                image_path=image_paths[2],
                annotation_path=annotation_path,
                label=0,
                label_name="non_pothole",
                country="United_States",
                split="validation",
            ),
            BinaryPotholeSample(
                image_path=image_paths[3],
                annotation_path=annotation_path,
                label=1,
                label_name="pothole",
                country="United_States",
                split="validation",
            ),
        ],
        manifest_path=tmp_path / "validation.csv",
    )
    tuner = GridThresholdTuner(
        inference_runner=FakeGridRunner(
            {
                image_paths[0]: 0.2,
                image_paths[1]: 0.4,
                image_paths[2]: 0.6,
                image_paths[3]: 0.8,
            }
        ),
        threshold_values=(0.3, 0.5, 0.7),
        metric_name="f1",
    )

    result = tuner.tune(manifest, threshold_path=threshold_path)

    assert result.threshold == pytest.approx(0.3)
    assert result.metric_name == "f1"
    assert result.metric_value == pytest.approx(0.8)
    assert result.threshold_path == threshold_path
    assert len(result.threshold_metrics) == 3
    assert threshold_path.is_file()
    assert load_grid_decision_threshold(threshold_path) == pytest.approx(0.3)
    assert load_grid_decision_threshold(
        tmp_path / "missing_threshold.json",
        default_threshold=0.55,
    ) == pytest.approx(0.55)


def test_grid_threshold_tuner_rejects_empty_threshold_values() -> None:
    with pytest.raises(ValueError, match="threshold_values"):
        GridThresholdTuner(
            inference_runner=FakeGridRunner({}),
            threshold_values=(),
        )


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
    assert result.metrics["experiment_name"] == "full_image_baseline"
    assert result.metrics["training_input_mode"] == "full_image"
    assert result.metrics["evaluation_input_mode"] == "full_image"
    assert result.metrics["grid_size"] == pytest.approx(0.0)
    assert result.metrics["threshold"] == pytest.approx(0.5)
    assert result.metrics["num_images"] == pytest.approx(4.0)
    assert result.metrics["num_patches_per_image"] == pytest.approx(1.0)
    assert result.metrics["avg_image_inference_ms"] >= 0.0
    assert result.metrics["images_per_second"] >= 0.0
    assert result.metrics_json_path.is_file()
    assert result.metrics_csv_path.is_file()
    assert result.confusion_matrix_path.is_file()
    assert result.inference_metrics_path is not None
    assert result.inference_metrics_path.is_file()
    assert result.confusion_matrix_plot_path is not None
    assert result.confusion_matrix_plot_path.is_file()
    assert result.roc_curve_plot_path is not None
    assert result.roc_curve_plot_path.is_file()
    assert result.metric_bar_plot_path is not None
    assert result.metric_bar_plot_path.is_file()
    assert "test metrics: accuracy=1.0000" in output


def test_binary_pothole_evaluator_grid_image_mode_tunes_threshold_and_saves_timing(
    monkeypatch,
    tmp_path,
) -> None:
    validation_paths = [tmp_path / f"validation_{index}.jpg" for index in range(4)]
    test_paths = [tmp_path / f"test_{index}.jpg" for index in range(4)]
    validation_manifest = IterableManifest(
        "validation",
        {
            validation_paths[0]: 0,
            validation_paths[1]: 1,
            validation_paths[2]: 0,
            validation_paths[3]: 1,
        },
    )
    test_manifest = IterableManifest(
        "test",
        {
            test_paths[0]: 0,
            test_paths[1]: 1,
            test_paths[2]: 0,
            test_paths[3]: 1,
        },
    )
    FakeGridInferenceRunner.scores_by_path = {
        validation_paths[0]: 0.2,
        validation_paths[1]: 0.4,
        validation_paths[2]: 0.6,
        validation_paths[3]: 0.8,
        test_paths[0]: 0.2,
        test_paths[1]: 0.4,
        test_paths[2]: 0.6,
        test_paths[3]: 0.8,
    }

    def fake_from_csv(manifest_path: Path, expected_split: str):
        return validation_manifest if expected_split == "validation" else test_manifest

    monkeypatch.setattr(
        evaluation_module.BinaryPotholeManifest,
        "from_csv",
        staticmethod(fake_from_csv),
    )
    monkeypatch.setattr(
        evaluation_module,
        "GridImageInferenceRunner",
        FakeGridInferenceRunner,
    )
    monkeypatch.setattr(
        evaluation_module,
        "make_image_transform",
        lambda model_name, is_train: lambda image: torch.ones(3, 4, 4),
    )

    config = RDDEvaluationConfig(
        model_name="tiny",
        evaluation_input_mode="grid_image",
        output_dir=tmp_path / "results",
        device="cpu",
        save_plots=False,
        threshold_values=(0.3, 0.5, 0.7),
    )
    config.model_output_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": FixedClassifier().state_dict()},
        config.checkpoint_path,
    )

    evaluator = BinaryPotholeEvaluator(config=config, model=FixedClassifier())
    result = evaluator.evaluate()

    assert result.metrics["threshold"] == pytest.approx(0.3)
    assert result.metrics["evaluation_input_mode"] == "grid_image"
    assert result.metrics["grid_size"] == pytest.approx(3.0)
    assert result.metrics["num_images"] == pytest.approx(4.0)
    assert result.metrics["num_patches_per_image"] == pytest.approx(9.0)
    assert result.metrics["f1"] == pytest.approx(0.8)
    assert config.threshold_path.is_file()
    assert result.inference_metrics_path == config.inference_metrics_path
    assert result.inference_metrics_path.is_file()
    assert result.confusion_matrix_plot_path is None
    assert result.roc_curve_plot_path is None
    assert result.metric_bar_plot_path is None


def test_binary_pothole_evaluator_rejects_invalid_evaluation_input_mode() -> None:
    with pytest.raises(ValueError, match="evaluation_input_mode must be one of"):
        BinaryPotholeEvaluator(
            RDDEvaluationConfig(
                model_name="tiny",
                evaluation_input_mode="bad_mode",
            ),
            model=FixedClassifier(),
        )


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


def test_load_evaluation_rows_preserves_mode_metadata(tmp_path) -> None:
    model_dir = tmp_path / "mobilevit_xxs"
    metrics_path = model_dir / "test_metrics.csv"
    write_evaluation_metrics_csv(
        metrics_path,
        model_name="mobilevit_xxs",
        split="test",
        metrics={
            "experiment_name": "annotation_patch_grid3",
            "training_input_mode": "annotation_patch",
            "evaluation_input_mode": "grid_image",
            "grid_size": 3.0,
            "threshold": 0.35,
            "avg_image_inference_ms": 12.5,
            "images_per_second": 80.0,
            "f1": 0.6,
        },
    )

    rows = load_evaluation_rows_from_metrics_csv(
        model_names=("mobilevit_xxs",),
        output_dir=tmp_path,
    )

    assert rows == [
        {
            "model_name": "mobilevit_xxs",
            "checkpoint_path": str(model_dir / "best.pt"),
            "experiment_name": "annotation_patch_grid3",
            "training_input_mode": "annotation_patch",
            "evaluation_input_mode": "grid_image",
            "grid_size": 3.0,
            "threshold": 0.35,
            "avg_image_inference_ms": 12.5,
            "images_per_second": 80.0,
            "f1": 0.6,
        }
    ]
