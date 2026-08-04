from pathlib import Path
from types import SimpleNamespace

import pytest

import src.rdd_benchmark.experiments.runner as runner_module
from src.rdd_benchmark.experiments.dataset_builder import RDDExperimentManifestPaths
from src.rdd_benchmark.experiments.experiment_registry import get_rdd_experiment_config
from src.rdd_benchmark.experiments.runner import (
    RDDExperimentRunConfig,
    RDDExperimentRunner,
)


def make_manifest_paths(tmp_path: Path) -> RDDExperimentManifestPaths:
    return RDDExperimentManifestPaths(tmp_path / "dataset")


def test_rdd_experiment_runner_rejects_empty_model_names():
    run_config = RDDExperimentRunConfig(
        experiment_config=get_rdd_experiment_config("A"),
        model_names=(),
    )

    with pytest.raises(ValueError, match="At least one model"):
        RDDExperimentRunner(run_config)


def test_rdd_experiment_runner_trains_evaluates_and_compares(
    monkeypatch,
    tmp_path,
):
    manifest_paths = make_manifest_paths(tmp_path)
    captured = {"trainer_configs": [], "evaluation_configs": []}

    def fake_build_experiment_manifest_paths(config, **kwargs):
        captured["builder"] = (config.experiment_id, kwargs)
        return manifest_paths

    class FakeTrainer:
        def __init__(self, config):
            captured["trainer_configs"].append(config)

        def train(self):
            return SimpleNamespace(
                best_checkpoint_path=tmp_path / "best.pt",
                best_metric_value=0.75,
                best_epoch=2,
            )

    class FakeEvaluator:
        def __init__(self, config):
            captured["evaluation_configs"].append(config)

        def evaluate(self):
            return {
                "balanced_accuracy": 0.75,
                "f1": 0.72,
                "recall": 0.70,
            }

    def fake_write_model_comparison_csv(
        comparison_path,
        evaluation_rows,
        ranking_metric,
    ):
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text("model_name\n", encoding="utf-8")
        captured["comparison"] = (
            comparison_path,
            tuple(evaluation_rows),
            ranking_metric,
        )
        return [
            {"rank": 1, "model_name": "tiny", "balanced_accuracy": 0.75, "f1": 0.72}
        ]

    monkeypatch.setattr(
        runner_module,
        "build_experiment_manifest_paths",
        fake_build_experiment_manifest_paths,
    )
    monkeypatch.setattr(runner_module, "BinaryPotholeTrainer", FakeTrainer)
    monkeypatch.setattr(runner_module, "BinaryPotholeEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        runner_module,
        "write_model_comparison_csv",
        fake_write_model_comparison_csv,
    )

    experiment_config = get_rdd_experiment_config("C")
    run_config = RDDExperimentRunConfig(
        experiment_config=experiment_config,
        model_names=("tiny",),
        output_dir=tmp_path / "results",
        run_training=True,
        run_evaluation=True,
        run_comparison=True,
    )
    RDDExperimentRunner(run_config).run()

    trainer_config = captured["trainer_configs"][0]
    evaluation_config = captured["evaluation_configs"][0]

    assert captured["comparison"][0] == (
        tmp_path
        / "results"
        / experiment_config.experiment_name
        / "model_comparison.csv"
    )
    assert trainer_config.train_manifest_path == manifest_paths.train_manifest_path
    assert trainer_config.validation_manifest_path == (
        manifest_paths.validation_manifest_path
    )
    assert trainer_config.sampler_strategy == "weighted_sampler"
    assert trainer_config.target_pothole_fraction == 0.5
    assert trainer_config.augmentation_strategy == "strong"
    assert trainer_config.best_metric == "balanced_accuracy"
    assert evaluation_config.test_manifest_path == manifest_paths.test_manifest_path
    assert captured["comparison"][2] == "f1"


def test_rdd_experiment_runner_skips_existing_checkpoint(
    monkeypatch,
    tmp_path,
):
    manifest_paths = make_manifest_paths(tmp_path)
    trainer_called = False

    def fake_build_experiment_manifest_paths(config, **kwargs):
        return manifest_paths

    class FakeTrainer:
        def __init__(self, config):
            nonlocal trainer_called
            trainer_called = True

    monkeypatch.setattr(
        runner_module,
        "build_experiment_manifest_paths",
        fake_build_experiment_manifest_paths,
    )
    monkeypatch.setattr(runner_module, "BinaryPotholeTrainer", FakeTrainer)

    experiment_config = get_rdd_experiment_config("A")
    checkpoint_path = (
        tmp_path / "results" / experiment_config.experiment_name / "tiny" / "best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    run_config = RDDExperimentRunConfig(
        experiment_config=experiment_config,
        model_names=("tiny",),
        output_dir=tmp_path / "results",
        run_training=True,
        run_evaluation=False,
        run_comparison=False,
    )
    RDDExperimentRunner(run_config).run()

    assert trainer_called is False


def test_rdd_experiment_runner_loads_comparison_rows_when_evaluation_not_run(
    monkeypatch,
    tmp_path,
):
    manifest_paths = make_manifest_paths(tmp_path)
    captured = {}

    monkeypatch.setattr(
        runner_module,
        "build_experiment_manifest_paths",
        lambda config, **kwargs: manifest_paths,
    )
    monkeypatch.setattr(
        runner_module,
        "load_evaluation_rows_from_metrics_csv",
        lambda model_names, output_dir, skip_missing: [
            {"model_name": "tiny", "balanced_accuracy": 0.8, "f1": 0.74, "recall": 0.7}
        ],
    )

    def fake_write_model_comparison_csv(
        comparison_path,
        evaluation_rows,
        ranking_metric,
    ):
        captured["comparison"] = (
            comparison_path,
            tuple(evaluation_rows),
            ranking_metric,
        )
        return [{"rank": 1, "model_name": "tiny", "balanced_accuracy": 0.8, "f1": 0.74}]

    monkeypatch.setattr(
        runner_module,
        "write_model_comparison_csv",
        fake_write_model_comparison_csv,
    )

    experiment_config = get_rdd_experiment_config("A")
    run_config = RDDExperimentRunConfig(
        experiment_config=experiment_config,
        model_names=("tiny",),
        output_dir=tmp_path / "results",
        run_training=False,
        run_evaluation=False,
        run_comparison=True,
    )
    RDDExperimentRunner(run_config).run()

    assert captured["comparison"][0] == (
        tmp_path
        / "results"
        / experiment_config.experiment_name
        / "model_comparison.csv"
    )
    assert captured["comparison"][1][0]["model_name"] == "tiny"
