from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from src.constants import RDD_TRAINING_RESULTS_DIR
from src.rdd_benchmark.constants import (
    RDD_COMPARISON_RANKING_METRIC,
    RDD_COMPARISON_TOP_K,
    RDD_SKIP_FAILED_MODELS,
    RDD_TRAINING_SKIP_EXISTING_CHECKPOINTS,
    RUN_RDD_COMPARISON,
    RUN_RDD_EVALUATION,
    RUN_RDD_TRAINING,
)
from src.rdd_benchmark.experiments.dataset_builder import (
    RDDExperimentManifestPaths,
    build_experiment_manifest_paths,
)
from src.rdd_benchmark.experiments.experiment_registry import RDDExperimentConfig
from src.rdd_benchmark.training.evaluation import (
    BinaryPotholeEvaluator,
    RDDEvaluationConfig,
)
from src.rdd_benchmark.training.trainer import BinaryPotholeTrainer, RDDTrainingConfig
from src.rdd_benchmark.training.utils import (
    load_evaluation_rows_from_metrics_csv,
    write_failure_log,
    write_model_comparison_csv,
)


@dataclass(frozen=True)
class RDDExperimentRunConfig:
    experiment_config: RDDExperimentConfig
    model_names: tuple[str, ...]
    best_previous_experiment_name: str | None = None
    output_dir: Path = RDD_TRAINING_RESULTS_DIR
    run_training: bool = RUN_RDD_TRAINING
    run_evaluation: bool = RUN_RDD_EVALUATION
    run_comparison: bool = RUN_RDD_COMPARISON
    skip_existing_checkpoints: bool = RDD_TRAINING_SKIP_EXISTING_CHECKPOINTS
    skip_failed_models: bool = RDD_SKIP_FAILED_MODELS
    comparison_ranking_metric: str = RDD_COMPARISON_RANKING_METRIC
    comparison_top_k: int = RDD_COMPARISON_TOP_K

    @property
    def experiment_output_dir(self) -> Path:
        return self.output_dir / self.experiment_config.experiment_name


@dataclass(frozen=True)
class RDDExperimentRunResult:
    experiment_name: str
    manifest_paths: RDDExperimentManifestPaths
    trained_model_names: tuple[str, ...]
    evaluation_rows: tuple[dict[str, float | str], ...]
    comparison_path: Path | None


class RDDExperimentRunner:
    def __init__(self, config: RDDExperimentRunConfig) -> None:
        if not config.model_names:
            raise ValueError("At least one model name is required.")
        self.config = config

    def run(self) -> RDDExperimentRunResult:
        experiment_config = self.config.experiment_config
        manifest_paths = build_experiment_manifest_paths(
            experiment_config,
            best_previous_experiment_name=self.config.best_previous_experiment_name,
        )

        print()
        print(f"RDD2022 experiment: {experiment_config.experiment_name}")
        print(f"  dataset_dir: {manifest_paths.dataset_dir}")
        print(f"  train_manifest: {manifest_paths.train_manifest_path}")
        print(f"  validation_manifest: {manifest_paths.validation_manifest_path}")
        print(f"  test_manifest: {manifest_paths.test_manifest_path}")

        trained_model_names = self._run_training(manifest_paths)
        evaluation_rows = self._run_evaluation(manifest_paths, trained_model_names)
        comparison_path = self._run_comparison(trained_model_names, evaluation_rows)

        return RDDExperimentRunResult(
            experiment_name=experiment_config.experiment_name,
            manifest_paths=manifest_paths,
            trained_model_names=tuple(trained_model_names),
            evaluation_rows=tuple(evaluation_rows),
            comparison_path=comparison_path,
        )

    def _run_training(
        self,
        manifest_paths: RDDExperimentManifestPaths,
    ) -> list[str]:
        if not self.config.run_training:
            return []

        trained_model_names = []
        print()
        print(f"RDD2022 binary pothole training: {self.config.experiment_config.experiment_name}")

        for model_name in self.config.model_names:
            print()
            print(f"Training {model_name}")
            checkpoint_path = self.config.experiment_output_dir / model_name / "best.pt"
            if self.config.skip_existing_checkpoints and checkpoint_path.is_file():
                trained_model_names.append(model_name)
                print(f"  skipping existing checkpoint: {checkpoint_path}")
                continue

            try:
                trainer = BinaryPotholeTrainer(
                    RDDTrainingConfig(
                        model_name=model_name,
                        experiment_name=self.config.experiment_config.experiment_name,
                        training_input_mode=(
                            self.config.experiment_config.training_input_mode
                        ),
                        train_manifest_path=manifest_paths.train_manifest_path,
                        validation_manifest_path=(
                            manifest_paths.validation_manifest_path
                        ),
                        sampler_strategy=self.config.experiment_config.sampler_strategy,
                        target_pothole_fraction=(
                            self.config.experiment_config.target_pothole_fraction
                        ),
                        augmentation_strategy=(
                            self.config.experiment_config.augmentation_strategy
                        ),
                        output_dir=self.config.output_dir,
                        best_metric=self.config.experiment_config.target_metric,
                    )
                )
                result = trainer.train()
                trained_model_names.append(result.model_name)
                print(
                    "  best_checkpoint: "
                    f"{result.best_checkpoint_path} "
                    f"({result.best_metric_name}={result.best_metric_value:.4f}, "
                    f"epoch={result.best_epoch})"
                )
            except Exception:
                self._handle_model_failure(model_name, "training")
                if not self.config.skip_failed_models:
                    raise
                continue

        return trained_model_names

    def _run_evaluation(
        self,
        manifest_paths: RDDExperimentManifestPaths,
        trained_model_names: list[str],
    ) -> list[dict[str, float | str]]:
        if not self.config.run_evaluation:
            return []

        evaluation_rows = []
        evaluation_model_names = tuple(trained_model_names) or self.config.model_names

        print()
        print(f"RDD2022 binary pothole evaluation: {self.config.experiment_config.experiment_name}")
        for model_name in evaluation_model_names:
            print()
            print(f"Evaluating {model_name}")
            try:
                evaluator = BinaryPotholeEvaluator(
                    RDDEvaluationConfig(
                        model_name=model_name,
                        experiment_name=self.config.experiment_config.experiment_name,
                        training_input_mode=(
                            self.config.experiment_config.training_input_mode
                        ),
                        evaluation_input_mode=(
                            self.config.experiment_config.evaluation_input_mode
                        ),
                        validation_manifest_path=(
                            manifest_paths.validation_manifest_path
                        ),
                        test_manifest_path=manifest_paths.test_manifest_path,
                        output_dir=self.config.output_dir,
                    )
                )
                result = evaluator.evaluate()
                evaluation_rows.append(result.comparison_row())
                print(f"  metrics_json: {result.metrics_json_path}")
                print(f"  confusion_matrix: {result.confusion_matrix_path}")
            except Exception:
                self._handle_model_failure(model_name, "evaluation")
                if not self.config.skip_failed_models:
                    raise
                continue

        return evaluation_rows

    def _run_comparison(
        self,
        trained_model_names: list[str],
        evaluation_rows: list[dict[str, float | str]],
    ) -> Path | None:
        if not self.config.run_comparison:
            return None

        print()
        print(f"RDD2022 binary pothole model comparison: {self.config.experiment_config.experiment_name}")
        comparison_model_names = tuple(trained_model_names) or self.config.model_names
        if not evaluation_rows:
            evaluation_rows = load_evaluation_rows_from_metrics_csv(
                comparison_model_names,
                output_dir=self.config.experiment_output_dir,
                skip_missing=self.config.skip_failed_models,
            )
        if not evaluation_rows:
            print("  no evaluation rows available; skipping comparison.")
            return None

        comparison_path = self.config.experiment_output_dir / "model_comparison.csv"
        ranked_rows = write_model_comparison_csv(
            comparison_path,
            evaluation_rows=evaluation_rows,
            ranking_metric=self.config.comparison_ranking_metric,
            top_k=self.config.comparison_top_k,
        )
        print(
            "  model_comparison: "
            f"{comparison_path} "
            f"(ranked by {self.config.comparison_ranking_metric})"
        )
        print(f"  top {self.config.comparison_top_k}:")
        for row in ranked_rows[: self.config.comparison_top_k]:
            print(
                "    "
                f"#{row['rank']} {row['model_name']} "
                f"{self.config.comparison_ranking_metric}="
                f"{row[self.config.comparison_ranking_metric]:.4f}"
            )
        return comparison_path

    def _handle_model_failure(self, model_name: str, phase: str) -> None:
        error_log_path = (
            self.config.experiment_output_dir
            / model_name
            / f"{phase}_error.log"
        )
        write_failure_log(error_log_path, traceback.format_exc())
        print(f"  {phase} failed for {model_name}: {error_log_path}")
