from __future__ import annotations

import traceback

from src.constants import RDD2022_BINARY_POTHOLE_DIR, RDD_TRAINING_RESULTS_DIR
from src.rdd_benchmark.constants import (
    RDD_ACTIVE_EXPERIMENT_IDS,
    RDD_COMPARISON_RANKING_METRIC,
    RDD_COMPARISON_TOP_K,
    RDD_EXPERIMENT_NAME,
    RDD_MODEL_MODE,
    RDD_SKIP_FAILED_MODELS,
    RDD_TRAINING_SKIP_EXISTING_CHECKPOINTS,
    RUN_RDD_FULL_IMAGE_PREPROCESSING,
    RUN_RDD_PATCH_PREPROCESSING,
    RUN_RDD_COMPARISON,
    RUN_RDD_EVALUATION,
    RUN_RDD_TRAINING,
)
from src.rdd_benchmark.data_loader.dataset import BinaryPotholeDataset, BinaryPotholeManifest
from src.rdd_benchmark.training.evaluation import BinaryPotholeEvaluator, RDDEvaluationConfig
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole import prepare_binary_pothole_manifests
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole_patches import (
    prepare_binary_pothole_patch_manifests,
)
from src.rdd_benchmark.experiments import select_rdd_experiment_configs
from src.rdd_benchmark.training.trainer import BinaryPotholeTrainer, RDDTrainingConfig
from src.rdd_benchmark.training.utils import (
    load_evaluation_rows_from_metrics_csv,
    load_and_adapt_all_binary_pothole_models,
    load_and_adapt_model_for_binary_pothole,
    select_rdd_model_names,
    write_failure_log,
    write_model_comparison_csv,
)


SPLIT_MANIFESTS = {
    "train": RDD2022_BINARY_POTHOLE_DIR / "train.csv",
    "validation": RDD2022_BINARY_POTHOLE_DIR / "validation.csv",
    "test": RDD2022_BINARY_POTHOLE_DIR / "test.csv",
}


if __name__ == "__main__":
    selected_model_names = select_rdd_model_names()
    selected_experiment_configs = select_rdd_experiment_configs(
        RDD_ACTIVE_EXPERIMENT_IDS
    )
    experiment_output_dir = RDD_TRAINING_RESULTS_DIR / RDD_EXPERIMENT_NAME

    print("RDD2022 experiment plan")
    for experiment_config in selected_experiment_configs:
        print(
            "  "
            f"{experiment_config.experiment_id}: "
            f"{experiment_config.experiment_name}"
        )
        print(f"    dataset_strategy: {experiment_config.dataset_strategy}")
        print(f"    reason: {experiment_config.reason}")
        print(f"    sampler_strategy: {experiment_config.sampler_strategy}")
        print(f"    augmentation_strategy: {experiment_config.augmentation_strategy}")
        print(f"    target_pothole_fraction: {experiment_config.target_pothole_fraction}")
        print(
            "    non_potholes_per_pothole: "
            f"{experiment_config.non_potholes_per_pothole}"
        )
        print(f"    synthetic_pothole_ratio: {experiment_config.synthetic_pothole_ratio}")
    print()

    if RUN_RDD_FULL_IMAGE_PREPROCESSING:
        print("RDD2022 binary pothole full-image preprocessing")
        prepare_binary_pothole_manifests()
        print()

    if RUN_RDD_PATCH_PREPROCESSING:
        print("RDD2022 binary pothole patch preprocessing")
        prepare_binary_pothole_patch_manifests()
        print()

    print("RDD2022 binary pothole dataset loader")
    print(f"Manifest directory: {RDD2022_BINARY_POTHOLE_DIR}")

    for split, manifest_path in SPLIT_MANIFESTS.items():
        manifest = BinaryPotholeManifest.from_csv(
            manifest_path,
            expected_split=split,
        )
        dataset = BinaryPotholeDataset(manifest)
        first_item = dataset[0]
        class_counts = manifest.class_counts()

        print()
        print(f"{split}:")
        print(f"  samples: {len(dataset)}")
        print(f"  non_pothole: {class_counts[0]}")
        print(f"  pothole: {class_counts[1]}")
        print(f"  pothole_fraction: {manifest.positive_fraction():.3f}")
        print(f"  countries: {dict(manifest.country_counts())}")
        print(
            "  first_sample: "
            f"{first_item['image'].mode} {first_item['image'].size}, "
            f"label={first_item['label']} ({first_item['label_name']}), "
            f"country={first_item['country']}"
        )

    print()
    print("Model adaptation smoke test")
    if RDD_MODEL_MODE == "all":
        adapted_models = load_and_adapt_all_binary_pothole_models()
        for model_name, adapted_model in adapted_models.items():
            print(f"  adapted_model: {model_name} -> {type(adapted_model).__name__}")
    elif RDD_MODEL_MODE == "single":
        model_name = select_rdd_model_names()[0]
        adapted_model = load_and_adapt_model_for_binary_pothole(model_name)
        print(
            "  adapted_model: "
            f"{model_name} -> {type(adapted_model).__name__}"
        )
    else:
        raise ValueError("RDD_MODEL_MODE must be 'single' or 'all'.")

    trained_model_names = []
    evaluation_rows = []
    if RUN_RDD_TRAINING:
        print()
        print(f"RDD2022 binary pothole training: {RDD_EXPERIMENT_NAME}")
        for model_name in selected_model_names:
            print()
            print(f"Training {model_name}")
            checkpoint_path = experiment_output_dir / model_name / "best.pt"
            if RDD_TRAINING_SKIP_EXISTING_CHECKPOINTS and checkpoint_path.is_file():
                trained_model_names.append(model_name)
                print(f"  skipping existing checkpoint: {checkpoint_path}")
                continue

            try:
                trainer = BinaryPotholeTrainer(RDDTrainingConfig(model_name=model_name))
                result = trainer.train()
                trained_model_names.append(result.model_name)
                print(
                    "  best_checkpoint: "
                    f"{result.best_checkpoint_path} "
                    f"({result.best_metric_name}={result.best_metric_value:.4f}, "
                    f"epoch={result.best_epoch})"
                )
                if result.loss_curve_path is not None:
                    print(f"  loss_curve: {result.loss_curve_path}")
                if result.f1_curve_path is not None:
                    print(f"  f1_curve: {result.f1_curve_path}")
                if result.accuracy_curve_path is not None:
                    print(f"  accuracy_curve: {result.accuracy_curve_path}")
            except Exception:
                error_log_path = (
                    experiment_output_dir / model_name / "training_error.log"
                )
                write_failure_log(error_log_path, traceback.format_exc())
                print(f"  training failed for {model_name}: {error_log_path}")
                if not RDD_SKIP_FAILED_MODELS:
                    raise
                continue

    if RUN_RDD_EVALUATION:
        print()
        print("RDD2022 binary pothole evaluation")
        evaluation_model_names = tuple(trained_model_names) or selected_model_names

        for model_name in evaluation_model_names:
            print()
            print(f"Evaluating {model_name}")
            try:
                evaluator = BinaryPotholeEvaluator(
                    RDDEvaluationConfig(model_name=model_name)
                )
                result = evaluator.evaluate()
                evaluation_rows.append(result.comparison_row())
                print(f"  metrics_json: {result.metrics_json_path}")
                print(f"  confusion_matrix: {result.confusion_matrix_path}")
                if result.confusion_matrix_plot_path is not None:
                    print(f"  confusion_matrix_plot: {result.confusion_matrix_plot_path}")
                if result.roc_curve_plot_path is not None:
                    print(f"  roc_curve_plot: {result.roc_curve_plot_path}")
                if result.metric_bar_plot_path is not None:
                    print(f"  metric_bar_plot: {result.metric_bar_plot_path}")
            except Exception:
                error_log_path = (
                    experiment_output_dir / model_name / "evaluation_error.log"
                )
                write_failure_log(error_log_path, traceback.format_exc())
                print(f"  evaluation failed for {model_name}: {error_log_path}")
                if not RDD_SKIP_FAILED_MODELS:
                    raise
                continue

    if RUN_RDD_COMPARISON:
        print()
        print("RDD2022 binary pothole model comparison")
        comparison_model_names = tuple(trained_model_names) or selected_model_names
        if not evaluation_rows:
            evaluation_rows = load_evaluation_rows_from_metrics_csv(
                comparison_model_names,
                output_dir=experiment_output_dir,
                skip_missing=RDD_SKIP_FAILED_MODELS,
            )
        if not evaluation_rows:
            print("  no evaluation rows available; skipping comparison.")
            raise SystemExit(0)

        comparison_path = experiment_output_dir / "model_comparison.csv"
        ranked_rows = write_model_comparison_csv(
            comparison_path,
            evaluation_rows=evaluation_rows,
            ranking_metric=RDD_COMPARISON_RANKING_METRIC,
            top_k=RDD_COMPARISON_TOP_K,
        )
        print()
        print(
            "Model comparison: "
            f"{comparison_path} "
            f"(ranked by {RDD_COMPARISON_RANKING_METRIC})"
        )
        print(f"Top {RDD_COMPARISON_TOP_K}:")
        for row in ranked_rows[:RDD_COMPARISON_TOP_K]:
            print(
                "  "
                f"#{row['rank']} {row['model_name']} "
                f"{RDD_COMPARISON_RANKING_METRIC}="
                f"{row[RDD_COMPARISON_RANKING_METRIC]:.4f}, "
                f"recall={row['recall']:.4f}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}"
            )
