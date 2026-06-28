from __future__ import annotations

from src.constants import RDD2022_BINARY_POTHOLE_DIR, RDD_TRAINING_RESULTS_DIR
from src.rdd_training.constants import (
    RDD_EVALUATION_RANKING_METRIC,
    RDD_EVALUATION_TOP_K,
    RDD_MODEL_MODE,
    RUN_RDD_EVALUATION,
    RUN_RDD_TRAINING,
)
from src.rdd_training.dataset import BinaryPotholeDataset, BinaryPotholeManifest
from src.rdd_training.evaluation import BinaryPotholeEvaluator, RDDEvaluationConfig
from src.rdd_training.trainer import BinaryPotholeTrainer, RDDTrainingConfig
from src.rdd_training.utils import (
    load_and_adapt_all_binary_pothole_models,
    load_and_adapt_model_for_binary_pothole,
    select_rdd_model_names,
    write_model_comparison_csv,
)


SPLIT_MANIFESTS = {
    "train": RDD2022_BINARY_POTHOLE_DIR / "train.csv",
    "validation": RDD2022_BINARY_POTHOLE_DIR / "validation.csv",
    "test": RDD2022_BINARY_POTHOLE_DIR / "test.csv",
}


if __name__ == "__main__":
    selected_model_names = select_rdd_model_names()

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
    if RUN_RDD_TRAINING:
        print()
        print("RDD2022 binary pothole training")
        for model_name in selected_model_names:
            print()
            print(f"Training {model_name}")
            trainer = BinaryPotholeTrainer(RDDTrainingConfig(model_name=model_name))
            result = trainer.train()
            trained_model_names.append(result.model_name)
            print(
                "  best_checkpoint: "
                f"{result.best_checkpoint_path} "
                f"({result.best_metric_name}={result.best_metric_value:.4f}, "
                f"epoch={result.best_epoch})"
            )

    if RUN_RDD_EVALUATION:
        print()
        print("RDD2022 binary pothole evaluation")
        evaluation_model_names = tuple(trained_model_names) or selected_model_names
        evaluation_rows = []

        for model_name in evaluation_model_names:
            print()
            print(f"Evaluating {model_name}")
            evaluator = BinaryPotholeEvaluator(RDDEvaluationConfig(model_name=model_name))
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

        comparison_path = RDD_TRAINING_RESULTS_DIR / "model_comparison.csv"
        ranked_rows = write_model_comparison_csv(
            comparison_path,
            evaluation_rows=evaluation_rows,
            ranking_metric=RDD_EVALUATION_RANKING_METRIC,
            top_k=RDD_EVALUATION_TOP_K,
        )
        print()
        print(
            "Model comparison: "
            f"{comparison_path} "
            f"(ranked by {RDD_EVALUATION_RANKING_METRIC})"
        )
        print(f"Top {RDD_EVALUATION_TOP_K}:")
        for row in ranked_rows[:RDD_EVALUATION_TOP_K]:
            print(
                "  "
                f"#{row['rank']} {row['model_name']} "
                f"{RDD_EVALUATION_RANKING_METRIC}="
                f"{row[RDD_EVALUATION_RANKING_METRIC]:.4f}, "
                f"recall={row['recall']:.4f}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}"
            )
