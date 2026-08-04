from __future__ import annotations

import argparse

from src.rdd_benchmark.constants import (
    RDD_ACTIVE_EXPERIMENT_IDS,
    RDD_MODEL_NAMES,
    RDD_PREDICTION_MODELS,
    RDD_PREDICTION_PREVIEW_DEFAULT_SAMPLES,
    RUN_RDD_COMPARISON,
    RUN_RDD_EVALUATION,
    RUN_RDD_PREPROCESSING,
    RUN_RDD_TRAINING,
)
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole import (
    prepare_binary_pothole_manifests,
)
from src.rdd_benchmark.experiments.experiment_registry import (
    select_rdd_experiment_configs,
)
from src.rdd_benchmark.experiments.runner import (
    RDDExperimentRunConfig,
    RDDExperimentRunner,
)
from src.rdd_benchmark.training.prediction_export import (
    RDDPredictionExportConfig,
    RDDPredictionExporter,
)
from src.rdd_benchmark.training.prediction_preview import (
    RDDPredictionPreviewConfig,
    RDDPredictionPreviewer,
)


def print_experiment_plan(selected_experiment_configs, selected_model_names) -> None:
    print("RDD2022 experiment plan")
    print(f"  models: {', '.join(selected_model_names)}")
    for experiment_config in selected_experiment_configs:
        print()
        print(
            f"  {experiment_config.experiment_id}: {experiment_config.experiment_name}"
        )
        print(f"    sampler_strategy: {experiment_config.sampler_strategy}")
        print(f"    augmentation_strategy: {experiment_config.augmentation_strategy}")
        print(
            f"    target_pothole_fraction: {experiment_config.target_pothole_fraction}"
        )
        print(
            "    non_potholes_per_pothole: "
            f"{experiment_config.non_potholes_per_pothole}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RDD2022 benchmark.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "export-predictions",
        help="Evaluate configured checkpoints on every test image and export CSVs.",
    )
    preview_parser = subparsers.add_parser(
        "preview-predictions",
        help="Create a random visual comparison from exported prediction CSVs.",
    )
    preview_parser.add_argument(
        "--num-samples",
        type=int,
        default=RDD_PREDICTION_PREVIEW_DEFAULT_SAMPLES,
        help=(
            "Number of random test images to preview "
            f"(default: {RDD_PREDICTION_PREVIEW_DEFAULT_SAMPLES})."
        ),
    )
    arguments = parser.parse_args()

    if arguments.command == "export-predictions":
        print("RDD2022 full-test softmax prediction export")
        for model_name, experiment_id in RDD_PREDICTION_MODELS.items():
            RDDPredictionExporter(
                RDDPredictionExportConfig(
                    model_name=model_name,
                    experiment_id=experiment_id,
                )
            ).export()
        raise SystemExit(0)

    if arguments.command == "preview-predictions":
        RDDPredictionPreviewer(
            RDDPredictionPreviewConfig(
                model_experiments=RDD_PREDICTION_MODELS,
                sample_count=arguments.num_samples,
            )
        ).create()
        raise SystemExit(0)

    if RUN_RDD_PREPROCESSING:
        print("RDD2022 binary pothole full-image preprocessing")
        prepare_binary_pothole_manifests()
        print()

    if RUN_RDD_TRAINING or RUN_RDD_EVALUATION or RUN_RDD_COMPARISON:
        selected_experiment_configs = select_rdd_experiment_configs(
            RDD_ACTIVE_EXPERIMENT_IDS
        )
        print_experiment_plan(selected_experiment_configs, RDD_MODEL_NAMES)
        for experiment_config in selected_experiment_configs:
            runner = RDDExperimentRunner(
                RDDExperimentRunConfig(
                    experiment_config=experiment_config,
                    model_names=RDD_MODEL_NAMES,
                )
            )
            runner.run()
