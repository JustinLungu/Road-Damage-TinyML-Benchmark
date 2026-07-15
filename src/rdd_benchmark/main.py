from __future__ import annotations

from src.rdd_benchmark.constants import (
    RDD_ACTIVE_EXPERIMENT_IDS,
    RDD_BEST_PREVIOUS_EXPERIMENT_NAME,
    RUN_RDD_FULL_IMAGE_PREPROCESSING,
    RUN_RDD_PATCH_PREPROCESSING,
)
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole import (
    prepare_binary_pothole_manifests,
)
from src.rdd_benchmark.data_preprocessing.prepare_binary_pothole_patches import (
    prepare_binary_pothole_patch_manifests,
)
from src.rdd_benchmark.experiments import (
    RDDExperimentRunConfig,
    RDDExperimentRunner,
    select_rdd_experiment_configs,
)
from src.rdd_benchmark.training.utils import select_rdd_model_names


def print_experiment_plan(selected_experiment_configs, selected_model_names) -> None:
    print("RDD2022 experiment plan")
    print(f"  models: {', '.join(selected_model_names)}")
    for experiment_config in selected_experiment_configs:
        print()
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


if __name__ == "__main__":
    selected_model_names = select_rdd_model_names()
    selected_experiment_configs = select_rdd_experiment_configs(
        RDD_ACTIVE_EXPERIMENT_IDS
    )
    print_experiment_plan(selected_experiment_configs, selected_model_names)

    if RUN_RDD_FULL_IMAGE_PREPROCESSING:
        print("RDD2022 binary pothole full-image preprocessing")
        prepare_binary_pothole_manifests()
        print()

    if RUN_RDD_PATCH_PREPROCESSING:
        print("RDD2022 binary pothole patch preprocessing")
        prepare_binary_pothole_patch_manifests()
        print()

    for experiment_config in selected_experiment_configs:
        runner = RDDExperimentRunner(
            RDDExperimentRunConfig(
                experiment_config=experiment_config,
                model_names=selected_model_names,
                best_previous_experiment_name=RDD_BEST_PREVIOUS_EXPERIMENT_NAME,
            )
        )
        runner.run()
