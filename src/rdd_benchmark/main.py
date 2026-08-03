from __future__ import annotations

from src.rdd_benchmark.constants import (
    RDD_ACTIVE_EXPERIMENT_IDS,
    RDD_MODEL_NAMES,
    RUN_RDD_PREPROCESSING,
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
    selected_model_names = RDD_MODEL_NAMES
    selected_experiment_configs = select_rdd_experiment_configs(
        RDD_ACTIVE_EXPERIMENT_IDS
    )
    print_experiment_plan(selected_experiment_configs, selected_model_names)

    if RUN_RDD_PREPROCESSING:
        print("RDD2022 binary pothole full-image preprocessing")
        prepare_binary_pothole_manifests()
        print()

    for experiment_config in selected_experiment_configs:
        runner = RDDExperimentRunner(
            RDDExperimentRunConfig(
                experiment_config=experiment_config,
                model_names=selected_model_names,
            )
        )
        runner.run()
