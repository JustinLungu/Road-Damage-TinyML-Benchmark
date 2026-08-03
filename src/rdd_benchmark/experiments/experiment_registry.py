from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RDDExperimentConfig:
    experiment_id: str
    experiment_name: str
    sampler_strategy: str
    target_pothole_fraction: float | None
    augmentation_strategy: str
    non_potholes_per_pothole: int | None


RDD_EXPERIMENT_REGISTRY = {
    "A": RDDExperimentConfig(
        experiment_id="A",
        experiment_name="A_clean400_full_image_natural_standard_aug",
        sampler_strategy="none",
        target_pothole_fraction=None,
        augmentation_strategy="standard",
        non_potholes_per_pothole=None,
    ),
    "B": RDDExperimentConfig(
        experiment_id="B",
        experiment_name="B_clean400_full_image_weighted_sampler_standard_aug",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
        augmentation_strategy="standard",
        non_potholes_per_pothole=None,
    ),
    "C": RDDExperimentConfig(
        experiment_id="C",
        experiment_name="C_clean400_full_image_weighted_sampler_strong_aug",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
        augmentation_strategy="strong",
        non_potholes_per_pothole=None,
    ),
    "D": RDDExperimentConfig(
        experiment_id="D",
        experiment_name="D_clean400_full_image_downsample_majority_strong_aug",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
        augmentation_strategy="strong",
        non_potholes_per_pothole=3,
    ),
    "F": RDDExperimentConfig(
        experiment_id="F",
        experiment_name="F_clean400_full_image_weighted_sampler_025_standard_aug",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.25,
        augmentation_strategy="standard",
        non_potholes_per_pothole=None,
    ),
    "G": RDDExperimentConfig(
        experiment_id="G",
        experiment_name="G_clean400_full_image_downsample_1to5_standard_aug",
        sampler_strategy="none",
        target_pothole_fraction=None,
        augmentation_strategy="standard",
        non_potholes_per_pothole=5,
    ),
    "H": RDDExperimentConfig(
        experiment_id="H",
        experiment_name="H_clean400_full_image_downsample_1to5_strong_aug",
        sampler_strategy="none",
        target_pothole_fraction=None,
        augmentation_strategy="strong",
        non_potholes_per_pothole=5,
    ),
}


def get_rdd_experiment_config(experiment_id: str) -> RDDExperimentConfig:
    try:
        return RDD_EXPERIMENT_REGISTRY[experiment_id]
    except KeyError as exc:
        valid_ids = ", ".join(sorted(RDD_EXPERIMENT_REGISTRY))
        raise ValueError(
            f"Unknown RDD experiment id: {experiment_id}. Valid ids: {valid_ids}."
        ) from exc


def select_rdd_experiment_configs(
    experiment_ids: tuple[str, ...],
) -> tuple[RDDExperimentConfig, ...]:
    if not experiment_ids:
        raise ValueError("At least one RDD experiment id must be selected.")
    return tuple(
        get_rdd_experiment_config(experiment_id) for experiment_id in experiment_ids
    )
