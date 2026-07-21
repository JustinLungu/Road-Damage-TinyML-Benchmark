from __future__ import annotations

from dataclasses import dataclass

from src.rdd_benchmark.experiments.constants import (
    RDD_EXPERIMENT_A,
    RDD_EXPERIMENT_A30,
    RDD_EXPERIMENT_B,
    RDD_EXPERIMENT_C,
    RDD_EXPERIMENT_D,
    RDD_EXPERIMENT_E,
    RDD_EXPERIMENT_F,
    RDD_EXPERIMENT_G,
    RDD_EXPERIMENT_H,
)


@dataclass(frozen=True)
class RDDExperimentConfig:
    experiment_id: str
    experiment_name: str
    dataset_strategy: str
    reason: str
    sampler_strategy: str
    target_pothole_fraction: float | None
    augmentation_strategy: str
    synthetic_pothole_ratio: float
    non_potholes_per_pothole: int | None
    training_input_mode: str = "full_image"
    evaluation_input_mode: str = "full_image"
    target_metric: str = "balanced_accuracy"
    random_seed: int = 42


RDD_EXPERIMENT_REGISTRY = {
    RDD_EXPERIMENT_A: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_A,
        experiment_name="A_clean400_full_image_natural_standard_aug",
        dataset_strategy="Natural data + standard augmentation",
        reason="Baseline",
        sampler_strategy="none",
        target_pothole_fraction=None,
        augmentation_strategy="standard",
        synthetic_pothole_ratio=0.0,
        non_potholes_per_pothole=None,
    ),
    RDD_EXPERIMENT_A30: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_A30,
        experiment_name="A30_clean400_full_image_natural_standard_aug_e30",
        dataset_strategy="Natural data + standard augmentation",
        reason="Longer from-scratch tiny/small model training run",
        sampler_strategy="none",
        target_pothole_fraction=None,
        augmentation_strategy="standard",
        synthetic_pothole_ratio=0.0,
        non_potholes_per_pothole=None,
    ),
    RDD_EXPERIMENT_B: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_B,
        experiment_name="B_clean400_full_image_weighted_sampler_standard_aug",
        dataset_strategy="Weighted sampler / balanced batches + standard augmentation",
        reason="Tests upsampling effect",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
        augmentation_strategy="standard",
        synthetic_pothole_ratio=0.0,
        non_potholes_per_pothole=None,
    ),
    RDD_EXPERIMENT_C: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_C,
        experiment_name="C_clean400_full_image_weighted_sampler_minority_aug",
        dataset_strategy="Weighted sampler + stronger minority augmentation",
        reason="Most likely to help",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
        augmentation_strategy="minority_strong",
        synthetic_pothole_ratio=0.0,
        non_potholes_per_pothole=None,
    ),
    RDD_EXPERIMENT_D: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_D,
        experiment_name="D_clean400_full_image_downsample_majority_minority_aug",
        dataset_strategy=(
            "Moderate majority downsampling + weighted sampler + "
            "minority augmentation"
        ),
        reason="Tests whether reducing majority dominance helps",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
        augmentation_strategy="minority_strong",
        synthetic_pothole_ratio=0.0,
        non_potholes_per_pothole=3,
    ),
    RDD_EXPERIMENT_E: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_E,
        experiment_name="E_clean400_full_image_best_plus_synthetic_potholes",
        dataset_strategy="Best of C/D + small synthetic pothole addition",
        reason="Tests AI generation safely",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
        augmentation_strategy="minority_strong",
        synthetic_pothole_ratio=0.2,
        non_potholes_per_pothole=None,
    ),
    RDD_EXPERIMENT_F: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_F,
        experiment_name="F_clean400_full_image_weighted_sampler_025_standard_aug",
        dataset_strategy="Gentler weighted sampler + standard augmentation",
        reason="Tests whether 25% pothole sampling improves precision vs B",
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.25,
        augmentation_strategy="standard",
        synthetic_pothole_ratio=0.0,
        non_potholes_per_pothole=None,
    ),
    RDD_EXPERIMENT_G: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_G,
        experiment_name="G_clean400_full_image_downsample_1to5_standard_aug",
        dataset_strategy="Majority downsampling only + standard augmentation",
        reason="Tests downsampling without weighted-sampler overcorrection",
        sampler_strategy="none",
        target_pothole_fraction=None,
        augmentation_strategy="standard",
        synthetic_pothole_ratio=0.0,
        non_potholes_per_pothole=5,
    ),
    RDD_EXPERIMENT_H: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_H,
        experiment_name="H_clean400_full_image_downsample_1to5_minority_aug",
        dataset_strategy="Majority downsampling only + minority augmentation",
        reason="Tests gentler downsampling with stronger pothole augmentation",
        sampler_strategy="none",
        target_pothole_fraction=None,
        augmentation_strategy="minority_strong",
        synthetic_pothole_ratio=0.0,
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
    return tuple(get_rdd_experiment_config(experiment_id) for experiment_id in experiment_ids)
