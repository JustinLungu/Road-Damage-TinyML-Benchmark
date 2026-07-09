from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.rdd_benchmark.constants import RDD_SPLIT_MODE
from src.rdd_benchmark.experiments.constants import (
    RDD_EXPERIMENT_A,
    RDD_EXPERIMENT_B,
    RDD_EXPERIMENT_C,
    RDD_EXPERIMENT_D,
    RDD_EXPERIMENT_E,
)


@dataclass(frozen=True)
class RDDExperimentConfig:
    experiment_id: str
    experiment_name: str
    description: str
    dataset_strategy: str
    reason: str
    split_mode: str
    training_input_mode: str
    evaluation_input_mode: str
    balancing_strategy: str
    sampler_strategy: str
    augmentation_strategy: str
    synthetic_strategy: str
    target_pothole_fraction: float | None
    non_potholes_per_pothole: int | None
    fallback_non_potholes_per_pothole: int | None
    synthetic_pothole_ratio: float
    use_bbox_aware_crops: bool
    target_metric: str
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RDD_EXPERIMENT_REGISTRY = {
    RDD_EXPERIMENT_A: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_A,
        experiment_name="A_full_image_natural_standard_aug",
        description="Natural data with standard augmentation.",
        dataset_strategy="Natural data + standard augmentation",
        reason="Baseline",
        split_mode=RDD_SPLIT_MODE,
        training_input_mode="full_image",
        evaluation_input_mode="full_image",
        balancing_strategy="none",
        sampler_strategy="none",
        augmentation_strategy="standard",
        synthetic_strategy="none",
        target_pothole_fraction=None,
        non_potholes_per_pothole=None,
        fallback_non_potholes_per_pothole=None,
        synthetic_pothole_ratio=0.0,
        use_bbox_aware_crops=False,
        target_metric="accuracy",
    ),
    RDD_EXPERIMENT_B: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_B,
        experiment_name="B_full_image_weighted_sampler_standard_aug",
        description="Weighted sampler or balanced batches with standard augmentation.",
        dataset_strategy="Weighted sampler / balanced batches + standard augmentation",
        reason="Tests upsampling effect",
        split_mode=RDD_SPLIT_MODE,
        training_input_mode="full_image",
        evaluation_input_mode="full_image",
        balancing_strategy="none",
        sampler_strategy="weighted_sampler",
        augmentation_strategy="standard",
        synthetic_strategy="none",
        target_pothole_fraction=0.5,
        non_potholes_per_pothole=None,
        fallback_non_potholes_per_pothole=None,
        synthetic_pothole_ratio=0.0,
        use_bbox_aware_crops=False,
        target_metric="accuracy",
    ),
    RDD_EXPERIMENT_C: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_C,
        experiment_name="C_full_image_weighted_sampler_minority_aug",
        description="Weighted sampler with stronger pothole/minority augmentation.",
        dataset_strategy="Weighted sampler + stronger minority augmentation",
        reason="Most likely to help",
        split_mode=RDD_SPLIT_MODE,
        training_input_mode="full_image",
        evaluation_input_mode="full_image",
        balancing_strategy="none",
        sampler_strategy="weighted_sampler",
        augmentation_strategy="minority_strong",
        synthetic_strategy="none",
        target_pothole_fraction=0.5,
        non_potholes_per_pothole=None,
        fallback_non_potholes_per_pothole=None,
        synthetic_pothole_ratio=0.0,
        use_bbox_aware_crops=False,
        target_metric="accuracy",
    ),
    RDD_EXPERIMENT_D: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_D,
        experiment_name="D_full_image_downsample_majority_minority_aug",
        description="Moderate majority downsampling plus weighted sampler and augmentation.",
        dataset_strategy=(
            "Moderate majority downsampling + weighted sampler + "
            "minority augmentation"
        ),
        reason="Tests whether reducing majority dominance helps",
        split_mode=RDD_SPLIT_MODE,
        training_input_mode="full_image",
        evaluation_input_mode="full_image",
        balancing_strategy="majority_downsample_moderate",
        sampler_strategy="weighted_sampler",
        augmentation_strategy="minority_strong",
        synthetic_strategy="none",
        target_pothole_fraction=0.5,
        non_potholes_per_pothole=3,
        fallback_non_potholes_per_pothole=4,
        synthetic_pothole_ratio=0.0,
        use_bbox_aware_crops=False,
        target_metric="accuracy",
    ),
    RDD_EXPERIMENT_E: RDDExperimentConfig(
        experiment_id=RDD_EXPERIMENT_E,
        experiment_name="E_full_image_best_plus_synthetic_potholes",
        description="Best of C/D plus a small synthetic pothole addition.",
        dataset_strategy="Best of C/D + small synthetic pothole addition",
        reason="Tests AI generation safely",
        split_mode=RDD_SPLIT_MODE,
        training_input_mode="full_image",
        evaluation_input_mode="full_image",
        balancing_strategy="best_previous",
        sampler_strategy="best_previous",
        augmentation_strategy="best_previous",
        synthetic_strategy="small_pothole_addition",
        target_pothole_fraction=0.5,
        non_potholes_per_pothole=None,
        fallback_non_potholes_per_pothole=None,
        synthetic_pothole_ratio=0.2,
        use_bbox_aware_crops=False,
        target_metric="accuracy",
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
