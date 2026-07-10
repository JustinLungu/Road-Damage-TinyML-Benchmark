import pytest

from src.rdd_benchmark.experiments.constants import RDD_EXPERIMENT_IDS
from src.rdd_benchmark.experiments.experiment_registry import (
    RDD_EXPERIMENT_REGISTRY,
    get_rdd_experiment_config,
    select_rdd_experiment_configs,
)


def test_all_planned_rdd_experiments_are_registered():
    assert tuple(RDD_EXPERIMENT_REGISTRY) == RDD_EXPERIMENT_IDS


def test_rdd_experiment_names_are_unique():
    experiment_names = [
        config.experiment_name for config in RDD_EXPERIMENT_REGISTRY.values()
    ]

    assert len(experiment_names) == len(set(experiment_names))


def test_shared_rdd_experiment_defaults_are_consistent():
    for config in RDD_EXPERIMENT_REGISTRY.values():
        assert config.training_input_mode == "full_image"
        assert config.evaluation_input_mode == "full_image"
        assert config.use_bbox_aware_crops is False
        assert config.target_metric == "balanced_accuracy"


def test_get_rdd_experiment_config_returns_metadata_dict():
    config = get_rdd_experiment_config("A")

    assert config.experiment_name == "A_full_image_natural_standard_aug"
    assert config.to_dict()["experiment_id"] == "A"


def test_rdd_experiment_b_sets_explicit_balanced_batch_target():
    config = get_rdd_experiment_config("B")

    assert config.sampler_strategy == "weighted_sampler"
    assert config.target_pothole_fraction == 0.5


def test_rdd_experiment_d_uses_minority_strong_augmentation():
    config = get_rdd_experiment_config("D")

    assert config.balancing_strategy == "majority_downsample_moderate"
    assert config.sampler_strategy == "weighted_sampler"
    assert config.augmentation_strategy == "minority_strong"
    assert config.non_potholes_per_pothole == 3


def test_rdd_experiment_e_sets_explicit_synthetic_ratio():
    config = get_rdd_experiment_config("E")

    assert config.synthetic_strategy == "small_pothole_addition"
    assert config.synthetic_pothole_ratio == 0.2


def test_select_rdd_experiment_configs_preserves_requested_order():
    configs = select_rdd_experiment_configs(("C", "A"))

    assert [config.experiment_id for config in configs] == ["C", "A"]


def test_select_rdd_experiment_configs_rejects_empty_selection():
    with pytest.raises(ValueError, match="At least one"):
        select_rdd_experiment_configs(())


def test_get_rdd_experiment_config_rejects_unknown_id():
    with pytest.raises(ValueError, match="Unknown RDD experiment id"):
        get_rdd_experiment_config("Z")
