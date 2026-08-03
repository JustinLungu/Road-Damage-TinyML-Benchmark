import pytest

from src.rdd_benchmark.experiments.experiment_registry import (
    RDD_EXPERIMENT_REGISTRY,
    get_rdd_experiment_config,
    select_rdd_experiment_configs,
)


def test_experiment_names_are_unique():
    names = [config.experiment_name for config in RDD_EXPERIMENT_REGISTRY.values()]
    assert len(names) == len(set(names))


def test_weighted_sampler_experiments_define_target_fraction():
    assert get_rdd_experiment_config("B").target_pothole_fraction == 0.5
    assert get_rdd_experiment_config("F").target_pothole_fraction == 0.25


def test_downsampling_experiments_define_majority_ratio():
    assert get_rdd_experiment_config("D").non_potholes_per_pothole == 3
    assert get_rdd_experiment_config("G").non_potholes_per_pothole == 5
    assert get_rdd_experiment_config("H").non_potholes_per_pothole == 5


def test_select_experiments_preserves_requested_order():
    configs = select_rdd_experiment_configs(("H", "A"))
    assert [config.experiment_id for config in configs] == ["H", "A"]


def test_select_experiments_rejects_empty_selection():
    with pytest.raises(ValueError, match="At least one"):
        select_rdd_experiment_configs(())


def test_get_experiment_rejects_unknown_id():
    with pytest.raises(ValueError, match="Unknown RDD experiment id"):
        get_rdd_experiment_config("Z")
