from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.rdd_benchmark.data_loader.sampling import (
    calculate_class_weights,
    calculate_target_fraction_sample_weights,
    get_dataset_labels,
    make_rdd_sampling_config,
    make_weighted_sampler,
)


@dataclass(frozen=True)
class Sample:
    label: int


class Manifest:
    def __init__(self, labels):
        self.samples = [Sample(label=label) for label in labels]


class LabelDataset(Dataset):
    def __init__(self, labels):
        self.manifest = Manifest(labels)

    def __len__(self):
        return len(self.manifest.samples)

    def __getitem__(self, index):
        return self.manifest.samples[index]


def test_get_dataset_labels_reads_manifest_samples():
    dataset = LabelDataset([0, 1, 0])

    assert get_dataset_labels(dataset) == [0, 1, 0]


def test_get_dataset_labels_rejects_dataset_without_manifest_samples():
    with pytest.raises(ValueError, match="manifest.samples"):
        get_dataset_labels(Dataset())


def test_calculate_class_weights_matches_inverse_frequency():
    weights = calculate_class_weights({0: 3, 1: 1})

    assert torch.allclose(weights, torch.tensor([2 / 3, 2.0]))


def test_target_fraction_sample_weights_balance_total_class_mass():
    weights = calculate_target_fraction_sample_weights(
        labels=[0, 0, 0, 1],
        target_pothole_fraction=0.5,
    )

    assert weights.dtype == torch.double
    assert weights[:3].sum().item() == pytest.approx(0.5)
    assert weights[3:].sum().item() == pytest.approx(0.5)


def test_target_fraction_sample_weights_support_custom_target_fraction():
    weights = calculate_target_fraction_sample_weights(
        labels=[0, 0, 0, 1],
        target_pothole_fraction=0.4,
    )

    assert weights[:3].sum().item() == pytest.approx(0.6)
    assert weights[3:].sum().item() == pytest.approx(0.4)


def test_target_fraction_sample_weights_reject_invalid_fraction():
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_target_fraction_sample_weights(
            labels=[0, 1],
            target_pothole_fraction=1.0,
        )


def test_target_fraction_sample_weights_requires_both_classes():
    with pytest.raises(ValueError, match="both pothole and non-pothole"):
        calculate_target_fraction_sample_weights(
            labels=[0, 0],
            target_pothole_fraction=0.5,
        )


def test_make_weighted_sampler_uses_one_sample_per_dataset_row():
    sampler = make_weighted_sampler(
        labels=[0, 0, 1],
        target_pothole_fraction=0.5,
    )

    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 3
    assert sampler.replacement is True


def test_sampling_config_uses_shuffle_for_normal_training():
    config = make_rdd_sampling_config(
        dataset=LabelDataset([0, 1]),
        is_train=True,
        sampler_strategy="none",
    )

    assert config.sampler is None
    assert config.shuffle is True


def test_sampling_config_disables_shuffle_for_evaluation():
    config = make_rdd_sampling_config(
        dataset=LabelDataset([0, 1]),
        is_train=False,
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
    )

    assert config.sampler is None
    assert config.shuffle is False


def test_sampling_config_builds_weighted_sampler_for_training():
    config = make_rdd_sampling_config(
        dataset=LabelDataset([0, 0, 1]),
        is_train=True,
        sampler_strategy="weighted_sampler",
        target_pothole_fraction=0.5,
    )

    assert isinstance(config.sampler, WeightedRandomSampler)
    assert config.shuffle is False


def test_sampling_config_requires_target_fraction_for_weighted_sampler():
    with pytest.raises(ValueError, match="target_pothole_fraction"):
        make_rdd_sampling_config(
            dataset=LabelDataset([0, 1]),
            is_train=True,
            sampler_strategy="weighted_sampler",
        )


def test_sampling_config_rejects_unknown_sampler_strategy():
    with pytest.raises(ValueError, match="sampler_strategy"):
        make_rdd_sampling_config(
            dataset=LabelDataset([0, 1]),
            is_train=True,
            sampler_strategy="balanced_magic",
        )
