from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from src.rdd_benchmark.constants import NEGATIVE_LABEL, NUM_BINARY_CLASSES, POSITIVE_LABEL
from src.rdd_benchmark.data_loader.constants import (
    RDD_SAMPLER_NONE,
    RDD_SUPPORTED_SAMPLER_STRATEGIES,
)


@dataclass(frozen=True)
class RDDSamplingConfig:
    sampler: Sampler | None
    shuffle: bool


def get_dataset_labels(dataset: Dataset) -> list[int]:
    manifest = getattr(dataset, "manifest", None)
    samples = getattr(manifest, "samples", None)
    if samples is None:
        raise ValueError("Dataset must expose manifest.samples with labels.")

    labels = [int(sample.label) for sample in samples]
    if not labels:
        raise ValueError("Cannot sample from an empty dataset.")
    return labels


def calculate_class_weights(class_counts: dict[int, int]) -> torch.Tensor:
    total_count = sum(class_counts.get(label, 0) for label in range(NUM_BINARY_CLASSES))
    if total_count == 0:
        raise ValueError("Cannot calculate class weights for an empty dataset.")

    weights = []
    for label in range(NUM_BINARY_CLASSES):
        label_count = class_counts.get(label, 0)
        if label_count == 0:
            raise ValueError(f"Cannot calculate class weight for missing label: {label}")
        weights.append(total_count / (NUM_BINARY_CLASSES * label_count))

    return torch.tensor(weights, dtype=torch.float32)


def calculate_target_fraction_sample_weights(
    labels: Iterable[int],
    target_pothole_fraction: float,
) -> torch.Tensor:
    if not 0.0 < target_pothole_fraction < 1.0:
        raise ValueError("target_pothole_fraction must be between 0 and 1.")

    labels = [int(label) for label in labels]
    counts = Counter(labels)
    positive_count = counts.get(POSITIVE_LABEL, 0)
    negative_count = counts.get(NEGATIVE_LABEL, 0)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Weighted sampling requires both pothole and non-pothole labels.")

    class_weight_by_label = {
        POSITIVE_LABEL: target_pothole_fraction / positive_count,
        NEGATIVE_LABEL: (1.0 - target_pothole_fraction) / negative_count,
    }
    return torch.tensor(
        [class_weight_by_label[int(label)] for label in labels],
        dtype=torch.double,
    )


def make_weighted_sampler(
    labels: Iterable[int],
    target_pothole_fraction: float,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    sample_weights = calculate_target_fraction_sample_weights(
        labels,
        target_pothole_fraction,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def make_rdd_sampling_config(
    dataset: Dataset,
    is_train: bool,
    sampler_strategy: str = RDD_SAMPLER_NONE,
    target_pothole_fraction: float | None = None,
    generator: torch.Generator | None = None,
) -> RDDSamplingConfig:
    if sampler_strategy not in RDD_SUPPORTED_SAMPLER_STRATEGIES:
        raise ValueError(
            "sampler_strategy must be one of: "
            f"{', '.join(RDD_SUPPORTED_SAMPLER_STRATEGIES)}."
        )
    if not is_train:
        return RDDSamplingConfig(sampler=None, shuffle=False)
    if sampler_strategy == RDD_SAMPLER_NONE:
        return RDDSamplingConfig(sampler=None, shuffle=True)
    if target_pothole_fraction is None:
        raise ValueError("target_pothole_fraction is required for weighted sampling.")

    labels = get_dataset_labels(dataset)
    sampler = make_weighted_sampler(
        labels,
        target_pothole_fraction,
        generator,
    )
    return RDDSamplingConfig(sampler=sampler, shuffle=False)
