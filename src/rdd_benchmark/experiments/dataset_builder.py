from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.constants import (
    RDD2022_BINARY_POTHOLE_DIR,
    RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
)
from src.rdd_benchmark.data_preprocessing.balancing import (
    prepare_balanced_binary_pothole_manifests,
)
from src.rdd_benchmark.data_preprocessing.constants import RDD_SPLIT_RANDOM_SEED
from src.rdd_benchmark.experiments.experiment_registry import RDDExperimentConfig


@dataclass(frozen=True)
class RDDExperimentManifestPaths:
    dataset_dir: Path

    @property
    def train_manifest_path(self) -> Path:
        return self.dataset_dir / "train.csv"

    @property
    def validation_manifest_path(self) -> Path:
        return self.dataset_dir / "validation.csv"

    @property
    def test_manifest_path(self) -> Path:
        return self.dataset_dir / "test.csv"


def validate_manifest_paths(manifest_paths: RDDExperimentManifestPaths) -> None:
    for manifest_path in (
        manifest_paths.train_manifest_path,
        manifest_paths.validation_manifest_path,
        manifest_paths.test_manifest_path,
    ):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Experiment manifest is missing: {manifest_path}")


def build_experiment_manifest_paths(
    config: RDDExperimentConfig,
    source_dir: Path = RDD2022_BINARY_POTHOLE_DIR,
    experiment_output_root: Path = RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
) -> RDDExperimentManifestPaths:
    dataset_dir = source_dir
    if config.non_potholes_per_pothole is not None:
        dataset_dir = prepare_balanced_binary_pothole_manifests(
            config.experiment_name,
            config.non_potholes_per_pothole,
            source_dir,
            experiment_output_root,
            RDD_SPLIT_RANDOM_SEED,
        )

    manifest_paths = RDDExperimentManifestPaths(dataset_dir)
    validate_manifest_paths(manifest_paths)
    return manifest_paths
