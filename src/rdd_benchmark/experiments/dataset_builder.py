from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.constants import (
    RDD2022_BINARY_POTHOLE_DIR,
    RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
    RDD2022_SYNTHETIC_POTHOLE_DIR,
)
from src.rdd_benchmark.data_preprocessing.balancing import (
    prepare_balanced_binary_pothole_manifests,
)
from src.rdd_benchmark.data_preprocessing.synthetic import (
    prepare_synthetic_binary_pothole_manifests,
)
from src.rdd_benchmark.experiments.constants import (
    RDD_EXPERIMENT_D,
    RDD_EXPERIMENT_E,
)
from src.rdd_benchmark.experiments.experiment_registry import RDDExperimentConfig


@dataclass(frozen=True)
class RDDExperimentManifestPaths:
    experiment_name: str
    dataset_dir: Path
    train_manifest_path: Path
    validation_manifest_path: Path
    test_manifest_path: Path


def make_manifest_paths(
    experiment_name: str,
    dataset_dir: Path,
) -> RDDExperimentManifestPaths:
    return RDDExperimentManifestPaths(
        experiment_name=experiment_name,
        dataset_dir=dataset_dir,
        train_manifest_path=dataset_dir / "train.csv",
        validation_manifest_path=dataset_dir / "validation.csv",
        test_manifest_path=dataset_dir / "test.csv",
    )


def validate_manifest_paths(manifest_paths: RDDExperimentManifestPaths) -> None:
    for manifest_path in (
        manifest_paths.train_manifest_path,
        manifest_paths.validation_manifest_path,
        manifest_paths.test_manifest_path,
    ):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Experiment manifest is missing: {manifest_path}")


def build_original_manifest_paths(
    config: RDDExperimentConfig,
    source_dir: Path = RDD2022_BINARY_POTHOLE_DIR,
) -> RDDExperimentManifestPaths:
    manifest_paths = make_manifest_paths(config.experiment_name, source_dir)
    validate_manifest_paths(manifest_paths)
    return manifest_paths


def build_downsampled_manifest_paths(
    config: RDDExperimentConfig,
    source_dir: Path = RDD2022_BINARY_POTHOLE_DIR,
    output_root: Path = RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
) -> RDDExperimentManifestPaths:
    output_dir = prepare_balanced_binary_pothole_manifests(
        experiment_name=config.experiment_name,
        non_potholes_per_pothole=config.non_potholes_per_pothole,
        source_dir=source_dir,
        output_root=output_root,
        random_seed=config.random_seed,
    )
    manifest_paths = make_manifest_paths(config.experiment_name, output_dir)
    validate_manifest_paths(manifest_paths)
    return manifest_paths


def build_synthetic_manifest_paths(
    config: RDDExperimentConfig,
    best_previous_experiment_name: str | None,
    experiment_output_root: Path = RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
    synthetic_source_dir: Path = RDD2022_SYNTHETIC_POTHOLE_DIR,
) -> RDDExperimentManifestPaths:
    if not best_previous_experiment_name:
        raise ValueError(
            "Experiment E requires best_previous_experiment_name so synthetic "
            "rows are added to the selected best C/D dataset."
        )

    source_dir = experiment_output_root / best_previous_experiment_name
    output_dir = prepare_synthetic_binary_pothole_manifests(
        experiment_name=config.experiment_name,
        synthetic_pothole_ratio=config.synthetic_pothole_ratio,
        source_dir=source_dir,
        synthetic_source_dir=synthetic_source_dir,
        output_root=experiment_output_root,
        random_seed=config.random_seed,
    )
    manifest_paths = make_manifest_paths(config.experiment_name, output_dir)
    validate_manifest_paths(manifest_paths)
    return manifest_paths


def build_experiment_manifest_paths(
    config: RDDExperimentConfig,
    source_dir: Path = RDD2022_BINARY_POTHOLE_DIR,
    experiment_output_root: Path = RDD2022_BINARY_POTHOLE_EXPERIMENTS_DIR,
    synthetic_source_dir: Path = RDD2022_SYNTHETIC_POTHOLE_DIR,
    best_previous_experiment_name: str | None = None,
) -> RDDExperimentManifestPaths:
    if config.experiment_id == RDD_EXPERIMENT_D:
        return build_downsampled_manifest_paths(
            config,
            source_dir,
            experiment_output_root,
        )
    if config.experiment_id == RDD_EXPERIMENT_E:
        return build_synthetic_manifest_paths(
            config,
            best_previous_experiment_name,
            experiment_output_root,
            synthetic_source_dir,
        )
    return build_original_manifest_paths(config, source_dir)
