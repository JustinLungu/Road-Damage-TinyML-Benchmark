from __future__ import annotations

from src.constants import RDD2022_BINARY_POTHOLE_DIR
from src.rdd_training.constants import (
    RDD_MODEL_MODE,
    RUN_RDD_TRAINING,
)
from src.rdd_training.dataset import BinaryPotholeDataset, BinaryPotholeManifest
from src.rdd_training.trainer import BinaryPotholeTrainer, RDDTrainingConfig
from src.rdd_training.utils import (
    load_and_adapt_all_binary_pothole_models,
    load_and_adapt_model_for_binary_pothole,
    select_rdd_model_names,
)


SPLIT_MANIFESTS = {
    "train": RDD2022_BINARY_POTHOLE_DIR / "train.csv",
    "validation": RDD2022_BINARY_POTHOLE_DIR / "validation.csv",
    "test": RDD2022_BINARY_POTHOLE_DIR / "test.csv",
}


if __name__ == "__main__":
    print("RDD2022 binary pothole dataset loader")
    print(f"Manifest directory: {RDD2022_BINARY_POTHOLE_DIR}")

    for split, manifest_path in SPLIT_MANIFESTS.items():
        manifest = BinaryPotholeManifest.from_csv(
            manifest_path,
            expected_split=split,
        )
        dataset = BinaryPotholeDataset(manifest)
        first_item = dataset[0]
        class_counts = manifest.class_counts()

        print()
        print(f"{split}:")
        print(f"  samples: {len(dataset)}")
        print(f"  non_pothole: {class_counts[0]}")
        print(f"  pothole: {class_counts[1]}")
        print(f"  pothole_fraction: {manifest.positive_fraction():.3f}")
        print(f"  countries: {dict(manifest.country_counts())}")
        print(
            "  first_sample: "
            f"{first_item['image'].mode} {first_item['image'].size}, "
            f"label={first_item['label']} ({first_item['label_name']}), "
            f"country={first_item['country']}"
        )

    print()
    print("Model adaptation smoke test")
    if RDD_MODEL_MODE == "all":
        adapted_models = load_and_adapt_all_binary_pothole_models()
        for model_name, adapted_model in adapted_models.items():
            print(f"  adapted_model: {model_name} -> {type(adapted_model).__name__}")
    elif RDD_MODEL_MODE == "single":
        model_name = select_rdd_model_names()[0]
        adapted_model = load_and_adapt_model_for_binary_pothole(model_name)
        print(
            "  adapted_model: "
            f"{model_name} -> {type(adapted_model).__name__}"
        )
    else:
        raise ValueError("RDD_MODEL_MODE must be 'single' or 'all'.")

    if RUN_RDD_TRAINING:
        print()
        print("RDD2022 binary pothole training")
        for model_name in select_rdd_model_names():
            print()
            print(f"Training {model_name}")
            trainer = BinaryPotholeTrainer(RDDTrainingConfig(model_name=model_name))
            result = trainer.train()
            print(
                "  best_checkpoint: "
                f"{result.best_checkpoint_path} "
                f"({result.best_metric_name}={result.best_metric_value:.4f}, "
                f"epoch={result.best_epoch})"
            )
