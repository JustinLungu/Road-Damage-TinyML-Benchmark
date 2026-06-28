from __future__ import annotations

from src.constants import RDD2022_BINARY_POTHOLE_DIR
from src.rdd_training.constants import (
    MODEL_ADAPTATION_SMOKE_TEST_MODE,
    MODEL_ADAPTATION_SMOKE_TEST_MODEL,
)
from src.rdd_training.dataset import BinaryPotholeDataset, BinaryPotholeManifest
from src.rdd_training.utils import (
    load_and_adapt_all_binary_pothole_models,
    load_and_adapt_model_for_binary_pothole,
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
    if MODEL_ADAPTATION_SMOKE_TEST_MODE == "all":
        adapted_models = load_and_adapt_all_binary_pothole_models()
        for model_name, adapted_model in adapted_models.items():
            print(f"  adapted_model: {model_name} -> {type(adapted_model).__name__}")
    elif MODEL_ADAPTATION_SMOKE_TEST_MODE == "single":
        adapted_model = load_and_adapt_model_for_binary_pothole(
            MODEL_ADAPTATION_SMOKE_TEST_MODEL,
        )
        print(
            "  adapted_model: "
            f"{MODEL_ADAPTATION_SMOKE_TEST_MODEL} -> {type(adapted_model).__name__}"
        )
    else:
        raise ValueError(
            "MODEL_ADAPTATION_SMOKE_TEST_MODE must be 'single' or 'all'."
        )
