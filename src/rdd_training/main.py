from __future__ import annotations

from pathlib import Path

from src.constants import RDD2022_BINARY_POTHOLE_DIR
from src.rdd_training.dataset import BinaryPotholeDataset, BinaryPotholeManifest


SPLIT_MANIFESTS = {
    "train": RDD2022_BINARY_POTHOLE_DIR / "train.csv",
    "validation": RDD2022_BINARY_POTHOLE_DIR / "validation.csv",
    "test": RDD2022_BINARY_POTHOLE_DIR / "test.csv",
}

def summarize_split(split: str, manifest_path: Path) -> None:
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


if __name__ == "__main__":
    print("RDD2022 binary pothole dataset loader")
    print(f"Manifest directory: {RDD2022_BINARY_POTHOLE_DIR}")

    for split, manifest_path in SPLIT_MANIFESTS.items():
        summarize_split(split, manifest_path)



