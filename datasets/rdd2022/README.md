# RDD2022 Country Training Subset

RDD2022 is a multi-country road-damage object-detection dataset released for
CRDDC 2022. Its annotations use Pascal VOC XML files and cover four challenge
classes:

- `D00`: longitudinal crack
- `D10`: transverse crack
- `D20`: alligator crack
- `D40`: pothole

RDD2022 does not provide an annotated validation split. The official `train`
split contains images and annotations, while the challenge `test` split
contains images only. For the first domain-specific experiments, this
repository therefore downloads one country's annotated training data rather
than the full multi-country archive.

## How The Official Test Set Was Evaluated

The test annotations were hidden from challenge participants, not absent from
the organizers' evaluation system. Participants downloaded the test images,
ran their models, and submitted predicted classes, bounding boxes, and
confidence scores to the CRDDC evaluation server. The server compared those
predictions with private ground-truth annotations and returned the challenge
score.

This prevents participants from tuning models directly against the final test
labels. The public download therefore contains test images without their XML
annotations.

For local experiments, create deterministic training, validation, and test
partitions from the annotated country `train` data. A reasonable initial split
is 80% training, 10% validation, and 10% local testing. Keep the local test
partition untouched until final evaluation. When images are sequential road
frames, split related frames as a group so near-duplicate scenes do not leak
between partitions.

## Download The Default Subset

Run the script from the repository root:

```bash
./scripts/download_rdd2022_subset.sh
```

The default is India. It is a road-facing, vehicle-mounted subset and has
substantial pothole representation. The source archive is approximately
502.3 MB.

Only `India/train/` is extracted. The unannotated test images and downloaded
ZIP file are not retained.

## Select Another Country

List the official country-specific archives:

```bash
./scripts/download_rdd2022_subset.sh --list
```

Download another annotated training subset:

```bash
./scripts/download_rdd2022_subset.sh china-motorbike
./scripts/download_rdd2022_subset.sh czech
./scripts/download_rdd2022_subset.sh united-states
```

Download every available country-specific annotated training subset:

```bash
./scripts/download_rdd2022_subset.sh all
```

The `all` command skips country folders that already contain a `train/`
directory. Norway is available but its official archive is approximately
9.9 GB, so the full download is large.

## Resulting Layout

The default command creates:

```text
datasets/rdd2022/
├── README.md
└── India/
    └── train/
        ├── annotations/
        │   └── xmls/
        │       └── *.xml
        └── images/
            └── *.jpg
```

Each XML file contains the image metadata, damage class, and bounding-box
coordinates for its corresponding image.

## Binary Pothole Image Classification

For the initial supervisor-requested image-classification task, the repository
can create binary pothole/non-pothole manifests from the downloaded XML
annotations. All RDD training actions are controlled from
`src/rdd_benchmark/constants.py` and run through the single entry point:

```bash
uv run python -m src.rdd_benchmark.main
```

For a detailed explanation of each RDD pipeline constant, see
`docs/rdd_benchmark_constants.md`.

The binary target is:

- `1`, `pothole`: the image contains at least one `D40` object.
- `0`, `non_pothole`: the image contains no `D40` object.

`non_pothole` therefore includes images with other road-damage classes, repair
labels, and images with no annotated objects. This keeps the first task aligned
with the requested question: pothole versus non-pothole.

The default fine-tuning split is now `stratified_by_country`:

- every country contributes to train, validation, and test;
- pothole and non-pothole images are split separately inside each country;
- the default ratio is `70%` train, `15%` validation, and `15%` test.

This gives validation enough pothole examples for model selection while keeping
the test set meaningful. The older country-holdout split is still available for
a harder cross-country generalization benchmark:

| Split | Countries |
| --- | --- |
| `train` | `China_Drone`, `China_MotorBike`, `Czech`, `India` |
| `validation` | `United_States` |
| `test` | `Japan`, `Norway` |

The script writes local generated CSV files under:

```text
datasets/rdd2022/binary_pothole/
├── all.csv
├── train.csv
├── validation.csv
├── test.csv
└── summary.csv
```

These generated manifests are ignored by Git. Each row contains the image path,
annotation path, country, split, binary label, object-label metadata, and image
size. Use `summary.csv` to check class balance before training.

To regenerate these full-image manifests, set:

```python
RUN_RDD_PREPROCESSING = True
```

Then run:

```bash
uv run python -m src.rdd_benchmark.main
```

After the manifests are generated, set the preprocessing flag back to `False`
unless you changed the dataset split or want to overwrite the CSV files.

Load the generated manifests for training with:

```python
from pathlib import Path

from src.rdd_benchmark.data_loader.dataset import BinaryPotholeDataset

train_dataset = BinaryPotholeDataset(
    Path("datasets/rdd2022/binary_pothole/train.csv"),
    expected_split="train",
)
```

Smoke-test the loader and print split summaries with:

```bash
uv run python -m src.rdd_benchmark.main
```

Adapt a pretrained image-classification model for binary pothole fine-tuning:

```python
from src.load_model import load_model
from src.rdd_benchmark.training.model_adapter import adapt_model_for_binary_pothole

model_name = "mobilenet_v3_small"
model = load_model(model_name)
model = adapt_model_for_binary_pothole(model_name, model)
```

Only image-classification models are supported for this RDD fine-tuning path:
MobileNet, EfficientNet-B0, ResNet18, InceptionV3, MobileViT, and
EfficientFormer. YOLO detectors and SmolVLM vision-language models are
intentionally excluded.

To change the country split, edit `SPLIT_COUNTRIES` in
`src/rdd_benchmark/data_preprocessing/constants.py`, enable preprocessing, then
rerun:

```bash
uv run python -m src.rdd_benchmark.main
```

## Source And License

- Official project:
  <https://github.com/sekilab/RoadDamageDetector#crowdsensing-based-road-damage-detection-challenge-crddc2022>
- Dataset record:
  <https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547>
- Dataset article:
  <https://doi.org/10.1111/gdj3.155>

The official project states that dataset images are distributed under
CC BY-SA 4.0. Review the official dataset page before redistributing derived
data.
