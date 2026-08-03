# RDD2022 Dataset

RDD2022 is a multi-country road-damage object-detection dataset released for
CRDDC 2022. Its annotations use Pascal VOC XML files and cover four challenge
classes:

- `D00`: longitudinal crack
- `D10`: transverse crack
- `D20`: alligator crack
- `D40`: pothole

RDD2022 does not provide an annotated validation split. The official `train`
split contains images and annotations, while the challenge `test` split
contains images only. This repository therefore derives local train,
validation, and test manifests from the annotated training data.

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

These local splits support reproducible model development, but they are not the
official challenge evaluation. The default split is image-level and does not
group adjacent frames, so results should not be presented as sequence-level or
private-test performance.

## Download The Dataset

The benchmark expects every country listed in
`src/rdd_benchmark/data_preprocessing/constants.py`. Download all available
annotated country archives from the repository root:

```bash
./scripts/download_rdd2022_subset.sh all
```

The downloader extracts each annotated `train/` directory, skips countries
already present, and does not retain the downloaded ZIP files. Norway is
available but its official archive is approximately 9.9 GB, so the complete
download is large.

For dataset inspection or a custom country configuration, list and download
individual archives instead:

```bash
./scripts/download_rdd2022_subset.sh --list
./scripts/download_rdd2022_subset.sh
./scripts/download_rdd2022_subset.sh china-motorbike
./scripts/download_rdd2022_subset.sh czech
./scripts/download_rdd2022_subset.sh united-states
```

The command without an argument downloads India.

## Resulting Layout

Each country uses the same layout:

```text
datasets/rdd2022/
├── README.md
├── India/
│   └── train/
│       ├── annotations/
│       │   └── xmls/
│       │       └── *.xml
│       └── images/
│           └── *.jpg
└── <other_country>/
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

The repository creates binary pothole/non-pothole manifests from the downloaded
XML annotations. All RDD actions are controlled from
`src/rdd_benchmark/constants.py` and run through the single entry point:

```bash
uv run python -m src.rdd_benchmark.main
```

For a detailed explanation of each RDD pipeline constant, see
`docs/rdd_benchmark_constants.md`.

The clean400 binary target is:

- `1`, `pothole`: the image contains at least one `D40` bounding box with an
  area of at least 400 square pixels.
- `0`, `non_pothole`: the image contains no qualifying `D40` bounding box.

`non_pothole` therefore includes images with other road-damage classes, repair
labels, no annotated objects, or only smaller `D40` boxes.

The default split is `stratified_by_country`:

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

Preprocessing writes generated CSV files under:

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

Only image-classification models are supported by this benchmark. Pretrained
MobileNet, EfficientNet, ResNet, Inception, MobileViT, EfficientFormer, and
ShuffleNet models are fine-tuned. The custom tiny classifiers are trained from
scratch. YOLO detectors and SmolVLM vision-language models are excluded.

To change country-holdout assignments, edit `SPLIT_COUNTRIES`. To change the
countries used by the stratified split, edit `RDD_AVAILABLE_COUNTRIES`. Both
live in `src/rdd_benchmark/data_preprocessing/constants.py`. Then enable
preprocessing and rerun:

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
