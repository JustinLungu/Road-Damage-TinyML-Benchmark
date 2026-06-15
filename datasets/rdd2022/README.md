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

Norway is available but its official archive is approximately 9.9 GB.

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
