# Imagenette Validation Dataset

Imagenette is a ten-class subset of ImageNet created by fastai. This repository
uses the 160-pixel version's labeled validation split as the initial accuracy
dataset for all ImageNet-pretrained classifiers.

Download and prepare it from the repository root:

```bash
./scripts/download_imagenette_val.sh
```

The script downloads the official archive, keeps only `val/`, and creates:

```text
datasets/imagenette/
├── README.md
├── validation/
│   └── images/
│       ├── n01440764/
│       ├── ...
│       └── n03888257/
└── validation_labels.csv
```

The manifest contains paths and standard zero-based ImageNet-1K class IDs:

```csv
image_path,class_id
validation/images/n01440764/example.JPEG,0
```

## Why This Dataset

COCO and RDD2022 are object-detection datasets. Their labels cannot determine
whether a 1000-class ImageNet classifier predicted the correct whole-image
class.

Imagenette provides labeled classification images while remaining much smaller
than the full ImageNet validation set. It is suitable for an initial comparison
between the repository's classifiers, but its ten easy classes are not a
replacement for reporting full ImageNet-1K validation accuracy.

## Classes

| Synset | ImageNet class ID | Class |
| --- | ---: | --- |
| `n01440764` | 0 | tench |
| `n02102040` | 217 | English springer |
| `n02979186` | 482 | cassette player |
| `n03000684` | 491 | chain saw |
| `n03028079` | 497 | church |
| `n03394916` | 566 | French horn |
| `n03417042` | 569 | garbage truck |
| `n03425413` | 571 | gas pump |
| `n03445777` | 574 | golf ball |
| `n03888257` | 701 | parachute |

## Source

- Official Imagenette repository:
  <https://github.com/fastai/imagenette>
- Official 160-pixel archive:
  <https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz>

Review the source repository and ImageNet image terms before redistributing the
downloaded images.
