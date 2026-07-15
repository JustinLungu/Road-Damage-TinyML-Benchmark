# Detection Metrics Experiment

This experiment evaluates prediction quality rather than runtime performance.
It supports object detection and image classification. Semantic interpretation
is left out until a separate text-output evaluation is designed.

## Object Detection

YOLO models use the COCO 2017 validation images and instance annotations:

```bash
uv run python experiments/detection/detection_benchmark.py \
  --model yolov8n \
  --device cuda:0
```

Use `--num-images` for a smoke test:

```bash
uv run python experiments/detection/detection_benchmark.py \
  --model yolov8n \
  --device cpu \
  --num-images 10
```

COCO mAP uses all retained predictions. Precision, recall, F1, and mean IoU use
confidence `0.25` and IoU `0.50` by default. Change these with
`--confidence-threshold` and `--iou-threshold`.

## Image Classification

COCO labels cannot evaluate ImageNet classifiers. Prepare the default labeled
Imagenette validation split:

```bash
./scripts/download_imagenette_val.sh
```

Then classifiers use it automatically:

```bash
uv run python experiments/detection/detection_benchmark.py \
  --model mobilenet_v2 efficientnet_b0 resnet18 inception_v3 \
  --device cuda:0
```

Imagenette is a labeled ten-class ImageNet subset. It provides a practical
initial comparison but does not replace full ImageNet-1K validation results.

To evaluate another classification dataset, supply a compatible manifest with
`image_path,class_id` columns:

```bash
uv run python experiments/detection/detection_benchmark.py \
  --model mobilenet_v3_small \
  --device cuda:0 \
  --classification-labels datasets/imagenet/validation_labels.csv \
  --classification-dataset-name imagenet
```

The class ID must be the zero-based ImageNet-1K output index. Relative image
paths are resolved from the manifest directory.

Use `--classification-dataset-name` and `--classification-split` to control the
metadata written to the result row.

## Output

Rows are appended to:

`results/detection_metrics/detection_benchmark_results.csv`

Use `-o` to remove the previous CSV before the current run. Multiple selected
models run in separate processes to release model memory between evaluations.

## Plot Results

Create the detection benchmark plots after the CSV has been populated:

```bash
uv run python experiments/detection/plot_detection_benchmark.py
```

The script creates three folders under `results/detection_metrics/`:

- `object_detection/` contains mAP 50-95, mAP 50, mAP 75, precision, recall,
  F1, and mean IoU plots for YOLO models.
- `image_classification/` contains top-1 accuracy, top-5 accuracy, precision,
  recall, and F1 plots for classifiers.
- `common_metrics/` contains precision, recall, and F1 plots with every model.
  Blue bars represent image classification and orange bars represent object
  detection.

The common plots compare identically named metrics, but those metrics have
task-specific definitions and use different datasets. They provide context
across the complete model set; they are not a single cross-task leaderboard.

Use `--csv` to plot another compatible result file:

```bash
uv run python experiments/detection/plot_detection_benchmark.py \
  --csv path/to/detection_benchmark_results.csv
```

Existing `*_bar.png` files in the three output folders are removed before new
plots are written.
