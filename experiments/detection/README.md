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
