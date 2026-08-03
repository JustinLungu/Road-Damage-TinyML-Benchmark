# Vision Quality Experiment

This experiment measures prediction quality rather than runtime performance.
It supports pretrained YOLO object detectors and pretrained ImageNet-style
classifiers. The tasks use separate datasets, metrics, result files, and plots.

## Object Detection

Evaluate YOLO on COCO 2017 validation images and annotations:

```bash
uv run python experiments/vision/vision_benchmark.py \
  --model yolov8n \
  --device cuda:0
```

COCO mAP uses all retained predictions. Precision, recall, F1, and mean IoU use
confidence `0.25` and IoU `0.50` by default. Override those operating-point
thresholds with `--confidence-threshold` and `--iou-threshold`.

## Image Classification

Prepare the default Imagenette validation manifest:

```bash
./scripts/download_imagenette_val.sh
```

Then evaluate one or more pretrained classifiers:

```bash
uv run python experiments/vision/vision_benchmark.py \
  --model mobilenet_v2 efficientnet_b0 resnet18 \
  --device cuda:0
```

Custom source-defined models are intentionally excluded because they have no
pretrained ImageNet weights. Train and evaluate those models through
`src.rdd_benchmark.main`.

To use another classification dataset, provide a CSV containing
`image_path,class_id`, where `class_id` is the zero-based ImageNet-1K output
index:

```bash
uv run python experiments/vision/vision_benchmark.py \
  --model mobilenet_v3_small \
  --classification-labels datasets/imagenet/validation_labels.csv \
  --classification-dataset-name imagenet
```

Use `--num-images` for a smoke test. Multiple selected models run in separate
processes so model memory is released between evaluations.

## Results

The tasks write separate files:

```text
results/vision_metrics/classification_results.csv
results/vision_metrics/object_detection_results.csv
```

Rerunning the same model, dataset, split, image count, and detection thresholds
replaces the previous matching row. Use `-o` to clear both files before a run.

## Plots

Generate task-specific plots after evaluation:

```bash
uv run python experiments/vision/plot_vision_benchmark.py
```

Plots are written under:

```text
results/vision_metrics/image_classification/
results/vision_metrics/object_detection/
```

Classification and object-detection metrics are not combined because they have
different definitions and are measured on different datasets.
