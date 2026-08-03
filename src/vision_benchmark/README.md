# Vision Benchmark

This package evaluates whether a model produced the correct visual prediction.
It is separate from `performance_benchmark`, which measures runtime cost.

Two task-specific paths are supported:

- Object detection evaluates YOLO models against COCO bounding-box annotations.
- Image classification evaluates ImageNet-style classifiers against a CSV label
  manifest.

Semantic interpretation models are intentionally not included yet because
free-text outputs require a different evaluation design.

The default classification dataset is the Imagenette validation split prepared
by `scripts/download_imagenette_val.sh`.

Only classifiers with pretrained ImageNet weights are supported here. Custom
models such as `tiny_cnn` and `resnet8` start with random weights and belong in
the RDD training benchmark instead.

## Object Detection Metrics

COCO mAP is calculated with `pycocotools`:

- `map_50_95` is mean average precision over IoU thresholds 0.50 through 0.95.
- `map_50` is average precision at IoU 0.50.
- `map_75` is average precision at IoU 0.75.

Precision, recall, F1, and mean IoU use one explicit operating point. By
default, predictions need confidence `0.25` and IoU `0.50`. Predictions are
matched greedily by descending confidence to an unmatched ground-truth box of
the same class. COCO crowd boxes can ignore overlapping predictions but do not
count as false negatives.

`mean_iou` is the mean IoU of true-positive matches at that operating point.

## Classification Metrics

The manifest must contain:

```csv
image_path,class_id
images/ILSVRC2012_val_00000001.JPEG,65
```

Paths may be absolute or relative to the manifest. `class_id` must be the
zero-based output index used by the pretrained classifier.

The benchmark reports top-1 accuracy, top-5 accuracy, and macro precision,
recall, and F1. Macro metrics are averaged over classes represented in the
selected ground-truth subset.
