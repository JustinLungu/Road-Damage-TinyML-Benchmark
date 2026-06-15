# Detection Metrics Results

`detection_benchmark_results.csv` contains one row for each accuracy evaluation.
The `task`, `dataset_name`, and `split` columns identify which rows are directly
comparable.

## Shared Columns

- `precision`, `recall`, and `f1_score` are task-specific quality metrics.
- `num_images` records the evaluated subset size.
- Non-applicable metric columns are blank.

## Object Detection Rows

- `map_50_95`, `map_50`, and `map_75` are standard COCO bounding-box AP.
- `precision`, `recall`, and `f1_score` use the recorded confidence and IoU
  thresholds.
- `mean_iou` averages IoU over true-positive matches at those thresholds.

## Image Classification Rows

- `top1_accuracy` and `top5_accuracy` measure whether the correct zero-based
  class ID appears in the highest one or five logits.
- `precision`, `recall`, and `f1_score` are macro averages over classes present
  in the selected ground truth.

The default classification rows use Imagenette, a ten-class subset of ImageNet.
Compare rows only when `dataset_name` and `split` match.

Do not compare classification scores directly with object-detection scores.

## Plot Folders

Run:

```bash
uv run python experiments/detection/plot_detection_benchmark.py
```

This creates:

- `object_detection/` for YOLO-only detection metrics.
- `image_classification/` for classifier-only metrics.
- `common_metrics/` for precision, recall, and F1 across every model.

Common plots use blue for image classification and orange for object detection.
They also include a warning that shared metric names retain task-specific
definitions and are calculated on different datasets.
