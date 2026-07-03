# RDD Training Constants

All RDD preprocessing, training, evaluation, and comparison work should be run
through:

```bash
uv run python -m src.rdd_training.main
```

Change behavior by editing `src/rdd_training/constants.py`.

## Preprocessing Switches

```python
RUN_RDD_FULL_IMAGE_PREPROCESSING = False
```

If `True`, `main.py` regenerates full-image binary manifests:

```text
datasets/rdd2022/binary_pothole/
```

Each CSV row is one full road image. The image is labeled `pothole` if its XML
contains at least one `D40` object.

```python
RUN_RDD_PATCH_PREPROCESSING = True
```

If `True`, `main.py` regenerates annotation-patch manifests:

```text
datasets/rdd2022/binary_pothole_patches/
```

Each CSV row is one crop around an XML bounding box. `D40` boxes are positive
pothole patches. Non-`D40` boxes are negative non-pothole patches.

After the CSVs exist, set preprocessing flags back to `False` unless you want to
overwrite the manifests.

## Experiment Name

```python
RDD_EXPERIMENT_NAME = "annotation_patch_grid3_smoke"
```

This controls where results are saved:

```text
results/rdd_trained_models/<RDD_EXPERIMENT_NAME>/<model_name>/
```

Use a new experiment name when changing the pipeline, model group, or important
hyperparameters.

## Training Input Mode

```python
RDD_TRAINING_INPUT_MODE = "annotation_patch"
```

Available modes:

- `"full_image"`: train directly on full road images from
  `datasets/rdd2022/binary_pothole/`.
- `"annotation_patch"`: train on XML annotation-centered crops from
  `datasets/rdd2022/binary_pothole_patches/`.

For the paper-aligned patch-grid pipeline, use:

```python
RDD_TRAINING_INPUT_MODE = "annotation_patch"
```

## Evaluation Input Mode

```python
RDD_EVALUATION_INPUT_MODE = "grid_image"
```

Available modes:

- `"full_image"`: evaluate each full test image directly.
- `"grid_image"`: split each full test image into a grid, score every patch,
  and classify the full image using the maximum patch pothole probability.

For realistic patch-grid inference without test-time bounding boxes, use:

```python
RDD_EVALUATION_INPUT_MODE = "grid_image"
```

## Patch Settings

```python
RDD_PATCH_PADDING = 0.15
RDD_MIN_BOX_AREA = 400
```

`RDD_PATCH_PADDING` expands annotation boxes before cropping patches.
`RDD_MIN_BOX_AREA` skips tiny boxes that are likely too small/noisy for useful
classification.

These settings affect patch manifest generation, so rerun patch preprocessing if
you change them.

## Grid Settings

```python
RDD_GRID_SIZE = 3
RDD_GRID_OVERLAP = 0.0
RDD_PATCH_DECISION_THRESHOLD = 0.5
```

`RDD_GRID_SIZE = 3` means every full image is split into `3 x 3 = 9` patches.

`RDD_GRID_OVERLAP` is reserved for later. Currently only `0.0` is implemented.

`RDD_PATCH_DECISION_THRESHOLD` is the fallback threshold. If threshold tuning is
enabled, the tuned validation threshold is used instead.

## Threshold Tuning

```python
RDD_TUNE_PATCH_THRESHOLD = True
RDD_THRESHOLD_METRIC = "f1"
RDD_THRESHOLD_VALUES = tuple(index / 100 for index in range(5, 96, 5))
```

If tuning is enabled, validation full images are evaluated using grid inference.
The threshold that gives the best `RDD_THRESHOLD_METRIC` is saved to:

```text
threshold.json
```

Then test evaluation uses that threshold.

This is important because RDD pothole labels are imbalanced, so a fixed `0.5`
threshold may not maximize F1 or recall.

## Model Selection

```python
RDD_MODEL_MODE = "single"
RDD_SINGLE_MODEL = "mobilevit_xxs"
```

Available modes:

- `"single"`: train/evaluate only `RDD_SINGLE_MODEL`.
- `"all"`: train/evaluate every model listed in `RDD_MODEL_NAMES`.

For smoke testing, use one small model:

```python
RDD_MODEL_MODE = "single"
RDD_SINGLE_MODEL = "mobilevit_xxs"
```

## Training Controls

```python
RUN_RDD_TRAINING = True
RDD_TRAINING_EPOCHS = 1
RDD_TRAINING_BATCH_SIZE = 16
```

For smoke testing, keep `RDD_TRAINING_EPOCHS = 1`.

For a real run, increase epochs after the smoke test passes, for example:

```python
RDD_TRAINING_EPOCHS = 20
```

## Evaluation And Comparison

```python
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
RDD_COMPARISON_RANKING_METRIC = "f1"
```

Evaluation writes per-model outputs:

```text
test_metrics.json
test_metrics.csv
threshold.json
inference_metrics.json
confusion_matrix.csv
confusion_matrix.png
roc_curve.png
test_metric_bars.png
```

Comparison writes:

```text
model_comparison.csv
top_models.csv
```

Models are ranked by `RDD_COMPARISON_RANKING_METRIC`. For this imbalanced binary
pothole task, `f1`, `recall`, or `balanced_accuracy` are more meaningful than
plain accuracy.

## Common Setups

### Current Smoke Test

```python
RUN_RDD_PATCH_PREPROCESSING = True
RDD_EXPERIMENT_NAME = "annotation_patch_grid3_smoke"
RDD_MODEL_MODE = "single"
RDD_SINGLE_MODEL = "mobilevit_xxs"
RDD_TRAINING_INPUT_MODE = "annotation_patch"
RDD_EVALUATION_INPUT_MODE = "grid_image"
RDD_GRID_SIZE = 3
RDD_TRAINING_EPOCHS = 1
RUN_RDD_TRAINING = True
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
```

Run:

```bash
uv run python -m src.rdd_training.main
```

### Full-Image Baseline

```python
RDD_EXPERIMENT_NAME = "full_image_baseline"
RDD_TRAINING_INPUT_MODE = "full_image"
RDD_EVALUATION_INPUT_MODE = "full_image"
RUN_RDD_PATCH_PREPROCESSING = False
```

### Patch-Grid Real Run

```python
RUN_RDD_PATCH_PREPROCESSING = False
RDD_EXPERIMENT_NAME = "annotation_patch_grid3"
RDD_MODEL_MODE = "all"
RDD_TRAINING_INPUT_MODE = "annotation_patch"
RDD_EVALUATION_INPUT_MODE = "grid_image"
RDD_GRID_SIZE = 3
RDD_TRAINING_EPOCHS = 20
RUN_RDD_TRAINING = True
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
```
