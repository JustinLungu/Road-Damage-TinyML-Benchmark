# RDD Benchmark Constants

All RDD preprocessing, training, evaluation, and comparison work should be run
through:

```bash
uv run python -m src.rdd_benchmark.main
```

Change behavior by editing `src/rdd_benchmark/constants.py`.

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

## Dataset Split

```python
RDD_SPLIT_MODE = "stratified_by_country"
RDD_SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
RDD_SPLIT_RANDOM_SEED = 42
```

Available split modes:

- `"stratified_by_country"`: each country contributes examples to
  train/validation/test, while splitting pothole and non-pothole images
  separately. This is the recommended mode for development runs because
  validation gets more pothole examples for threshold tuning.
- `"country_holdout"`: entire countries are held out into one split using
  `SPLIT_COUNTRIES`. This is useful as a harder cross-country generalization
  benchmark after the model behaves well.

Patch manifests inherit the full-image split, so all annotation patches from the
same original image stay in the same split.

## Experiment Registry

```python
RDD_ACTIVE_EXPERIMENT_IDS = ("A",)
```

This selects metadata from `src/rdd_benchmark/experiments/`. Valid IDs are
`"A"`, `"B"`, `"C"`, `"D"`, `"E"`, `"F"`, `"G"`, and `"H"`. `main.py` runs each selected
experiment through `RDDExperimentRunner`.

```python
RDD_BEST_PREVIOUS_EXPERIMENT_NAME = None
```

This is only needed for Experiment E. Set it to the experiment folder name that
E should use as its synthetic-data base, usually the better of C or D.

Current experiment defaults:

- `"A"`: natural full-image data, no balancing, no sampler, standard
  augmentation, no synthetic data.
- `"B"`: weighted sampler with standard augmentation and an explicit 50%
  pothole target per epoch/batch.
- `"C"`: weighted sampler with stronger minority/pothole augmentation and an
  explicit 50% pothole target.
- `"D"`: moderate majority downsampling, weighted sampler, stronger
  minority/pothole augmentation, and an explicit 50% pothole target. Moderate
  downsampling currently means pothole:non-pothole = 1:3.
- `"E"`: selected previous dataset plus small synthetic pothole addition, with
  synthetic potholes defaulting to 20% of the real pothole count. Its training
  settings are weighted sampler plus stronger minority augmentation.
- `"F"`: gentler weighted sampler with standard augmentation and a 25%
  pothole target.
- `"G"`: downsample-only 1:5 train set with standard augmentation.
- `"H"`: downsample-only 1:5 train set with stronger minority augmentation.

Balanced experiment manifests are written under:

```text
datasets/rdd2022/binary_pothole_experiments/<experiment_name>/
```

For downsampling experiments, only `train.csv` is changed. `validation.csv` and
`test.csv` are copied from the original full-image manifests unchanged.

The experiment dataset builder chooses manifests as follows:

- Experiments without `non_potholes_per_pothole` use the original full-image
  manifests.
- Experiments with `non_potholes_per_pothole` write a new downsampled training
  manifest and keep validation/test unchanged.
- `"E"` adds synthetic potholes to an explicitly selected best previous
  experiment dataset. `RDD_BEST_PREVIOUS_EXPERIMENT_NAME` must be set before
  running E.

The trainer accepts the selected experiment's concrete manifest paths plus the
generic training controls: `sampler_strategy`, `target_pothole_fraction`, and
`augmentation_strategy`. It does not hardcode individual experiment behavior.

The experiment runner executes one selected experiment end to end: build
manifest paths, train selected model(s), evaluate them, and write
`model_comparison.csv` under:

```text
results/rdd_trained_models/<experiment_name>/
```

`RDD_EXPERIMENT_NAME` still exists in `constants.py`, but it is now only the
low-level default used when the trainer/evaluator classes are instantiated
directly. Normal runs through `main.py` use the A-E experiment names from the
registry.

Synthetic pothole experiments expect generated assets under:

```text
datasets/rdd2022/synthetic_potholes/
```

The source folder must contain:

```text
manifest.csv
```

with the same columns as `datasets/rdd2022/binary_pothole/train.csv`. Synthetic
rows are added only to the training split and capped by
`synthetic_pothole_ratio`, for example `0.2` means at most 20% of the real
pothole training count.

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
RDD_PATCH_DECISION_THRESHOLD = 0.5
```

`RDD_GRID_SIZE = 3` means every full image is split into `3 x 3 = 9` patches.

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

### One-Model Experiment Smoke Test

```python
RUN_RDD_FULL_IMAGE_PREPROCESSING = False
RUN_RDD_PATCH_PREPROCESSING = False
RDD_ACTIVE_EXPERIMENT_IDS = ("A",)
RDD_MODEL_MODE = "single"
RDD_SINGLE_MODEL = "mobilevit_xxs"
RDD_TRAINING_INPUT_MODE = "full_image"
RDD_EVALUATION_INPUT_MODE = "full_image"
RDD_TRAINING_EPOCHS = 10
RUN_RDD_TRAINING = True
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
```

Run:

```bash
uv run python -m src.rdd_benchmark.main
```

### Run A-D On One Tester Model

```python
RDD_ACTIVE_EXPERIMENT_IDS = ("A", "B", "C", "D")
RDD_MODEL_MODE = "single"
RDD_SINGLE_MODEL = "mobilevit_xxs"
RDD_TRAINING_INPUT_MODE = "full_image"
RDD_EVALUATION_INPUT_MODE = "full_image"
RUN_RDD_FULL_IMAGE_PREPROCESSING = False
RUN_RDD_PATCH_PREPROCESSING = False
```

### Run F-H Gentler Balancing Tests

```python
RDD_ACTIVE_EXPERIMENT_IDS = ("F", "G", "H")
RDD_MODEL_MODE = "single"
RDD_SINGLE_MODEL = "mobilevit_xxs"
RDD_TRAINING_INPUT_MODE = "full_image"
RDD_EVALUATION_INPUT_MODE = "full_image"
RUN_RDD_FULL_IMAGE_PREPROCESSING = False
RUN_RDD_PATCH_PREPROCESSING = False
```

### Run The Best Experiment On All Models

```python
RDD_ACTIVE_EXPERIMENT_IDS = ("C",)
RDD_MODEL_MODE = "all"
RUN_RDD_FULL_IMAGE_PREPROCESSING = False
RUN_RDD_PATCH_PREPROCESSING = False
RDD_TRAINING_INPUT_MODE = "full_image"
RDD_EVALUATION_INPUT_MODE = "full_image"
RUN_RDD_TRAINING = True
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
```

### Optional Patch-Grid Run

```python
RUN_RDD_PATCH_PREPROCESSING = False
RDD_ACTIVE_EXPERIMENT_IDS = ("A",)
RDD_MODEL_MODE = "all"
RDD_TRAINING_INPUT_MODE = "annotation_patch"
RDD_EVALUATION_INPUT_MODE = "grid_image"
RDD_GRID_SIZE = 3
RUN_RDD_TRAINING = True
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
```
