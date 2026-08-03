# RDD Benchmark Settings

The benchmark has one entry point:

```bash
uv run python -m src.rdd_benchmark.main
```

If `uv` is unavailable, use the project environment directly:

```bash
.venv/bin/python -m src.rdd_benchmark.main
```

Edit `src/rdd_benchmark/constants.py` before running. The pipeline is always
full-image binary classification; settings only control preprocessing,
experiments, models, training, evaluation, and comparison.

## Typical Run

For a quick run of one model on the natural-data baseline:

```python
RUN_RDD_PREPROCESSING = False
RDD_ACTIVE_EXPERIMENT_IDS = ("A",)

RDD_MODEL_NAMES = ("mobilevit_xxs",)

RUN_RDD_TRAINING = True
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
```

The runner resumes interrupted work by default: an existing `best.pt` skips
training, and an existing `test_metrics.csv` skips evaluation.

## Dataset Preparation

```python
RUN_RDD_PREPROCESSING = False
RDD_SPLIT_MODE = "stratified_by_country"
```

Set `RUN_RDD_PREPROCESSING = True` only when the raw RDD data was downloaded or
the split settings changed. It regenerates:

```text
datasets/rdd2022/binary_pothole/train.csv
datasets/rdd2022/binary_pothole/validation.csv
datasets/rdd2022/binary_pothole/test.csv
```

An image is positive when its XML contains a `D40` pothole box of at least 400
pixels. The two split modes are:

- `stratified_by_country`: every country contributes to every split while
  preserving class proportions. This is the normal benchmark split.
- `country_holdout`: whole countries are assigned to train, validation, or
  test for a harder geographic generalization check.

Split fractions, country assignments, the random seed, and the 400-pixel
minimum live in `data_preprocessing/constants.py` because they are only used
during preprocessing.

## Experiments

```python
RDD_ACTIVE_EXPERIMENT_IDS = ("A", "G", "H")
```

Each experiment changes only the training data strategy. Validation and test
manifests remain unchanged.

| ID | Training strategy |
|---|---|
| `A` | Natural sampling and standard augmentation |
| `B` | 50% pothole weighted sampling and standard augmentation |
| `C` | 50% pothole weighted sampling and strong augmentation |
| `D` | 1:3 majority downsampling, 50% weighted sampling, and strong augmentation |
| `F` | 25% pothole weighted sampling and standard augmentation |
| `G` | 1:5 majority downsampling and standard augmentation |
| `H` | 1:5 majority downsampling and strong augmentation |

Downsampling changes only the generated training manifest under:

```text
datasets/rdd2022/binary_pothole_experiments/<experiment_name>/
```

The registry is `src/rdd_benchmark/experiments/experiment_registry.py`.

## Model Selection

Use one model for smoke tests:

```python
RDD_MODEL_NAMES = ("mobilevit_xxs",)
```

Add models to the same tuple for a full sweep:

```python
RDD_MODEL_NAMES = (
    "tiny_cnn",
    "resnet8",
    "ds_cnn_small",
    "mobilenet_v1_025",
    "shufflenet_v2_x0_5",
)
```

The supported names are listed beside `RDD_MODEL_NAMES` in the constants file.

## Training

```python
RUN_RDD_TRAINING = True
RDD_TRAINING_BATCH_SIZE = 32
RDD_TRAINING_EPOCHS = 30
RDD_TRAINING_LEARNING_RATE = 1e-4
RDD_TRAINING_WEIGHT_DECAY = 1e-4
RDD_TRAINING_BEST_METRIC = "balanced_accuracy"
RDD_TRAINING_EARLY_STOPPING_PATIENCE = 8
```

The trainer saves the checkpoint with the best validation metric and stops
when that metric has not improved for the configured patience. Every experiment
uses class-weighted cross-entropy to account for class imbalance; experiments
differ only in sampling, downsampling, and augmentation.

Keep `RDD_TRAINING_DROP_LAST_BATCH = True` for models with batch normalization.
It avoids a final one-sample training batch.

## Evaluation And Comparison

```python
RUN_RDD_EVALUATION = True
RUN_RDD_COMPARISON = True
RDD_COMPARISON_RANKING_METRIC = "f1"
```

Evaluation reports accuracy, balanced accuracy, precision, recall, F1,
ROC-AUC, confusion matrices, and end-to-end evaluation throughput on two views:

- the original imbalanced test split, which represents realistic prevalence;
- a deterministic balanced subset, which makes class-level behavior easier to
  compare.

Evaluation throughput includes image loading, preprocessing, transfer, and
batched model execution. It is not the batch-size-1 latency measured by the
system performance benchmark.

Results are written to:

```text
results/rdd_trained_models/<experiment_name>/<model_name>/
```

Each model folder contains the best checkpoint, training history and curves,
realistic and balanced-test metrics, confusion matrices, and the realistic-test
ROC curve. Evaluation timing is included in `test_metrics.csv`. The experiment
folder contains the complete ranked `model_comparison.csv`.

## Run Only One Stage

The three stage switches are independent. For example, to rebuild comparison
CSVs from existing evaluations:

```python
RUN_RDD_TRAINING = False
RUN_RDD_EVALUATION = False
RUN_RDD_COMPARISON = True
```

To force a rerun, set the relevant skip option to `False`:

```python
RDD_TRAINING_SKIP_EXISTING_CHECKPOINTS = False
RDD_EVALUATION_SKIP_EXISTING_RESULTS = False
```
