# Road-Damage TinyML Benchmark

Training and evaluation pipeline for lightweight road-damage image
classification, centered on binary pothole detection with RDD2022.

- preparing RDD2022 binary pothole datasets;
- running imbalance-aware data experiments;
- training custom tiny classifiers and fine-tuning pretrained classifiers;
- comparing models with accuracy, balanced accuracy, precision, recall, F1,
  ROC-AUC, confusion matrices, inference speed, and training curves;
- documenting model loading and demo inference paths.

## Setup

The project requires Python 3.10 or newer and uses the committed `uv.lock` for
reproducible environments. After installing `uv`, run from the repository root:

```bash
uv sync
```

Run commands through `uv run`, or use `.venv/bin/python` directly after the
environment has been created. Datasets, downloaded model weights, and generated
results are intentionally not included in the repository.

## What Is In The Repo

```text
.
├── datasets/                 # ignored dataset downloads and generated manifests
├── docs/                     # model notes, settings guide, inference demos
├── experiments/              # standalone performance/vision benchmark entry points
├── models/                   # ignored downloaded model weights, plus model notes
├── notebooks/                # RDD2022 exploration notebook
├── results/                  # ignored generated benchmark outputs
├── scripts/                  # download and demo helper scripts
├── src/                      # reusable benchmark implementation
│   ├── performance_benchmark/
│   ├── rdd_benchmark/
│   └── vision_benchmark/
└── tests/                    # unit and behavior tests
```

## RDD2022 Binary Pothole Benchmark

The main project code lives in:

```text
src/rdd_benchmark/
├── constants.py              # run switches and experiment/model selection
├── main.py                   # single entry point for preprocessing/training/eval
├── data_loader/              # manifest and dataset classes
├── data_preprocessing/       # splits, balancing, and augmentation
├── experiments/              # experiment registry and runner
└── training/                 # model adaptation, trainer, evaluation, comparison
```

All RDD behavior is controlled by editing:

```text
src/rdd_benchmark/constants.py
```

Then run:

```bash
.venv/bin/python -m src.rdd_benchmark.main
```

or, if `uv` is available:

```bash
uv run python -m src.rdd_benchmark.main
```

See [docs/rdd_benchmark_constants.md](docs/rdd_benchmark_constants.md) for the
full explanation of the switches.

The benchmark derives local train, validation, and test manifests from the
annotated RDD2022 training data. A positive image contains at least one `D40`
pothole annotation with a bounding-box area of at least 400 square pixels. The
task is full-image binary classification, not the official CRDDC
object-detection task.

## RDD Experiments

The benchmark supports reproducible experiment IDs:

- `A`: natural clean400 full-image baseline with standard augmentation;
- `F`: weighted sampler with 25 percent target pothole sampling and standard
  augmentation;
- `G`: majority downsampling to 1 pothole : 5 non-potholes with standard
  augmentation;
- `H`: majority downsampling to 1 pothole : 5 non-potholes with strong
  augmentation.

`B`, `C`, and `D` retain the earlier, more aggressive balancing strategies for
reproducibility. Select one or more IDs with `RDD_ACTIVE_EXPERIMENT_IDS`.
Every experiment uses the same class-weighted cross-entropy loss; experiment
IDs vary only the training-data sampling, downsampling, and augmentation.

The tiny/small model sweep currently uses:

- `tiny_cnn`
- `resnet8`
- `ds_cnn_small`
- `mobilenet_v1_025`
- `shufflenet_v2_x0_5`

Larger supported image classifiers include MobileViT, MobileNetV2/V3,
EfficientNet-B0, ResNet18, InceptionV3, and EfficientFormer.

## Outputs

Training and evaluation outputs are written under:

```text
results/rdd_trained_models/<experiment_name>/<model_name>/
```

Typical files include:

- `best.pt`
- `history.csv`
- `training_curves.png`
- `test_metrics.csv`
- `balanced_test_metrics.csv`
- `confusion_matrix.png`
- `balanced_confusion_matrix.png`
- `roc_curve.png`

The metrics CSV includes evaluation throughput and average image processing
time alongside the classification metrics.

Experiment-level comparisons are written as:

```text
results/rdd_trained_models/<experiment_name>/model_comparison.csv
```

## Dataset Downloads

Download COCO validation data for detector/performance demos:

```bash
./scripts/download_coco_val.sh
```

Download Imagenette validation data:

```bash
./scripts/download_imagenette_val.sh
```

Download all available RDD2022 country archives:

```bash
./scripts/download_rdd2022_subset.sh all
```

The raw image datasets are intentionally ignored by Git.

## Model Downloads

List supported model names:

```bash
./scripts/download_models.sh --list
```

Download selected pretrained models:

```bash
./scripts/download_models.sh mobilenet_v2 resnet18 shufflenet_v2_x0_5
```

Download all supported pretrained models:

```bash
./scripts/download_models.sh --all
```

Custom tiny classifiers such as `tiny_cnn`, `resnet8`, `ds_cnn_small`, and
`mobilenet_v1_025` are source-defined and do not download external weights.

## Demos

List available model demos:

```bash
./scripts/inference_demo.sh --list
```

Run a model-family demo:

```bash
./scripts/inference_demo.sh --model shufflenet
```

Run inference from a fine-tuned RDD checkpoint:

```bash
.venv/bin/python docs/models/rdd_binary_pothole/demo.py
```

Edit `CHECKPOINT_PATH` in that demo to point at any
`results/rdd_trained_models/.../best.pt` checkpoint.

## Tests

Run the fast test suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
```

With `uv`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

## Local SonarQube Analysis

If SonarQube is already set up:

```bash
docker start sonarqube
curl http://localhost:9000/api/system/status

mkdir -p reports

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg UV_CACHE_DIR=.uv-cache uv run pytest \
  -p pytest_cov \
  --cov=src \
  --cov=experiments \
  --cov-report=xml:coverage.xml \
  --junitxml=reports/pytest.xml

set -a
source .env
set +a

UV_CACHE_DIR=.uv-cache uv run pysonar
```

Open:

```text
http://localhost:9000/dashboard?id=road-damage-tinyml-benchmark
```

If the status endpoint fails right after startup, wait for SonarQube to finish
booting:

```bash
docker logs -f sonarqube
```
