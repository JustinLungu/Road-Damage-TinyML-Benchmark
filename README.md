# Road-Damage TinyML Benchmark

Benchmarking and fine-tuning pipeline for lightweight road-damage image
classification, centered on binary pothole detection with RDD2022.

This repository was originally started as a broader edge/VLM prototype. It is
now scoped as a self-contained benchmark project for:

- preparing RDD2022 binary pothole datasets;
- running imbalance-aware data experiments;
- fine-tuning tiny, small, and standard image classifiers;
- comparing models with accuracy, balanced accuracy, precision, recall, F1,
  ROC-AUC, confusion matrices, inference speed, and training curves;
- documenting model loading and demo inference paths.

## What Is In The Repo

```text
.
├── datasets/                 # ignored dataset downloads and generated manifests
├── docs/                     # model notes, RDD constants guide, demos
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

Removed from the old roadmap:

- `configs/`: no active config files; RDD settings live in
  `src/rdd_benchmark/constants.py`.
- `jetson/`: Jetson deployment belongs in the next project, not this benchmark
  repo.

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

## RDD Experiments

The benchmark supports reproducible experiment IDs:

- `A`: natural clean400 full-image baseline with standard augmentation;
- `F`: weighted sampler with 25 percent target pothole sampling and standard
  augmentation;
- `G`: majority downsampling to 1 pothole : 5 non-potholes with standard
  augmentation;
- `H`: majority downsampling to 1 pothole : 5 non-potholes with strong
  augmentation applied to every training image.

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
- `loss_curve.png`
- `f1_curve.png`
- `accuracy_curve.png`
- `test_metrics.json`
- `test_metrics.csv`
- `balanced_test_metrics.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `roc_curve.png`
- `evaluation_timing.json`

Experiment-level comparisons are written as:

```text
results/rdd_trained_models/<experiment_name>/model_comparison.csv
results/rdd_trained_models/<experiment_name>/top_models.csv
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
