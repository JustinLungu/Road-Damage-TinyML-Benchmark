# Experiments

This directory contains reproducible experiment entry points, notes, and
experiment-specific documentation.

## Available Experiments

- `performance/` contains the system performance benchmark and plotting scripts
  for comparing loaded models on COCO validation images.
- `detection/` contains prediction-quality benchmarks for COCO object detection
  and manifest-labeled image classification.

## Quick Start

Run the system performance benchmark for one model:

```bash
uv run python experiments/performance/system_performance.py --model yolov5nu --device cuda:0
```

Run multiple models:

```bash
uv run python experiments/performance/system_performance.py \
  --model yolov5nu yolov8n mobilenet_v3_small \
  --device cuda:0
```

Run every checkpoint already downloaded under `models/`:

```bash
uv run python experiments/performance/system_performance.py --model all-loaded --device cuda:0
```

`all-loaded` skips models that are known to be too large for common local GPU
runs, currently `smolvlm_2b`. Explicit model selection still works.

Use `--num-images` for a smaller test run and `-o` to overwrite the existing
CSV before writing the new results:

```bash
uv run python experiments/performance/system_performance.py \
  --model yolov5nu \
  --device cuda:0 \
  --num-images 100 \
  -o
```

After the benchmark writes
`results/system_metrics/system_performance_results.csv`, create the comparison
plots with:

```bash
uv run python experiments/performance/plot_system_performance.py
```

Plots are separated into `image_classification/`, `object_detection/`,
`semantic_interpretation/`, and `all_models/` under
`results/system_metrics/`.

See `performance/README.md` for the full benchmark details.

See `detection/README.md` for detection and classification metric definitions.
