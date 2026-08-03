# Source Code

This directory contains the reusable implementation for model loading,
benchmarking, and RDD2022 binary pothole training and evaluation.

## Packages

- `rdd_benchmark/` contains the RDD2022 data preprocessing, dataset loading,
  experiment registry, training, evaluation, and model-comparison pipeline.
- `vision_benchmark/` contains model-quality benchmarks for object detection
  and image classification datasets.
- `performance_benchmark/` contains latency, FPS, memory, utilization, power,
  and energy sampling utilities.
- `custom_models.py` defines the tiny/source-defined image classifiers used in
  the RDD benchmark.
- `load_model.py` is the shared model registry and loader.

## Shared Constants

`constants.py` contains repo-wide constants that are shared across source
modules and experiment scripts, including project paths, dataset/result paths,
model storage directories, model IDs, and checkpoint/cache locations.

Package-specific constants should stay close to their package. For example,
`src/performance_benchmark/constants.py` keeps benchmark defaults such as warmup
runs, progress intervals, sampling intervals, and parser patterns.
