# Source Code

This directory will contain the main implementation for inference, detection, VLM integration, and supporting utilities.

## Shared Constants

`constants.py` contains repo-wide constants that are shared across source
modules and experiment scripts, including project paths, dataset/result paths,
model storage directories, model IDs, and checkpoint/cache locations.

Package-specific constants should stay close to their package. For example,
`src/performance_benchmark/constants.py` keeps benchmark defaults such as warmup
runs, progress intervals, sampling intervals, and parser patterns.
