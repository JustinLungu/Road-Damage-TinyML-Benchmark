# Performance Benchmark

This package contains the reusable code for measuring system-level inference
performance on COCO validation images. It measures FPS, latency, memory usage,
GPU utilization, power draw, and energy per inference without calculating model
accuracy.

## Files

- `performance_benchmark.py` contains the main `PerformanceBenchmark` class
  that runs warmup passes, times inference, and creates the final result.
- `model_inference_adapter.py` normalizes YOLO, image-classifier, and SmolVLM
  calls. Classifier preprocessing is shared with the vision benchmark through
  `src/image_classification_inference_adapter.py`.
- `system_metrics_sampler.py` samples process RAM, CUDA memory, GPU utilization,
  and power while inference is running.
- `nvml_monitor.py` reads GPU utilization and GPU board power through NVML on
  supported NVIDIA systems.
- `tegrastats_monitor.py` reads GPU utilization and input-rail system power from
  `tegrastats` on Jetson devices.
- `benchmark_result.py` defines the result fields written to the CSV file.
- `constants.py` contains the prompt, sampling defaults, timing settings, and
  tegrastats parser patterns.
- `utils.py` contains shared helpers for devices, COCO image paths, image
  loading, and statistics.

The experiment entry point is
`experiments/performance/system_performance.py`, and its results are written to
`results/system_metrics/system_performance_results.csv`.

## Pipeline

1. The experiment runner reads the selected model names, device, optional image
   limit, and optional results overwrite flag from the command line. If multiple
   models are selected, it starts a fresh Python process for each model.
2. `src/load_model.py` loads the requested pretrained model, and the COCO
   validation image paths are collected.
3. `ModelInferenceAdapter` prepares the correct preprocessing and inference call
   for the selected model family. SmolVLM uses one forward pass with the fixed
   prompt `Describe the image briefly.` and does not generate text.
4. `PerformanceBenchmark` runs five warmup passes that are excluded from the
   results. The first model calls can be slower because libraries may initialize
   execution backends, allocate memory, build internal caches, or select
   optimized kernels. Warmup passes reduce the effect of this one-time setup so
   the reported latency and FPS better represent steady-state inference.
5. `SystemMetricsSampler` starts collecting available RAM, GPU utilization, and
   power samples in the background.
6. Each COCO image is loaded, preprocessed, passed through the model, and timed.
   CUDA execution is synchronized before and after inference so GPU latency is
   measured correctly.
7. The benchmark calculates FPS, average latency, p95 latency, memory values,
   average utilization, average power, and energy per inference.
8. The experiment runner saves one result row to
   `results/system_metrics/system_performance_results.csv`. A matching model,
   workload, device, precision, and image-count configuration replaces its
   previous row. Hardware metrics that are not available are left blank.

Each row records the task, exact workload, device and device name, model
precision, batch size, timing scope, warmup count, and power source. SmolVLM is
explicitly labeled `prompted_forward_no_generation`; it is not end-to-end text
generation latency.

CPU runs do not initialize NVML or tegrastats and leave GPU metrics blank. CUDA
power uses either `nvml_gpu_board` on desktop NVIDIA GPUs or
`tegrastats_system_input` on Jetson. These sources measure different power
boundaries and should not be compared as if they were identical.
