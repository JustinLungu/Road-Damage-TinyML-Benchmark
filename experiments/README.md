# Experiments

This directory will contain reproducible experiment configurations, logs, metrics, and notes.

## System performance benchmark

Run one loaded model over the COCO validation images:

```bash
uv run python experiments/system_performance.py --model yolov5nu --device cuda:0
```

Run multiple models by listing their names after `--model`:

```bash
uv run python experiments/system_performance.py \
  --model yolov5nu yolov8n mobilenet_v3_small \
  --device cuda:0
```

Run every model with a checkpoint already downloaded under `models/`:

```bash
uv run python experiments/system_performance.py --model all-loaded --device cuda:0
```

The `all-loaded` option checks for model weight files under `models/`. It does
not select empty cache directories or checkpoints that only exist in a global
library cache.

The benchmark uses all validation images by default. Use `--num-images` for a
smaller run. The same limit is applied to every selected model:

```bash
uv run python experiments/system_performance.py --model yolov5nu --device cpu --num-images 100
```

During inference, the benchmark prints progress every 50 completed images and
once more when the final image is complete.

Results are appended to
`results/system_metrics/system_performance_results.csv`. Each run records FPS,
average and p95 latency, CPU RAM, GPU RAM, GPU utilization, power, energy per
inference, and the number of images.

Use `-o` to overwrite the existing results CSV with only the rows from the
current run:

```bash
uv run python experiments/system_performance.py \
  --model yolov5nu yolov8n mobilenet_v3_small \
  --device cuda:0 \
  -o
```

When multiple models are selected, each model runs in a fresh Python process so
memory left behind by an earlier model does not affect later RAM measurements.

The timed operation starts with a COCO JPEG file path and ends when model
inference completes. Model loading and five warmup runs are excluded. SmolVLM
uses one forward pass with the fixed prompt `Describe the image briefly.` and
does not generate text.

CPU RAM is the benchmark process RSS. GPU RAM is PyTorch CUDA allocated memory.
On Jetson, GPU utilization and input-rail system power are read from
`tegrastats`. On other NVIDIA systems, GPU utilization and GPU board power are
read through NVML. Any unavailable hardware metric is left blank in the CSV.
