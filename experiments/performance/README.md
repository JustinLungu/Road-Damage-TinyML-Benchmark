# System Performance Experiment

This experiment benchmarks system-level inference performance on COCO
validation images. It records FPS, latency, CPU RAM, GPU RAM, GPU utilization,
power, energy per inference, and the number of images. It does not calculate
model accuracy.

## Run Benchmarks

Run one loaded model:

```bash
uv run python experiments/performance/system_performance.py --model yolov5nu --device cuda:0
```

Run multiple models by listing their names after `--model`:

```bash
uv run python experiments/performance/system_performance.py \
  --model yolov5nu yolov8n mobilenet_v3_small \
  --device cuda:0
```

Run every model with a checkpoint already downloaded under `models/`:

```bash
uv run python experiments/performance/system_performance.py --model all-loaded --device cuda:0
```

The `all-loaded` option checks for model weight files under `models/`. It does
not select empty cache directories or checkpoints that only exist in a global
library cache. It also skips models listed in `ALL_LOADED_EXCLUDED_MODELS`,
currently `smolvlm_2b`, because that checkpoint can exceed common local GPU
memory. You can still benchmark it explicitly with `--model smolvlm_2b`.

The benchmark uses all validation images by default. Use `--num-images` for a
smaller run. The same limit is applied to every selected model:

```bash
uv run python experiments/performance/system_performance.py --model yolov5nu --device cpu --num-images 100
```

During inference, the benchmark prints progress every 50 completed images and
once more when the final image is complete.

Results are appended to
`results/system_metrics/system_performance_results.csv`.

Use `-o` to overwrite the existing results CSV with only the rows from the
current run:

```bash
uv run python experiments/performance/system_performance.py \
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

## Plot Results

Create one bar plot per metric from the system performance CSV:

```bash
uv run python experiments/performance/plot_system_performance.py
```

By default, the script reads
`results/system_metrics/system_performance_results.csv` and saves PNG plots in
two folders next to that CSV:

- `results/system_metrics/with_vlms/` contains plots for all models.
- `results/system_metrics/without_vlms/` contains plots after filtering out VLM
  model rows such as SmolVLM.

## Files

- `system_performance.py` runs the benchmark and writes the CSV row.
- `plot_system_performance.py` reads the CSV and creates comparison plots.
- `constants.py` contains experiment-local constants such as the `all-loaded`
  CLI token and plot output folder names.
