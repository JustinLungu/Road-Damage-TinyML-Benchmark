# System Metrics Results

This directory contains the output from the system performance experiment. The
main table is `system_performance_results.csv`, and the generated bar plots
visualize one metric at a time so models can be compared quickly.

## Files

- `system_performance_results.csv` stores one benchmark row per model run.
- `image_classification/` contains plots for MobileNet, MobileViT,
  EfficientFormer, EfficientNet, ResNet, ShuffleNet, and custom classifiers.
- `object_detection/` contains plots for YOLO object detectors.
- `vision_language/` contains plots for SmolVLM prompted forward passes.

The plot script uses the `task` recorded in each result row. It does not create
cross-task plots because SmolVLM prompted forward passes, classifier logits,
and YOLO detections are not equivalent workloads.

## How to Read the Plots

Each PNG is a bar plot. The x-axis is the model name, and the y-axis is the
metric value. Matching benchmark configurations replace their previous row.
Separate rows remain when hardware, precision, workload, or image count differs.

Before measuring, the benchmark runs five warmup passes that are not included
in the results. During the measured run, each image is timed with
`time.perf_counter()`. CUDA runs call `torch.cuda.synchronize()` before starting
and after finishing each inference so GPU work is included in the timing.

System metrics are sampled every 0.1 seconds while inference is running. CPU RAM
comes from the current Python process RSS through `psutil`. GPU memory comes
from PyTorch CUDA memory counters. GPU utilization and power come from
`tegrastats` on Jetson if it is available, otherwise from NVML on NVIDIA desktop
or laptop GPUs when available.

### `fps_bar.png`

Shows frames processed per second. Higher is better. This is the quickest view
of real-time throughput.

Calculated as:

```text
fps = number_of_images / sum(all_measured_inference_latencies_seconds)
```

### `avg_latency_ms_bar.png`

Shows the average time for one image inference in milliseconds. Lower is better.
This tells you how long the system usually waits for one prediction.

Calculated as:

```text
avg_latency_ms = mean(all_measured_inference_latencies_seconds) * 1000
```

### `p95_latency_ms_bar.png`

Shows the 95th percentile latency in milliseconds. Lower is better. This is more
conservative than average latency because it highlights slower tail cases.

Calculated by sorting all measured per-image latencies, taking the 95th
percentile with linear interpolation, then converting seconds to milliseconds.

### `avg_cpu_ram_mb_bar.png`

Shows average CPU RAM used by the benchmark process during inference. Lower is
better for memory-constrained edge devices.

Calculated as the mean of sampled process RSS values:

```text
avg_cpu_ram_mb = mean(psutil.Process(os.getpid()).memory_info().rss / 1024^2)
```

### `peak_cpu_ram_mb_bar.png`

Shows the highest CPU RAM observed during the run. Lower is better. This helps
catch short memory spikes that the average can hide.

Calculated as:

```text
peak_cpu_ram_mb = max(sampled_process_rss_mb)
```

### `avg_gpu_ram_mb_bar.png`

Shows average CUDA GPU memory allocated during inference. Lower is better when
GPU memory is limited.

Calculated as the mean of sampled PyTorch CUDA allocated memory:

```text
avg_gpu_ram_mb = mean(torch.cuda.memory_allocated(device) / 1024^2)
```

This field is blank for CPU runs.

### `peak_gpu_ram_mb_bar.png`

Shows the highest CUDA GPU memory allocated during the run. Lower is better.
This helps identify models that may fit on average but spike too high.

Calculated from PyTorch's CUDA peak memory counter after warmup:

```text
peak_gpu_ram_mb = torch.cuda.max_memory_allocated(device) / 1024^2
```

This field is blank for CPU runs.

### `avg_gpu_utilization_pct_bar.png`

Shows average GPU utilization percentage. This is diagnostic rather than simply
higher or lower is better. Very low GPU utilization can indicate a CPU or input
pipeline bottleneck. Very high utilization can indicate the model is saturating
the GPU.

Calculated as the mean of sampled GPU utilization percentages. On Jetson this
uses the `GR3D_FREQ` percentage reported by `tegrastats`. On other NVIDIA
systems this uses `nvmlDeviceGetUtilizationRates(...).gpu` through NVML.

This field is blank when GPU utilization cannot be read.

CPU runs do not initialize a GPU monitor, so unrelated GPU activity cannot
appear in their measurements.

### `avg_power_w_bar.png`

Shows average power draw in watts when power data is available. Lower is better
for battery-powered or thermally constrained systems.

Calculated as the mean of sampled power values in watts. On Jetson this uses
`POM_5V_IN` or `VDD_IN` from `tegrastats`, converted from milliwatts to watts.
On other NVIDIA systems this uses `nvmlDeviceGetPowerUsage`, also converted from
milliwatts to watts.

This field is blank when power cannot be read.

The `power_source` column identifies whether the value is discrete-GPU board
power from NVML or Jetson system-input power from tegrastats. These are not the
same measurement boundary.

### `energy_per_inference_j_bar.png`

Shows estimated energy used per inference in joules. Lower is better. This
combines power and latency, so a model that draws more power can still be more
efficient if it finishes much faster.

Calculated as:

```text
energy_per_inference_j = avg_power_w * avg_latency_seconds
```

This field is blank when average power is unavailable.

## Missing Values

Some hardware metrics may be blank if the machine cannot report them. For
example, Jetson-specific power readings require `tegrastats`, while desktop
NVIDIA systems use NVML when available.
