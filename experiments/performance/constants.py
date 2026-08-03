from pathlib import Path

from src.constants import SYSTEM_PERFORMANCE_RESULTS_CSV


ALL_LOADED = "all-loaded"
ALL_LOADED_EXCLUDED_MODELS = {"smolvlm_2b"}

RESULTS_CSV: Path = SYSTEM_PERFORMANCE_RESULTS_CSV

PERFORMANCE_RESULT_KEYS = (
    "model_name",
    "workload",
    "device",
    "device_name",
    "precision",
    "batch_size",
    "timing_scope",
    "warmup_runs",
    "num_images",
)

PERFORMANCE_TASKS = (
    "image_classification",
    "object_detection",
    "vision_language",
)
PERFORMANCE_METRICS = (
    "fps",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_cpu_ram_mb",
    "peak_cpu_ram_mb",
    "avg_gpu_ram_mb",
    "peak_gpu_ram_mb",
    "avg_gpu_utilization_pct",
    "avg_power_w",
    "energy_per_inference_j",
)
