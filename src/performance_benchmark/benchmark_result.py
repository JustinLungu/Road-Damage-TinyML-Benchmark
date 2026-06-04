from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    model_name: str
    fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cpu_ram_mb: float
    peak_cpu_ram_mb: float
    avg_gpu_ram_mb: float | None
    peak_gpu_ram_mb: float | None
    avg_gpu_utilization_pct: float | None
    avg_power_w: float | None
    energy_per_inference_j: float | None
    num_images: int
