import time
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from src.constants import MODEL_TASK_BY_NAME, MODEL_WORKLOAD_BY_NAME
from src.performance_benchmark.benchmark_result import PerformanceBenchmarkResult
from src.performance_benchmark.constants import (
    BYTES_PER_MB,
    DEFAULT_WARMUP_RUNS,
    INFERENCE_BATCH_SIZE,
    PROGRESS_INTERVAL_IMAGES,
    TIMING_SCOPE,
)
from src.performance_benchmark.model_inference_adapter import ModelInferenceAdapter
from src.performance_benchmark.system_metrics_sampler import SystemMetricsSampler
from src.performance_benchmark.utils import (
    average_or_none,
    describe_device,
    model_precision,
    percentile,
    synchronize_device,
)


class PerformanceBenchmark:
    """Run one timed system-performance benchmark for one loaded model."""

    def __init__(
        self,
        model_name: str,
        model: Any,
        image_paths: list[Path],
        device: torch.device,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
    ) -> None:
        if not image_paths:
            raise ValueError("At least one image is required for the benchmark.")

        self.model_name = model_name
        self.image_paths = image_paths
        self.device = device
        self.warmup_runs = warmup_runs
        self.task = MODEL_TASK_BY_NAME.get(model_name, "unknown")
        self.workload = MODEL_WORKLOAD_BY_NAME.get(model_name, "unknown")
        self.device_name = describe_device(device)
        self.precision = model_precision(model)
        self.adapter = ModelInferenceAdapter(model_name, model, device)

    def run(self) -> PerformanceBenchmarkResult:
        self._run_warmup()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        sampler = SystemMetricsSampler(self.device)
        latencies_s = self._measure_latencies(sampler)
        return self._build_result(latencies_s, sampler)

    def _run_warmup(self) -> None:
        with torch.inference_mode():
            for _ in range(self.warmup_runs):
                self.adapter.infer(self.image_paths[0])
                synchronize_device(self.device)

    def _measure_latencies(self, sampler: SystemMetricsSampler) -> list[float]:
        latencies_s: list[float] = []

        sampler.start()
        try:
            with torch.inference_mode():
                for image_number, image_path in enumerate(self.image_paths, start=1):
                    synchronize_device(self.device)
                    start_time = time.perf_counter()
                    self.adapter.infer(image_path)
                    synchronize_device(self.device)
                    latencies_s.append(time.perf_counter() - start_time)

                    if (
                        image_number % PROGRESS_INTERVAL_IMAGES == 0
                        or image_number == len(self.image_paths)
                    ):
                        print(
                            f"Progress: {image_number}/{len(self.image_paths)} images",
                            flush=True,
                        )
        finally:
            sampler.stop()

        return latencies_s

    def _build_result(
        self,
        latencies_s: list[float],
        sampler: SystemMetricsSampler,
    ) -> PerformanceBenchmarkResult:
        total_inference_time_s = sum(latencies_s)
        avg_latency_s = fmean(latencies_s)
        avg_power_w = average_or_none(sampler.samples.power_w)

        if self.device.type == "cuda":
            peak_gpu_ram_mb = (
                torch.cuda.max_memory_allocated(self.device) / BYTES_PER_MB
            )
        else:
            peak_gpu_ram_mb = None

        return PerformanceBenchmarkResult(
            model_name=self.model_name,
            task=self.task,
            workload=self.workload,
            device=str(self.device),
            device_name=self.device_name,
            precision=self.precision,
            batch_size=INFERENCE_BATCH_SIZE,
            timing_scope=TIMING_SCOPE,
            warmup_runs=self.warmup_runs,
            power_source=sampler.power_source,
            fps=len(self.image_paths) / total_inference_time_s,
            avg_latency_ms=avg_latency_s * 1000,
            p95_latency_ms=percentile(latencies_s, 95) * 1000,
            avg_cpu_ram_mb=fmean(sampler.samples.cpu_ram_mb),
            peak_cpu_ram_mb=max(sampler.samples.cpu_ram_mb),
            avg_gpu_ram_mb=average_or_none(sampler.samples.gpu_ram_mb),
            peak_gpu_ram_mb=peak_gpu_ram_mb,
            avg_gpu_utilization_pct=average_or_none(
                sampler.samples.gpu_utilization_pct
            ),
            avg_power_w=avg_power_w,
            energy_per_inference_j=(
                avg_power_w * avg_latency_s if avg_power_w is not None else None
            ),
            num_images=len(self.image_paths),
        )
