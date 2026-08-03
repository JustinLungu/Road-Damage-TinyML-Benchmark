import os
import shutil
import threading
from dataclasses import dataclass, field

import psutil
import torch

from src.performance_benchmark.constants import (
    BYTES_PER_MB,
    DEFAULT_SAMPLE_INTERVAL_S,
    TEGRASTATS_COMMAND,
)
from src.performance_benchmark.nvml_monitor import NvmlMonitor
from src.performance_benchmark.tegrastats_monitor import TegrastatsMonitor


@dataclass
class MetricSamples:
    cpu_ram_mb: list[float] = field(default_factory=list)
    gpu_ram_mb: list[float] = field(default_factory=list)
    gpu_utilization_pct: list[float] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)


class SystemMetricsSampler:
    """Collect process and hardware samples during timed inference."""

    def __init__(
        self,
        device: torch.device,
        interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    ) -> None:
        self.device = device
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.samples = MetricSamples()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.power_source: str | None = None
        self.hardware_monitor = self._create_hardware_monitor()

    def _create_hardware_monitor(
        self,
    ) -> NvmlMonitor | TegrastatsMonitor | None:
        if self.device.type != "cuda":
            return None

        if shutil.which(TEGRASTATS_COMMAND):
            self.power_source = "tegrastats_system_input"
            return TegrastatsMonitor(interval_ms=max(1, int(self.interval_s * 1000)))

        device_index = self.device.index if self.device.index is not None else 0
        self.power_source = "nvml_gpu_board"
        return NvmlMonitor(device_index=device_index)

    def start(self) -> None:
        if self.hardware_monitor is not None:
            self.hardware_monitor.start()
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        cpu_ram_mb = self.process.memory_info().rss / BYTES_PER_MB
        self.samples.cpu_ram_mb.append(cpu_ram_mb)

        if self.device.type == "cuda":
            gpu_ram_mb = torch.cuda.memory_allocated(self.device) / BYTES_PER_MB
            self.samples.gpu_ram_mb.append(gpu_ram_mb)

        if self.hardware_monitor is None:
            return

        gpu_utilization, power_w = self.hardware_monitor.read()
        if gpu_utilization is not None:
            self.samples.gpu_utilization_pct.append(gpu_utilization)
        if power_w is not None:
            self.samples.power_w.append(power_w)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        if self.hardware_monitor is not None:
            self.hardware_monitor.stop()
