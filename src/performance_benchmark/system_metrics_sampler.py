import os
import shutil
import threading

import psutil
import torch

from src.performance_benchmark.constants import (
    BYTES_PER_MB,
    DEFAULT_SAMPLE_INTERVAL_S,
    TEGRASTATS_COMMAND,
)
from src.performance_benchmark.metric_samples import MetricSamples
from src.performance_benchmark.nvml_monitor import NvmlMonitor
from src.performance_benchmark.tegrastats_monitor import TegrastatsMonitor


class SystemMetricsSampler:
    """Collect process and hardware samples while timed inference is running.

    The benchmark thread runs model inference. This sampler starts one small
    background thread that wakes up every interval_s seconds and records the
    latest RAM/GPU/power values.
    """

    def __init__(
        self,
        device: torch.device,
        interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    ) -> None:
        self.device = device
        self.interval_s = interval_s
        # Handle for reading RAM usage from this benchmark process.
        self.process = psutil.Process(os.getpid())
        # Lists where sampled RAM, GPU, and power values are accumulated.
        self.samples = MetricSamples()
        # Thread-safe signal used to tell the background sampler to stop.
        self.stop_event = threading.Event()
        # Background thread that periodically calls _sample() while inference runs.
        self.thread: threading.Thread | None = None
        # Backend used for GPU utilization and power readings.
        self.power_source: str | None = None
        self.hardware_monitor = self._create_hardware_monitor()

    def _create_hardware_monitor(
        self,
    ) -> NvmlMonitor | TegrastatsMonitor | None:
        if self.device.type != "cuda":
            return None

        # Prefer tegrastats on Jetson; otherwise fall back to NVML if available.
        if shutil.which(TEGRASTATS_COMMAND):
            # tegrastats expects milliseconds, while this class stores seconds.
            self.power_source = "tegrastats_system_input"
            return TegrastatsMonitor(interval_ms=max(1, int(self.interval_s * 1000)))

        # torch.device("cuda") has index None, which means CUDA device 0.
        device_index = self.device.index if self.device.index is not None else 0
        self.power_source = "nvml_gpu_board"
        return NvmlMonitor(device_index=device_index)

    def start(self) -> None:
        # Start the hardware monitor before sampling so utilization/power can be read.
        if self.hardware_monitor is not None:
            self.hardware_monitor.start()
        # Capture an initial point before the background sampling thread sleeps.
        self._sample()
        # daemon=True means this helper thread will not keep Python alive by itself.
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        # Wait returns True as soon as stop_event is set, which exits the loop.
        while not self.stop_event.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        # RSS is the current CPU RAM used by this Python benchmark process.
        cpu_ram_mb = self.process.memory_info().rss / BYTES_PER_MB
        self.samples.cpu_ram_mb.append(cpu_ram_mb)

        if self.device.type == "cuda":
            # PyTorch reports allocated tensor memory for this process/device.
            gpu_ram_mb = torch.cuda.memory_allocated(self.device) / BYTES_PER_MB
            self.samples.gpu_ram_mb.append(gpu_ram_mb)

        # Hardware monitors may return None when a metric is unavailable.
        if self.hardware_monitor is None:
            return

        gpu_utilization, power_w = self.hardware_monitor.read()
        if gpu_utilization is not None:
            self.samples.gpu_utilization_pct.append(gpu_utilization)
        if power_w is not None:
            self.samples.power_w.append(power_w)

    def stop(self) -> None:
        # Tell the background loop to exit, then wait for it to finish cleanly.
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        # Capture a final point after the timed loop completes.
        self._sample()
        if self.hardware_monitor is not None:
            self.hardware_monitor.stop()
