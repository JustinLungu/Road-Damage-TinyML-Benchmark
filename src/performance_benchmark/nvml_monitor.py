from typing import Any

from src.performance_benchmark.constants import MILLIWATTS_PER_WATT


class NvmlMonitor:
    """Read utilization and power from NVIDIA's NVML API when available."""

    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        self.pynvml: Any | None = None
        self.handle: Any | None = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            # Hardware metrics are optional; failed NVML setup leaves them blank.
            self.pynvml = None
            self.handle = None

    def read(self) -> tuple[float | None, float | None]:
        if self.pynvml is None or self.handle is None:
            return None, None

        gpu_utilization = None
        power_w = None

        try:
            gpu_utilization = float(
                self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            )
        except Exception:
            # Keep partial metrics if utilization or power is unavailable.
            pass

        try:
            power_mw = float(self.pynvml.nvmlDeviceGetPowerUsage(self.handle))
            power_w = power_mw / MILLIWATTS_PER_WATT
        except Exception:
            # Keep partial metrics if utilization or power is unavailable.
            pass

        return gpu_utilization, power_w

    def stop(self) -> None:
        if self.pynvml is None:
            return

        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass
