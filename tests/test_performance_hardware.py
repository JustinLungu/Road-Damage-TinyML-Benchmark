import sys
from types import ModuleType, SimpleNamespace

import torch

import src.performance_benchmark.system_metrics_sampler as sampler_module
import src.performance_benchmark.tegrastats_monitor as tegrastats_module
from src.performance_benchmark.constants import BYTES_PER_MB
from src.performance_benchmark.nvml_monitor import NvmlMonitor
from src.performance_benchmark.system_metrics_sampler import SystemMetricsSampler
from src.performance_benchmark.tegrastats_monitor import TegrastatsMonitor


class FakeHardwareMonitor:
    def __init__(self, *args, **kwargs) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self) -> tuple[float, float]:
        return 25.0, 6.5

    def stop(self) -> None:
        self.stopped = True


class FakeProcess:
    def memory_info(self):
        return SimpleNamespace(rss=2 * BYTES_PER_MB)


def test_nvml_and_tegrastats_monitors_handle_success_and_failure(monkeypatch) -> None:
    monitor = NvmlMonitor(device_index=0)
    assert monitor.read() == (None, None)

    fake_pynvml = ModuleType("pynvml")
    fake_pynvml.nvmlInit = lambda: None
    fake_pynvml.nvmlDeviceGetHandleByIndex = lambda device_index: "handle"
    fake_pynvml.nvmlDeviceGetUtilizationRates = lambda handle: SimpleNamespace(gpu=42)
    fake_pynvml.nvmlDeviceGetPowerUsage = lambda handle: 12_500
    fake_pynvml.nvmlShutdown = lambda: None
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    monitor.start()
    assert monitor.read() == (42.0, 12.5)
    monitor.stop()

    class BrokenPynvml:
        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            raise RuntimeError("no utilization")

        @staticmethod
        def nvmlDeviceGetPowerUsage(handle):
            raise RuntimeError("no power")

    monitor.pynvml = BrokenPynvml()
    monitor.handle = "handle"
    assert monitor.read() == (None, None)

    tegra_monitor = TegrastatsMonitor(interval_ms=100)
    tegra_monitor.process = SimpleNamespace(
        stdout=[
            "RAM 1000/4000MB GR3D_FREQ 12% POM_5V_IN 2500mW/3000mW\n",
            "RAM 1000/4000MB GR3D_FREQ 88% VDD_IN 4200mW/4500mW\n",
        ]
    )
    tegra_monitor._read_output()
    assert tegra_monitor.read() == (88.0, 4.2)

    tegra_monitor = TegrastatsMonitor(interval_ms=100)
    tegra_monitor._read_output()
    assert tegra_monitor.read() == (None, None)

    monkeypatch.setattr(
        tegrastats_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    tegra_monitor.start()
    assert tegra_monitor.process is None


def test_system_metrics_sampler_selects_and_records_metrics(monkeypatch) -> None:
    monkeypatch.setattr(sampler_module.shutil, "which", lambda command: None)
    monkeypatch.setattr(sampler_module, "NvmlMonitor", FakeHardwareMonitor)
    sampler = SystemMetricsSampler(torch.device("cpu"))
    sampler.process = FakeProcess()
    assert sampler.hardware_monitor is None
    assert sampler.power_source is None

    sampler._sample()
    assert sampler.samples.cpu_ram_mb == [2.0]
    assert sampler.samples.gpu_utilization_pct == []
    assert sampler.samples.power_w == []
    assert sampler.samples.gpu_ram_mb == []

    sampler.start()
    sampler.stop()

    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    cuda_sampler = SystemMetricsSampler(torch.device("cuda:0"))
    cuda_sampler.process = FakeProcess()
    assert isinstance(cuda_sampler.hardware_monitor, FakeHardwareMonitor)
    assert cuda_sampler.power_source == "nvml_gpu_board"
    cuda_sampler._sample()
    assert cuda_sampler.samples.gpu_utilization_pct == [25.0]
    assert cuda_sampler.samples.power_w == [6.5]

    created_intervals = []

    class FakeTegrastatsMonitor(FakeHardwareMonitor):
        def __init__(self, interval_ms: int) -> None:
            created_intervals.append(interval_ms)
            super().__init__()

    monkeypatch.setattr(sampler_module.shutil, "which", lambda command: "/bin/tool")
    monkeypatch.setattr(sampler_module, "TegrastatsMonitor", FakeTegrastatsMonitor)

    first_tegra_sampler = SystemMetricsSampler(torch.device("cuda:0"), interval_s=0.25)
    SystemMetricsSampler(torch.device("cuda:0"), interval_s=0.0001)

    assert created_intervals == [250, 1]
    assert first_tegra_sampler.power_source == "tegrastats_system_input"
