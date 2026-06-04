from dataclasses import dataclass, field


@dataclass
class MetricSamples:
    cpu_ram_mb: list[float] = field(default_factory=list)
    gpu_ram_mb: list[float] = field(default_factory=list)
    gpu_utilization_pct: list[float] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)
