import csv
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from PIL import Image

import src.performance_benchmark.performance_benchmark as benchmark_module
from src.performance_benchmark.benchmark_result import BenchmarkResult
from src.performance_benchmark.metric_samples import MetricSamples
from src.performance_benchmark.performance_benchmark import PerformanceBenchmark
from src.performance_benchmark.utils import (
    append_result_csv,
    average_or_none,
    list_coco_images,
    load_rgb_image,
    move_inputs_to_device,
    percentile,
    resolve_device,
    synchronize_device,
)


class FakeAdapter:
    instances = []

    def __init__(self, model_name=None, model=None, device=None) -> None:
        self.calls = []
        FakeAdapter.instances.append(self)

    def infer(self, image_path: Path) -> None:
        self.calls.append(image_path)


class FakeSampler:
    def __init__(self, device=None) -> None:
        self.started = False
        self.stopped = False
        self.samples = MetricSamples(
            cpu_ram_mb=[100.0, 120.0],
            gpu_ram_mb=[10.0, 20.0],
            gpu_utilization_pct=[30.0, 50.0],
            power_w=[5.0, 7.0],
        )

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def make_result(model_name: str) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
        fps=10.0,
        avg_latency_ms=100.0,
        p95_latency_ms=120.0,
        avg_cpu_ram_mb=256.0,
        peak_cpu_ram_mb=300.0,
        avg_gpu_ram_mb=None,
        peak_gpu_ram_mb=None,
        avg_gpu_utilization_pct=None,
        avg_power_w=None,
        energy_per_inference_j=None,
        num_images=5,
    )


def test_utils_cover_files_devices_csv_images_and_statistics(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda:0")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        list_coco_images(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="No JPG images"):
        list_coco_images(tmp_path)

    (tmp_path / "b.jpg").write_bytes(b"fake")
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "ignore.png").write_bytes(b"fake")
    assert list_coco_images(tmp_path, num_images=1) == [tmp_path / "a.jpg"]
    with pytest.raises(ValueError, match="greater than zero"):
        list_coco_images(tmp_path, num_images=0)

    output_path = tmp_path / "nested" / "results.csv"
    first_result = make_result("model_a")
    second_result = make_result("model_b")
    append_result_csv(first_result, output_path)
    append_result_csv(second_result, output_path)
    with output_path.open(newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))
    assert [row["model_name"] for row in rows] == ["model_a", "model_b"]
    assert list(rows[0]) == list(asdict(first_result))

    image_path = tmp_path / "image.png"
    Image.new("L", (2, 2), color=128).save(image_path)
    assert load_rgb_image(image_path).mode == "RGB"

    class Movable:
        def to(self, device: torch.device) -> str:
            assert device == torch.device("cpu")
            return "moved"

    assert move_inputs_to_device(
        {"tensor": Movable(), "plain": 3}, torch.device("cpu")
    ) == {
        "tensor": "moved",
        "plain": 3,
    }

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    synchronize_device(torch.device("cpu"))
    synchronize_device(torch.device("cuda:0"))

    assert average_or_none([]) is None
    assert average_or_none([1.0, 3.0]) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([10.0, 20.0, 30.0], 95) == pytest.approx(29.0)


def test_performance_benchmark_runs_with_fakes(monkeypatch, capsys) -> None:
    FakeAdapter.instances.clear()
    image_paths = [Path(f"{index}.jpg") for index in range(120)]
    times = iter(float(index) for index in range(241))

    monkeypatch.setattr(benchmark_module, "ModelInferenceAdapter", FakeAdapter)
    monkeypatch.setattr(benchmark_module, "SystemMetricsSampler", FakeSampler)
    monkeypatch.setattr(benchmark_module, "synchronize_device", lambda device: None)
    monkeypatch.setattr(benchmark_module.time, "perf_counter", lambda: next(times))

    benchmark = PerformanceBenchmark(
        "fake_model",
        object(),
        image_paths,
        torch.device("cpu"),
        warmup_runs=5,
    )
    result = benchmark.run()

    assert len(FakeAdapter.instances[0].calls) == 125
    assert result.model_name == "fake_model"
    assert result.fps == pytest.approx(1.0)
    assert result.avg_latency_ms == pytest.approx(1000.0)
    assert result.p95_latency_ms == pytest.approx(1000.0)
    assert result.avg_cpu_ram_mb == pytest.approx(110.0)
    assert result.avg_gpu_ram_mb == pytest.approx(15.0)
    assert result.avg_gpu_utilization_pct == pytest.approx(40.0)
    assert result.energy_per_inference_j == pytest.approx(6.0)
    assert capsys.readouterr().out.splitlines() == [
        "Progress: 50/120 images",
        "Progress: 100/120 images",
        "Progress: 120/120 images",
    ]


def test_performance_benchmark_validation_and_cuda_peak(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_module, "ModelInferenceAdapter", FakeAdapter)
    with pytest.raises(ValueError, match="At least one image"):
        PerformanceBenchmark("model", object(), [], torch.device("cpu"))

    benchmark = PerformanceBenchmark.__new__(PerformanceBenchmark)
    benchmark.model_name = "cuda_model"
    benchmark.image_paths = [Path("a.jpg"), Path("b.jpg")]
    benchmark.device = torch.device("cuda:0")
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda device: 2 * 1024 * 1024
    )

    result = benchmark._build_result([0.1, 0.3], FakeSampler())

    assert result.fps == pytest.approx(5.0)
    assert result.avg_latency_ms == pytest.approx(200.0)
    assert result.p95_latency_ms == pytest.approx(290.0)
    assert result.peak_gpu_ram_mb == pytest.approx(2.0)
