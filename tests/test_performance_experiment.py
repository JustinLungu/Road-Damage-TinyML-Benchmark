import argparse
import subprocess
import sys

import pytest
import torch

import experiments.performance.system_performance as experiment
from src.performance_benchmark.benchmark_result import PerformanceBenchmarkResult


def make_result(model_name: str) -> PerformanceBenchmarkResult:
    return PerformanceBenchmarkResult(
        model_name=model_name,
        task="object_detection",
        workload="single_image_detection",
        device="cpu",
        device_name="CPU",
        precision="float32",
        batch_size=1,
        timing_scope="image_load_preprocess_inference",
        warmup_runs=5,
        power_source=None,
        fps=1.0,
        avg_latency_ms=2.0,
        p95_latency_ms=3.0,
        avg_cpu_ram_mb=4.0,
        peak_cpu_ram_mb=5.0,
        avg_gpu_ram_mb=None,
        peak_gpu_ram_mb=None,
        avg_gpu_utilization_pct=None,
        avg_power_w=None,
        energy_per_inference_j=None,
        num_images=6,
    )


def test_resolves_explicit_and_downloaded_models(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    assert experiment.resolve_model_names(
        ["yolov5nu", "yolov5nu", "yolov8n"], parser
    ) == ["yolov5nu", "yolov8n"]

    monkeypatch.setattr(
        experiment,
        "get_downloaded_model_names",
        lambda: ["yolov5nu", "smolvlm_2b", "mobilenet_v3_small"],
    )
    assert experiment.resolve_model_names([experiment.ALL_LOADED], parser) == [
        "yolov5nu",
        "mobilenet_v3_small",
    ]

    with pytest.raises(SystemExit):
        experiment.resolve_model_names([experiment.ALL_LOADED, "yolov5nu"], parser)


def test_runs_models_in_isolated_processes(monkeypatch) -> None:
    commands = []

    def fake_run(command, check):
        commands.append(command)
        return_code = int(command[command.index("--model") + 1] == "bad")
        return subprocess.CompletedProcess(command, return_code)

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    experiment.run_model_processes(["good"], "cpu", num_images=10)
    assert commands[0][-2:] == ["--num-images", "10"]

    with pytest.raises(RuntimeError, match="bad"):
        experiment.run_model_processes(["good", "bad"], "cpu", None)


def test_runs_and_saves_one_model(monkeypatch, tmp_path) -> None:
    loaded_model = object()
    saved = []

    class FakeBenchmark:
        def __init__(self, model_name, model, image_paths, device) -> None:
            assert (model_name, model, image_paths, device) == (
                "yolov5nu",
                loaded_model,
                [tmp_path / "image.jpg"],
                torch.device("cpu"),
            )

        def run(self):
            return make_result("yolov5nu")

    monkeypatch.setattr(experiment, "load_model", lambda name: loaded_model)
    monkeypatch.setattr(experiment, "PerformanceBenchmark", FakeBenchmark)
    monkeypatch.setattr(
        experiment,
        "save_result_csv",
        lambda result, path, keys: saved.append((result, path, keys)),
    )

    experiment.run_model_benchmark(
        "yolov5nu", [tmp_path / "image.jpg"], torch.device("cpu")
    )
    assert saved == [
        (
            make_result("yolov5nu"),
            experiment.RESULTS_CSV,
            experiment.PERFORMANCE_RESULT_KEYS,
        )
    ]


def test_overwrite_clears_results_before_multi_model_run(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "results.csv"
    output_path.write_text("old", encoding="utf-8")
    calls = []
    monkeypatch.setattr(experiment, "RESULTS_CSV", output_path)
    monkeypatch.setattr(experiment, "resolve_device", lambda name: torch.device(name))
    monkeypatch.setattr(
        experiment,
        "run_model_processes",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "system_performance.py",
            "--model",
            "yolov5nu",
            "yolov8n",
            "--device",
            "cpu",
            "-o",
        ],
    )

    experiment.main()
    assert not output_path.exists()
    assert calls == [(["yolov5nu", "yolov8n"], "cpu", None)]
