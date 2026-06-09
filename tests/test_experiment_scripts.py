import argparse
import subprocess
import sys

import pandas as pd
import pytest
import torch

import experiments.performance.plot_system_performance as plot_system_performance
import experiments.performance.system_performance as system_performance
from src.performance_benchmark.benchmark_result import BenchmarkResult


def make_result(model_name: str) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model_name,
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


def test_plot_system_performance_creates_grouped_plots(monkeypatch, tmp_path, capsys):
    csv_path = tmp_path / "system_performance_results.csv"
    pd.DataFrame(
        {
            "model_name": ["yolov5nu", "smolvlm_256m", "yolov5nu"],
            "fps": [100.0, 2.0, 110.0],
            "avg_power_w": [20.0, 50.0, 19.0],
            "not_numeric": ["a", "b", "c"],
        }
    ).to_csv(csv_path, index=False)
    plotted = []
    real_plot_metric = plot_system_performance.plot_metric

    def fake_plot_metric(plot_data, metric_column, output_path) -> None:
        plotted.append((plot_data.copy(), metric_column, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")

    monkeypatch.setattr(plot_system_performance, "plot_metric", fake_plot_metric)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_system_performance.py", "--csv", str(csv_path)],
    )

    plot_system_performance.main()
    created_lines = capsys.readouterr().out.splitlines()

    assert "Created 4 plots:" in created_lines
    assert plotted[0][0]["model_name"].tolist() == [
        "yolov5nu #1",
        "smolvlm_256m",
        "yolov5nu #2",
    ]
    assert plotted[2][0]["model_name"].tolist() == ["yolov5nu #1", "yolov5nu #2"]
    assert plot_system_performance.is_vlm_model("SmolVLM_256M") is True
    assert plot_system_performance.is_vlm_model("yolov5nu") is False
    assert plot_system_performance.format_metric_name("avg_latency_ms") == (
        "Avg Latency Ms"
    )
    assert plot_system_performance.safe_filename("Avg Power (W)") == "avg_power_w"

    real_plot_path = tmp_path / "real" / "fps_bar.png"
    real_plot_metric(
        pd.DataFrame({"model_name": ["a", "b"], "fps": [1.0, 2.0]}),
        "fps",
        real_plot_path,
    )
    assert real_plot_path.is_file()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        plot_system_performance.create_metric_plots(tmp_path / "missing.csv")

    missing_model_name_csv = tmp_path / "missing_model_name.csv"
    pd.DataFrame({"fps": [1.0]}).to_csv(missing_model_name_csv, index=False)
    with pytest.raises(ValueError, match="model_name"):
        plot_system_performance.create_metric_plots(missing_model_name_csv)

    empty_group = plot_system_performance.create_metric_plots_for_group(
        results=pd.DataFrame(columns=["model_name", "fps"]),
        metric_columns=["fps"],
        output_dir=tmp_path,
        group_name="without_vlms",
    )
    assert empty_group == []


def test_system_performance_runner_flow(monkeypatch, tmp_path) -> None:
    parser = argparse.ArgumentParser()
    assert system_performance.resolve_model_names(
        ["yolov5nu", "yolov5nu", "yolov8n"],
        parser,
    ) == ["yolov5nu", "yolov8n"]

    monkeypatch.setattr(
        system_performance,
        "get_downloaded_model_names",
        lambda: ["yolov5nu", "smolvlm_2b", "mobilenet_v3_small"],
    )
    assert system_performance.resolve_model_names(
        [system_performance.ALL_LOADED],
        parser,
    ) == ["yolov5nu", "mobilenet_v3_small"]

    with pytest.raises(SystemExit):
        system_performance.resolve_model_names(
            [system_performance.ALL_LOADED, "yolov5nu"],
            parser,
        )
    monkeypatch.setattr(
        system_performance,
        "get_downloaded_model_names",
        lambda: ["smolvlm_2b"],
    )
    with pytest.raises(SystemExit):
        system_performance.resolve_model_names([system_performance.ALL_LOADED], parser)

    commands = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        commands.append((command, check))
        return_code = 1 if command[command.index("--model") + 1] == "bad" else 0
        return subprocess.CompletedProcess(command, returncode=return_code)

    monkeypatch.setattr(system_performance.subprocess, "run", fake_run)
    system_performance.run_model_processes(["good"], "cpu", num_images=10)
    assert commands[0][0][-6:] == [
        "--model",
        "good",
        "--device",
        "cpu",
        "--num-images",
        "10",
    ]
    assert commands[0][1] is False
    with pytest.raises(RuntimeError, match="bad"):
        system_performance.run_model_processes(["good", "bad"], "cpu", None)

    loaded_model = object()
    appended = []

    class FakeBenchmark:
        def __init__(self, model_name, model, image_paths, device) -> None:
            assert model_name == "yolov5nu"
            assert model is loaded_model
            assert image_paths == [tmp_path / "image.jpg"]
            assert device == torch.device("cpu")

        def run(self) -> BenchmarkResult:
            return make_result("yolov5nu")

    monkeypatch.setattr(
        system_performance, "load_model", lambda model_name: loaded_model
    )
    monkeypatch.setattr(system_performance, "PerformanceBenchmark", FakeBenchmark)
    monkeypatch.setattr(
        system_performance,
        "append_result_csv",
        lambda result, output_path: appended.append((result, output_path)),
    )
    system_performance.run_model_benchmark(
        "yolov5nu",
        [tmp_path / "image.jpg"],
        torch.device("cpu"),
    )
    assert appended == [(make_result("yolov5nu"), system_performance.RESULTS_CSV)]

    output_path = tmp_path / "results.csv"
    output_path.write_text("old", encoding="utf-8")
    main_calls = []
    monkeypatch.setattr(system_performance, "RESULTS_CSV", output_path)
    monkeypatch.setattr(
        system_performance, "resolve_device", lambda name: torch.device(name)
    )
    monkeypatch.setattr(
        system_performance,
        "run_model_processes",
        lambda *args: main_calls.append(args),
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
    system_performance.main()

    assert not output_path.exists()
    assert main_calls == [(["yolov5nu", "yolov8n"], "cpu", None)]
