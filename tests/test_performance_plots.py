import pandas as pd
import pytest

import experiments.performance.plot_system_performance as plots


def test_creates_task_specific_metric_plots(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "system_performance_results.csv"
    pd.DataFrame(
        {
            "model_name": ["tiny_cnn", "yolov8n", "yolov8n", "smolvlm_256m"],
            "task": [
                "image_classification",
                "object_detection",
                "object_detection",
                "vision_language",
            ],
            "device_name": ["CPU", "GPU A", "GPU B", "GPU A"],
            "num_images": [100, 100, 50, 10],
            "fps": [80.0, 100.0, 110.0, 2.0],
            "avg_power_w": [None, 20.0, 19.0, 50.0],
            "not_a_metric": [1, 2, 3, 4],
        }
    ).to_csv(csv_path, index=False)
    plotted = []

    def fake_plot(plot_data, metric, output_path) -> None:
        plotted.append((plot_data.copy(), metric, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")

    monkeypatch.setattr(plots, "plot_metric", fake_plot)
    created = plots.create_metric_plots(csv_path)

    assert len(created) == 5
    assert {path.parent.name for path in created} == {
        "image_classification",
        "object_detection",
        "vision_language",
    }
    assert plotted[0][0]["model_name"].tolist() == ["tiny_cnn"]
    assert plotted[1][0]["model_name"].tolist() == [
        "yolov8n\nGPU A, n=100",
        "yolov8n\nGPU B, n=50",
    ]
    assert "not_a_metric" not in {metric for _, metric, _ in plotted}
    assert "num_images" not in {metric for _, metric, _ in plotted}


def test_plot_metric_writes_image(tmp_path) -> None:
    output_path = tmp_path / "fps_bar.png"
    plots.plot_metric(
        pd.DataFrame({"model_name": ["model"], "fps": [10.0]}),
        "fps",
        output_path,
    )
    assert output_path.is_file()
    assert plots.format_metric_name("avg_latency_ms") == "Average Latency (ms)"


def test_plot_input_validation_and_stale_cleanup(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        plots.create_metric_plots(tmp_path / "missing.csv")

    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"model_name": ["model"]}).to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        plots.create_metric_plots(invalid_csv)

    assert (
        plots.create_task_plots(pd.DataFrame(), tmp_path / "plots", "vision_language")
        == []
    )

    stale_plot = tmp_path / "plots" / "old_bar.png"
    stale_plot.parent.mkdir(exist_ok=True)
    stale_plot.write_text("old", encoding="utf-8")
    plots.remove_stale_plots(stale_plot.parent)
    assert not stale_plot.exists()
