import pandas as pd
import pytest

import experiments.vision.plot_vision_benchmark as plots


def test_creates_only_task_specific_plots(monkeypatch, tmp_path) -> None:
    classification_csv = tmp_path / "classification_results.csv"
    detection_csv = tmp_path / "object_detection_results.csv"
    pd.DataFrame(
        {
            "model_name": ["mobilenet_v2"],
            "top1_accuracy": [0.65],
            "top5_accuracy": [0.9],
            "precision": [0.8],
            "recall": [0.6],
            "f1_score": [0.68],
        }
    ).to_csv(classification_csv, index=False)
    pd.DataFrame(
        {
            "model_name": ["yolov8n"],
            "map_50_95": [0.4],
            "map_50": [0.5],
            "map_75": [0.3],
            "precision": [0.7],
            "recall": [0.5],
            "f1_score": [0.58],
            "mean_iou": [0.85],
        }
    ).to_csv(detection_csv, index=False)

    calls = []

    def fake_plot(results, metric, task_title, color, output_path):
        calls.append((metric, task_title, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")

    monkeypatch.setattr(plots, "plot_metric", fake_plot)
    created = plots.create_vision_plots(classification_csv, detection_csv)

    assert len(created) == 12
    assert {path.parent.name for path in created} == {
        "image_classification",
        "object_detection",
    }
    assert not (tmp_path / "common_metrics").exists()


def test_plot_metric_writes_image(tmp_path) -> None:
    output_path = tmp_path / "f1_score_bar.png"
    plots.plot_metric(
        pd.DataFrame({"model_name": ["model"], "f1_score": [0.75]}),
        "f1_score",
        "Image classification",
        "#4C78A8",
        output_path,
    )
    assert output_path.is_file()


def test_missing_csv_is_skipped_but_invalid_csv_is_rejected(tmp_path) -> None:
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"accuracy": [0.5]}).to_csv(invalid_csv, index=False)

    with pytest.raises(ValueError, match="model_name"):
        plots.create_vision_plots(invalid_csv, tmp_path / "missing.csv")

    assert (
        plots.create_vision_plots(
            tmp_path / "missing-a.csv", tmp_path / "missing-b.csv"
        )
        == []
    )


def test_stale_plots_are_removed(tmp_path) -> None:
    stale_plot = tmp_path / "plots" / "old_bar.png"
    stale_plot.parent.mkdir()
    stale_plot.write_text("old", encoding="utf-8")
    plots.remove_stale_plots(stale_plot.parent)
    assert not stale_plot.exists()
