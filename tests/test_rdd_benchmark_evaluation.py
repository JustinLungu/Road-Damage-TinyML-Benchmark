from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.rdd_benchmark.data_loader.dataset import (
    BinaryPotholeManifest,
    BinaryPotholeSample,
)
from src.rdd_benchmark.training.evaluation import (
    BinaryPotholeEvaluator,
    RDDEvaluationConfig,
    select_balanced_test_indices,
)


class AlwaysPotholeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        batch_size = images.shape[0]
        logits = torch.stack((-self.bias, self.bias)).repeat(batch_size, 1)
        return logits


def write_test_manifest(tmp_path: Path) -> Path:
    rows = []
    for index, label in enumerate((0, 0, 1, 1)):
        image = tmp_path / "images" / f"{index}.jpg"
        annotation = tmp_path / "annotations" / f"{index}.xml"
        image.parent.mkdir(parents=True, exist_ok=True)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=64 + index).save(image)
        annotation.write_text("<annotation />", encoding="utf-8")
        rows.append(
            f"{image},{annotation},India,test,{label},"
            f"{'pothole' if label else 'non_pothole'}"
        )
    manifest_path = tmp_path / "test.csv"
    manifest_path.write_text(
        "image_path,annotation_path,country,split,label,label_name\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_evaluator_writes_realistic_and_balanced_test_outputs(tmp_path):
    model = AlwaysPotholeModel()
    config = RDDEvaluationConfig(
        model_name="resnet18",
        experiment_name="test_experiment",
        test_manifest_path=write_test_manifest(tmp_path),
        output_dir=tmp_path / "results",
        batch_size=2,
        num_workers=0,
        save_plots=False,
        device="cpu",
    )
    config.model_output_dir.mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict()}, config.checkpoint_path)

    result = BinaryPotholeEvaluator(config, model=model).evaluate()

    assert result.metrics["accuracy"] == 0.5
    assert result.metrics["recall"] == 1.0
    assert result.metrics_csv_path.is_file()
    assert result.inference_metrics_path.is_file()
    assert (config.model_output_dir / "balanced_test_metrics.csv").is_file()
    assert result.confusion_matrix_plot_path is None


def test_balanced_test_indices_are_deterministic_and_class_balanced(tmp_path):
    samples = [
        BinaryPotholeSample(
            tmp_path / f"{index}.jpg",
            tmp_path / f"{index}.xml",
            label,
            "pothole" if label else "non_pothole",
            "India",
            "test",
        )
        for index, label in enumerate((0, 0, 0, 1, 1))
    ]
    manifest = BinaryPotholeManifest(samples, tmp_path / "test.csv")

    first = select_balanced_test_indices(manifest, random_seed=7)
    second = select_balanced_test_indices(manifest, random_seed=7)

    assert first == second
    assert len(first) == 4
    assert sum(samples[index].label for index in first) == 2


def test_balanced_test_indices_are_empty_when_one_class_is_missing(tmp_path):
    manifest = BinaryPotholeManifest(
        [
            BinaryPotholeSample(
                tmp_path / "a.jpg",
                tmp_path / "a.xml",
                0,
                "non_pothole",
                "India",
                "test",
            )
        ],
        tmp_path / "test.csv",
    )
    assert select_balanced_test_indices(manifest) == []
