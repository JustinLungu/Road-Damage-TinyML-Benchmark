import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.vision_benchmark.object_detection_benchmark as detection_module
import src.vision_benchmark.object_detection_inference_adapter as adapter_module
from src.vision_benchmark.object_detection_benchmark import (
    GroundTruthBox,
    box_iou,
    build_category_mapping,
    calculate_coco_map,
    match_detections,
)
from src.vision_benchmark.object_detection_inference_adapter import (
    ObjectDetectionInferenceAdapter,
    PredictedBox,
)


def write_coco_dataset(tmp_path: Path) -> tuple[Path, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "image.jpg").write_bytes(b"image")
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [
                    {"id": 1, "file_name": "image.jpg", "width": 100, "height": 100}
                ],
                "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return images_dir, annotation_path


def test_matching_iou_category_mapping_and_coco_map(tmp_path) -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert build_category_mapping(
        {0: "person", 1: "car"},
        [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
    ) == {0: 1, 1: 3}
    with pytest.raises(ValueError, match="do not match"):
        build_category_mapping({0: "unknown"}, [{"id": 1, "name": "person"}])

    counts = match_detections(
        {
            1: [
                GroundTruthBox(1, (0, 0, 10, 10), False),
                GroundTruthBox(1, (20, 20, 30, 30), True),
            ]
        },
        {
            1: [
                (1, PredictedBox(0, 0.9, (0, 0, 10, 10))),
                (1, PredictedBox(0, 0.8, (0, 0, 10, 10))),
                (1, PredictedBox(0, 0.7, (20, 20, 30, 30))),
                (2, PredictedBox(1, 0.6, (40, 40, 50, 50))),
            ]
        },
        0.25,
        0.5,
    )
    assert (counts.true_positive, counts.false_positive, counts.false_negative) == (
        1,
        2,
        0,
    )
    assert counts.matched_ious == (1.0,)

    _, annotation_path = write_coco_dataset(tmp_path)
    metrics = calculate_coco_map(
        annotation_path,
        [1],
        [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99}],
    )
    assert metrics == pytest.approx((1.0, 1.0, 1.0))


def test_object_detection_benchmark(monkeypatch, tmp_path) -> None:
    images_dir, annotation_path = write_coco_dataset(tmp_path)

    class FakeModel:
        names = {0: "object"}

    class FakeAdapter:
        def __init__(self, *args) -> None:
            pass

        def predict(self, image_path):
            return [PredictedBox(0, 0.9, (10, 10, 30, 30))]

    monkeypatch.setattr(
        detection_module, "ObjectDetectionInferenceAdapter", FakeAdapter
    )
    result = detection_module.ObjectDetectionBenchmark(
        "yolov8n",
        FakeModel(),
        images_dir,
        annotation_path,
        torch.device("cpu"),
        0.25,
        0.5,
    ).run()

    assert result.num_images == 1
    assert result.map_50_95 == pytest.approx(1.0)
    assert (result.precision, result.recall, result.f1_score, result.mean_iou) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )


def test_ultralytics_adapter_normalizes_boxes(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    calls = []

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    boxes = SimpleNamespace(
        xyxy=FakeTensor([[1, 2, 3, 4]]),
        conf=FakeTensor([0.9]),
        cls=FakeTensor([1.0]),
    )

    class FakeYolo:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=boxes)]

    predictions = ObjectDetectionInferenceAdapter(
        "yolov8n", FakeYolo(), torch.device("cpu")
    ).predict(image_path)
    assert predictions == [PredictedBox(1, 0.9, (1.0, 2.0, 3.0, 4.0))]
    assert calls[0]["conf"] == adapter_module.PREDICTION_CONFIDENCE_FLOOR

    with pytest.raises(ValueError, match="not an object detector"):
        ObjectDetectionInferenceAdapter("mobilenet_v2", object(), torch.device("cpu"))
