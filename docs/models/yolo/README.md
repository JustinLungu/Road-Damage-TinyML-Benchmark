# YOLO Detectors

This folder documents the Ultralytics YOLO object detectors used in the repository:

| Repo name | Checkpoint | Family | Local checkpoint size | Local top-level layers | Strides |
| --- | --- | --- | --- | --- | --- |
| `yolov5nu` | `models/cnn/yolov5nu.pt` | YOLOv5u nano | about 5.4 MB | 25 | 8, 16, 32 |
| `yolov8n` | `models/cnn/yolov8n.pt` | YOLOv8 nano | about 6.3 MB | 23 | 8, 16, 32 |

Both are COCO-pretrained object detectors with 80 classes. In this project they are used as fast object-detection baselines for inference demos and system-performance benchmarks.

The repo registry names are defined in `src/constants.py`:

```python
YOLO_MODEL_CHECKPOINTS = {
    "yolov5nu": "yolov5nu.pt",
    "yolov8n": "yolov8n.pt",
}
```

## What Makes YOLO Different

YOLO is a one-stage detector: one forward pass predicts object boxes and class scores directly, without a separate region proposal step.

Both local checkpoints use the modern Ultralytics anchor-free, objectness-free split detection head. That means they do not rely on predefined anchor box templates, and the head separates box localization from class prediction.

The main difference between the two local models is the feature extractor around that shared detection style:

- `yolov5nu` keeps a YOLOv5-style backbone/neck with C3/CSP bottleneck blocks, but uses the newer YOLOv5u anchor-free split head.
- `yolov8n` uses YOLOv8-style C2f blocks in the backbone/neck, plus the same kind of anchor-free split Detect head.

Official Ultralytics metrics at image size 640:

| Model | Parameters | FLOPs | COCO mAP50-95 |
| --- | --- | --- | --- |
| `yolov5nu.pt` | 2.6M | 7.7B | 34.3 |
| `yolov8n.pt` | 3.2M | 8.7B | 37.3 |

Sources:

- <https://docs.ultralytics.com/models/yolov5/>
- <https://docs.ultralytics.com/models/yolov8/>

## Internal Flow

At inference time, both models follow the same high-level path:

1. The Ultralytics predictor loads the image path, resizes/letterboxes it to the requested inference size, converts it to a tensor, normalizes it, and batches it.
2. The backbone applies Conv-BatchNorm-SiLU layers and model-specific blocks:
   - `yolov5nu`: C3/CSP bottleneck blocks;
   - `yolov8n`: C2f blocks.
3. SPPF near the deepest feature map increases receptive field cheaply by pooling at multiple effective scales.
4. The neck upsamples and concatenates features in a feature-pyramid style path, so high-level semantic features and higher-resolution spatial features meet.
5. The detection head predicts boxes and class scores at three scales:
   - stride 8 for smaller objects;
   - stride 16 for medium objects;
   - stride 32 for larger objects.
6. Ultralytics postprocesses raw predictions with confidence filtering and non-maximum suppression, then returns a `Results` object.

The important practical consequence is that this repo does not manually prepare image tensors for YOLO. The adapter passes a file path to Ultralytics and lets the library handle preprocessing and postprocessing.

## How This Repo Loads Them

`src/load_model.py` uses the shared registry and returns an Ultralytics `YOLO` wrapper:

```python
from src.load_model import load_model

model = load_model("yolov8n")
print(type(model))
```

Equivalent direct load:

```python
from ultralytics import YOLO

model = YOLO("models/cnn/yolov8n.pt")
```

The benchmark path is the same for both YOLO models:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model(model_name)
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> model.predict(source=str(image_path), device=str(device), verbose=False)
```

For YOLO, `ModelInferenceAdapter._infer_yolo()` is intentionally minimal:

```python
self.model.predict(
    source=str(image_path),
    device=str(self.device),
    verbose=False,
)
```

## Inference Example

```python
from pathlib import Path

from src.load_model import load_model

model_name = "yolov8n"
image_path = Path("datasets/coco/images/000000047585.jpg")
model = load_model(model_name)

results = model.predict(
    source=str(image_path),
    device="cpu",
    imgsz=640,
    conf=0.25,
    verbose=False,
)

result = results[0]
print(result.orig_shape)
print(result.boxes.data)
```

## Reading The Output

For object detection, `result.boxes` is the most important field.

```python
boxes = result.boxes

xyxy = boxes.xyxy      # x1, y1, x2, y2 in original image pixels
conf = boxes.conf      # confidence per detection
cls = boxes.cls        # class id per detection
names = result.names   # class id -> class name
```

`boxes.data` is a tensor shaped `(N, 6)`, where each row is:

```text
x1, y1, x2, y2, confidence, class_id
```

Example conversion to plain Python rows:

```python
rows = []
for box, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
    class_id_int = int(class_id.item())
    rows.append(
        {
            "label": result.names[class_id_int],
            "confidence": float(score.item()),
            "xyxy": [round(float(value), 1) for value in box.tolist()],
        }
    )

print(rows)
```

To create an annotated image:

```python
result.save(filename="docs/models/yolo/outputs/yolov8n_annotated.jpg")
```

## Quick Demo

Run the local demo from the repository root:

```bash
uv run python docs/models/yolo/demo.py
```

The demo is intentionally configured by constants at the top of `demo.py`, so edit these values before running:

```python
MODEL_NAMES = ["yolov5nu", "yolov8n"]
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"
RANDOM_SEED = None
DEVICE_NAME = "cpu"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.6
NMS_IOU_THRESHOLD = 0.7
SAVE_ANNOTATED_IMAGE = True
OUTPUT_DIR = MODEL_DOCS_DIR / "outputs"
```

Set `MODEL_NAMES` to one or both supported names. For example, `["yolov8n"]` runs only YOLOv8n, while `["yolov5nu", "yolov8n"]` runs both models on the same selected image. Set `USE_RANDOM_IMAGE = False` to use `IMAGE_PATH`. Set `USE_RANDOM_IMAGE = True` to ignore `IMAGE_PATH` and sample a random `.jpg` from `RANDOM_IMAGE_DIR`. Set `RANDOM_SEED` to an integer if you want the same random image across runs.

With `USE_RANDOM_IMAGE = False` and `CONFIDENCE_THRESHOLD = 0.25`, the default image produced:

| Model | Detections |
| --- | --- |
| `yolov5nu` | 6 |
| `yolov8n` | 7 |

The exact confidences can shift slightly across Ultralytics/PyTorch versions.

## Benchmark Commands

The system-performance benchmark measures latency, FPS, RAM, GPU metrics, power, and energy per inference. It does not report detection accuracy.

```bash
uv run python experiments/performance/system_performance.py \
  --model yolov5nu \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model yolov8n \
  --device cuda:0 \
  --num-images 100
```
