# YOLOv5nu

`yolov5nu` is the nano-sized Ultralytics YOLOv5u object detector. In this repository it is used as a fast COCO-pretrained detector and benchmark baseline, loaded from:

```text
models/cnn/yolov5nu.pt
```

The repo registry name is `yolov5nu`, defined in `src/constants.py`:

```python
YOLO_MODEL_CHECKPOINTS = {
    "yolov5nu": "yolov5nu.pt",
    "yolov8n": "yolov8n.pt",
}
```

## What Makes It Different

YOLO is a one-stage detector: one forward pass predicts object boxes and class scores directly, without a separate region proposal step.

The `u` in YOLOv5u matters. Ultralytics describes YOLOv5u as a YOLOv5-derived model with an anchor-free, objectness-free split detection head, a head design introduced in YOLOv8. That is the main internal difference from older YOLOv5 checkpoints, which used predefined anchor boxes and an objectness score. The local `yolov5nu.pt` checkpoint keeps the small YOLOv5-style backbone/neck but uses the modern Ultralytics detection head.

Useful local facts from the checkpoint:

- task: object detection;
- classes: 80 COCO classes;
- default image size in the model table: 640 pixels;
- top-level layers: 25;
- detection strides: 8, 16, and 32 pixels;
- checkpoint size in this repo: about 5.4 MB.

Official Ultralytics metrics for `yolov5nu.pt` report 2.6M parameters, 7.7B FLOPs, and 34.3 COCO mAP50-95 at image size 640.

Source: <https://docs.ultralytics.com/models/yolov5/>

## Internal Flow

At inference time, the model does roughly this:

1. The Ultralytics predictor loads the image path, resizes/letterboxes it to the requested inference size, converts it to a tensor, normalizes it, and batches it.
2. The backbone applies Conv-BatchNorm-SiLU layers and C3/CSP bottleneck blocks to build progressively lower-resolution feature maps.
3. SPPF near the deepest feature map increases receptive field cheaply by pooling at multiple effective scales.
4. The neck upsamples and concatenates features in a feature-pyramid style path, so high-level semantic features and higher-resolution spatial features meet.
5. The detection head predicts boxes and class scores at three scales:
   - stride 8 for smaller objects;
   - stride 16 for medium objects;
   - stride 32 for larger objects.
6. Ultralytics postprocesses raw predictions with confidence filtering and non-maximum suppression, then returns a `Results` object.

The important practical consequence is that this repo does not manually prepare image tensors for YOLO. The adapter passes a file path to Ultralytics and lets the library handle preprocessing and postprocessing.

## How This Repo Loads It

`src/load_model.py` uses the shared registry and returns an Ultralytics `YOLO` wrapper:

```python
from src.load_model import load_model

model = load_model("yolov5nu")
print(type(model))
```

Equivalent direct load:

```python
from ultralytics import YOLO

model = YOLO("models/cnn/yolov5nu.pt")
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model("yolov5nu")
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

image_path = Path("datasets/coco/images/000000047585.jpg")
model = load_model("yolov5nu")

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
result.save(filename="docs/models/yolov5nu/outputs/annotated.jpg")
```

## Quick Demo

Run the local demo from the repository root:

```bash
uv run python docs/models/yolov5nu/demo.py
```

The demo is intentionally configured by constants at the top of `demo.py`, so edit these values before running:

```python
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"
USE_RANDOM_IMAGE = False
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"
RANDOM_SEED = None
DEVICE_NAME = "cpu"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.7
SAVE_ANNOTATED_IMAGE = True
OUTPUT_PATH = MODEL_DOCS_DIR / "outputs" / "annotated.jpg"
```

Set `USE_RANDOM_IMAGE = True` to ignore `IMAGE_PATH` and sample a random `.jpg` from `RANDOM_IMAGE_DIR`. Set `RANDOM_SEED` to an integer if you want the same random image across runs.

On the current local checkpoint, the default image produced six detections. A run prints a table with class label, confidence, bounding box coordinates, and box size. The exact confidences can shift slightly across Ultralytics/PyTorch versions.

Example output from the current local checkpoint:

```text
Image: datasets/coco/images/000000047585.jpg
Original shape: (640, 424)
Detections: 6
#  label     conf   x1     y1     x2     y2     w      h
-  --------  -----  -----  -----  -----  -----  -----  -----
1  person    0.858  353.3  189.6  423.6  422.6  70.2   233.0
2  person    0.846  51.6   189.8  249.1  597.8  197.4  408.1
3  person    0.790  179.2  129.2  357.5  597.0  178.2  467.8
4  person    0.383  403.7  200.8  424.0  229.5  20.3   28.6
5  umbrella  0.380  11.6   32.7   363.5  253.9  351.9  221.1
6  umbrella  0.335  16.8   32.7   360.1  121.1  343.3  88.5
```

## Benchmark Command

The system-performance benchmark measures latency, FPS, RAM, GPU metrics, power, and energy per inference. It does not report detection accuracy.

```bash
uv run python experiments/performance/system_performance.py \
  --model yolov5nu \
  --device cuda:0 \
  --num-images 100
```
