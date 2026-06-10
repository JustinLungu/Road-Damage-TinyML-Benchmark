# MobileNetV3 Classifiers

This folder documents the Torchvision MobileNetV3 classifiers used in the repository:

| Repo name | Checkpoint location | Family | Parameters | FLOPs | ImageNet-1K acc@1 / acc@5 |
| --- | --- | --- | --- | --- | --- |
| `mobilenet_v3_small` | `models/cnn/hub/checkpoints/mobilenet_v3_small-047dcff4.pth` | MobileNetV3 Small | 2.54M | 0.057B | 67.668 / 87.402 |
| `mobilenet_v3_large` | `models/cnn/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth` | MobileNetV3 Large | 5.48M | 0.217B | 75.274 / 92.566 |

Both are ImageNet-1K pretrained image classifiers with 1000 output classes. In this project they are useful lightweight CNN baselines for system-performance benchmarks.

The parameters, FLOPs, and accuracy values above come from the local Torchvision weight metadata for the default pretrained weights.

For definitions of terms such as logits, softmax, top-k, depthwise separable convolution, inverted residual block, squeeze-and-excitation, and FLOPs, see [definitions.md](definitions.md).

The repo registry names are defined in `src/load_model.py` and checkpoint paths are defined in `src/constants.py`:

```python
MOBILENET_MODEL_CHECKPOINTS = {
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
    "mobilenet_v3_large": "mobilenet_v3_large-5c1a4163.pth",
}
```

## What Makes MobileNet Different

MobileNet models are CNNs designed for efficient inference. They trade some accuracy for lower latency, lower memory use, and lower compute cost.

The central idea is to replace many expensive standard convolutions with efficient blocks based on depthwise separable convolutions and inverted residuals. MobileNetV3 also uses squeeze-and-excitation and hardware-aware activation choices such as h-swish.

The difference between the local variants is scale:

- `mobilenet_v3_small` is smaller and cheaper. It has fewer parameters, lower FLOPs, and lower ImageNet accuracy.
- `mobilenet_v3_large` is larger and more accurate. It costs more memory and compute, but usually gives stronger classifications.

Unlike YOLO, MobileNetV3 is not an object detector. It does not return boxes. It returns one 1000-class ImageNet score vector for the whole image. The top-k output should be read as "which ImageNet labels best describe the full image crop?", not "which objects were detected and where are they?"

## Internal Flow

At inference time, both models follow this broad path:

```text
image file
  -> Torchvision preprocessing
  -> MobileNetV3 feature extractor
  -> global pooling
  -> classifier head
  -> logits
  -> softmax probabilities
  -> top-k labels
```

### 1. Preprocessing

The demo loads the image as RGB and applies the exact transform bundled with the selected Torchvision weights:

```python
transform = weights.transforms()
input_tensor = transform(image).unsqueeze(0)
```

The transforms resize, center crop, convert to tensor, and normalize with ImageNet statistics. The local default weights use a 224x224 crop:

- `mobilenet_v3_small`: resize to 256, center crop to 224;
- `mobilenet_v3_large`: resize to 232, center crop to 224.

This means the original COCO image can have any size. The model still receives a normalized 224x224 tensor.

### 2. Feature Extractor

The MobileNetV3 feature extractor is a stack of efficient convolutional blocks.

Instead of using only standard convolutions, MobileNetV3 relies heavily on inverted residual blocks and depthwise separable convolutions. These reduce computation by separating spatial filtering from channel mixing.

The network gradually reduces spatial resolution while increasing feature depth. Early layers detect simple patterns like edges and textures. Later layers represent higher-level object and scene cues.

### 3. Squeeze-and-Excitation and Activations

Some MobileNetV3 blocks use squeeze-and-excitation. This lets the model emphasize or suppress feature channels based on global context.

MobileNetV3 also uses hardware-aware nonlinearities such as h-swish. These choices are part of why MobileNetV3 is efficient on edge devices.

### 4. Global Pooling

After the convolutional feature extractor, the model uses global pooling to collapse spatial dimensions. Instead of keeping a grid of features, it summarizes the image into one feature vector.

This is one reason classification loses object location information. After global pooling, the model is focused on "what is in the image", not "where each object is".

### 5. Classifier Head

The classifier head maps the pooled feature vector to 1000 logits, one for each ImageNet-1K class.

The demo applies softmax to convert logits into probabilities, then prints the top-k classes:

```python
probabilities = logits.softmax(dim=1)[0]
scores, class_ids = probabilities.topk(TOP_K)
```

## How This Repo Loads Them

`src/load_model.py` returns Torchvision MobileNetV3 modules:

```python
from src.load_model import load_model

model = load_model("mobilenet_v3_large")
print(type(model))
```

The repo sets `TORCH_HOME` to `models/cnn` before loading. Torchvision then reads or writes weights under:

```text
models/cnn/hub/checkpoints/
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model(model_name)
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> weights.transforms()
  -> model(input_tensor)
```

## Inference Example

```python
from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Large_Weights

from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image

model_name = "mobilenet_v3_large"
image_path = Path("datasets/coco/images/000000047585.jpg")
weights = MobileNet_V3_Large_Weights.DEFAULT
model = load_model(model_name).eval()
transform = weights.transforms()
categories = weights.meta["categories"]

image = load_rgb_image(image_path)
input_tensor = transform(image).unsqueeze(0)

with torch.inference_mode():
    logits = model(input_tensor)
    probabilities = logits.softmax(dim=1)[0]
    scores, class_ids = probabilities.topk(5)

for score, class_id in zip(scores, class_ids):
    print(categories[int(class_id)], float(score))
```

## Reading The Output

MobileNetV3 returns logits shaped:

```text
batch_size, 1000
```

For a single image, the shape is usually:

```text
1, 1000
```

Each of the 1000 values corresponds to one ImageNet class. The largest value is the model's preferred class before softmax. After softmax, the largest probability is the model's most likely class.

The demo prints rows with:

- rank;
- label;
- probability;
- ImageNet class id.

For COCO images with multiple objects, the top label may describe the dominant object, person, clothing, scene, or a related ImageNet category. It should not be interpreted like a detector output.

## Quick Demo

Run the local demo from the repository root:

```bash
uv run python docs/models/mobilenet/demo.py
```

The demo is intentionally configured by constants at the top of `demo.py`, so edit these values before running:

```python
MODEL_NAMES = ["mobilenet_v3_small", "mobilenet_v3_large"]
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"
RANDOM_SEED = None
DEVICE_NAME = "cpu"
TOP_K = 5
SAVE_RESULTS_JSON = True
SAVE_INPUT_IMAGE = True
OUTPUT_DIR = MODEL_DOCS_DIR / "outputs"
```

Set `MODEL_NAMES` to one or both supported names. For example, `["mobilenet_v3_small"]` runs only the small model, while `["mobilenet_v3_small", "mobilenet_v3_large"]` runs both models on the same selected image.

Set `USE_RANDOM_IMAGE = False` to use `IMAGE_PATH`. Set `USE_RANDOM_IMAGE = True` to ignore `IMAGE_PATH` and sample a random `.jpg` from `RANDOM_IMAGE_DIR`. Set `RANDOM_SEED` to an integer if you want the same random image across runs.

When `SAVE_RESULTS_JSON = True`, each model writes a top-k JSON file under:

```text
docs/models/mobilenet/outputs/
```

When `SAVE_INPUT_IMAGE = True`, the selected source image is also copied to:

```text
docs/models/mobilenet/outputs/selected_image.jpg
```

This image is the visual reference for the top-k whole-image classifications. It is not annotated with boxes because MobileNetV3 does not predict object locations.

With `USE_RANDOM_IMAGE = False`, the default image produced `groom` as the top prediction for both local checkpoints.

## Benchmark Commands

The system-performance benchmark measures latency, FPS, RAM, GPU metrics, power, and energy per inference. It does not report classification accuracy.

```bash
uv run python experiments/performance/system_performance.py \
  --model mobilenet_v3_small \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model mobilenet_v3_large \
  --device cuda:0 \
  --num-images 100
```
