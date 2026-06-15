# MobileNet Classifiers

This folder documents the Torchvision MobileNet classifiers used in the
repository:

| Repo name | Checkpoint location | Family | Parameters | FLOPs | ImageNet-1K acc@1 / acc@5 |
| --- | --- | --- | --- | --- | --- |
| `mobilenet_v2` | `models/cnn/hub/checkpoints/mobilenet_v2-7ebf99e0.pth` | MobileNetV2 | 3.50M | 0.301B | 72.154 / 90.822 |
| `mobilenet_v3_small` | `models/cnn/hub/checkpoints/mobilenet_v3_small-047dcff4.pth` | MobileNetV3 Small | 2.54M | 0.057B | 67.668 / 87.402 |
| `mobilenet_v3_large` | `models/cnn/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth` | MobileNetV3 Large | 5.48M | 0.217B | 75.274 / 92.566 |

All three are ImageNet-1K pretrained image classifiers with 1000 output
classes. In this project they are lightweight CNN baselines for inference
demos and system-performance benchmarks.

The parameters, FLOPs, accuracy values, preprocessing transforms, and class
labels come from the Torchvision metadata attached to each default weight.

For definitions of terms such as logits, softmax, top-k, depthwise separable
convolution, inverted residual block, linear bottleneck, ReLU6,
squeeze-and-excitation, and FLOPs, see [definitions.md](definitions.md).

The checkpoint paths are registered in `src/constants.py`:

```python
MOBILENET_MODEL_CHECKPOINTS = {
    "mobilenet_v2": "mobilenet_v2-7ebf99e0.pth",
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
    "mobilenet_v3_large": "mobilenet_v3_large-5c1a4163.pth",
}
```

## What Makes MobileNet Different

MobileNet models are CNNs designed for efficient inference on mobile and edge
hardware. They replace many expensive standard convolutions with depthwise
separable convolutions and compact inverted residual blocks.

The local variants make different accuracy and compute tradeoffs:

- `mobilenet_v2` is the original inverted-residual and linear-bottleneck
  baseline. It is especially relevant to edge deployment because its operators
  are simple and widely supported.
- `mobilenet_v3_small` is the cheapest local variant. It uses substantially
  fewer FLOPs, but its ImageNet accuracy is lower.
- `mobilenet_v3_large` is the strongest local MobileNet classifier. It uses
  more parameters and memory than the other variants.

MobileNetV3 builds on MobileNetV2 by adding architecture choices found through
hardware-aware search, including squeeze-and-excitation blocks and h-swish
activations.

Unlike YOLO, MobileNet is not an object detector. It does not return boxes. It
returns one 1000-class ImageNet score vector for the whole image. The top-k
output should be read as "which ImageNet labels best describe the full image
crop?", not "which objects were detected and where are they?"

## Internal Flow

At inference time, all local MobileNet models follow this broad path:

```text
image file
  -> Torchvision preprocessing
  -> convolutional stem
  -> inverted residual feature blocks
  -> final convolution
  -> global average pooling
  -> classifier head
  -> logits
  -> softmax probabilities
  -> top-k labels
```

### 1. Preprocessing

The demo loads the image as RGB and applies the exact transform bundled with
the selected Torchvision weight:

```python
transform = weights.transforms()
input_tensor = transform(image).unsqueeze(0)
```

The transforms resize, center crop, convert to tensor, and normalize with
ImageNet statistics. The local default weights use a 224x224 crop:

- `mobilenet_v2`: resize to 232, center crop to 224;
- `mobilenet_v3_small`: resize to 256, center crop to 224;
- `mobilenet_v3_large`: resize to 232, center crop to 224.

The source COCO image can have any size. The model receives a normalized
batched tensor shaped `1 x 3 x 224 x 224`.

### 2. MobileNetV2 Feature Extractor

MobileNetV2 begins with a standard convolution, then uses inverted residual
blocks.

A typical block:

1. expands the channel width with a 1x1 convolution;
2. applies an efficient depthwise spatial convolution;
3. projects the channels back down with a linear 1x1 convolution;
4. adds a residual connection when the input/output shape allows it.

The narrow projection is called a linear bottleneck because it does not apply
another non-linearity after compressing the features. MobileNetV2 mainly uses
ReLU6 inside the expanded part of each block.

### 3. MobileNetV3 Feature Extractor

MobileNetV3 keeps the inverted-residual structure but changes the block
configuration for better hardware-aware efficiency.

Some blocks use squeeze-and-excitation to reweight feature channels from global
context. The network also mixes ReLU and h-swish activations. Small and Large
use different channel widths and block layouts to target different resource
budgets.

### 4. Global Pooling

After the feature extractor, global average pooling collapses the spatial
dimensions into one feature vector.

This removes explicit object-location information. The model answers what best
describes the whole crop rather than where an object appears.

### 5. Classifier Head

The classifier head maps the pooled feature vector to 1000 logits, one for each
ImageNet-1K class.

The demo applies softmax and selects the top-k classes:

```python
probabilities = logits.softmax(dim=1)[0]
scores, class_ids = probabilities.topk(TOP_K)
```

## How This Repo Loads Them

`src/load_model.py` returns Torchvision MobileNet modules:

```python
from src.load_model import load_model

model = load_model("mobilenet_v2")
print(type(model))
```

Before loading, the repo sets `TORCH_HOME` to `models/cnn`. Torchvision then
reads or downloads the weights under:

```text
models/cnn/hub/checkpoints/
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model(model_name)
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> matching Torchvision weight transforms
  -> model(input_tensor)
```

## MobileNetV2 Inference Example

```python
from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights

from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image

model_name = "mobilenet_v2"
image_path = Path("datasets/coco/images/000000047585.jpg")
weights = MobileNet_V2_Weights.DEFAULT
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
    class_id_int = int(class_id)
    print(categories[class_id_int], float(score))
```

## Reading The Output

Each MobileNet returns logits shaped:

```text
batch_size, 1000
```

For one image, the shape is:

```text
1, 1000
```

Each value corresponds to one ImageNet class. The largest value is the model's
preferred class before softmax. After softmax, the largest probability is the
model's most likely class.

The demo prints:

- rank;
- label;
- probability;
- ImageNet class ID.

For COCO images with multiple objects, the top label may describe the dominant
object, person, scene, or a related ImageNet category. It should not be
interpreted like detector output.

## Quick Demo

Run the local demo from the repository root:

```bash
uv run python docs/models/mobilenet/demo.py
```

The demo is configured through constants at the top of `demo.py`:

```python
MODEL_NAMES = ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]
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

Set `MODEL_NAMES = ["mobilenet_v2"]` to run only MobileNetV2. Multiple variants
reuse the same selected image so their top-k predictions can be compared.

Set `USE_RANDOM_IMAGE = False` to use `IMAGE_PATH`. Set it to `True` to sample a
random `.jpg` from `RANDOM_IMAGE_DIR`. Set `RANDOM_SEED` to an integer for
repeatable image selection.

When `SAVE_RESULTS_JSON = True`, each model writes a top-k JSON file under:

```text
docs/models/mobilenet/outputs/
```

When `SAVE_INPUT_IMAGE = True`, the selected source image is copied to:

```text
docs/models/mobilenet/outputs/selected_image.jpg
```

The input image is not annotated with boxes because MobileNet does not predict
object locations.

With `USE_RANDOM_IMAGE = False`, MobileNetV2 classified the configured example
image most strongly as `groom`. Exact probabilities can shift slightly across
Torchvision and PyTorch versions.

## Benchmark Commands

The system-performance benchmark measures latency, FPS, RAM, GPU metrics,
power, and energy per inference. It does not report classification accuracy.

```bash
uv run python experiments/performance/system_performance.py \
  --model mobilenet_v2 \
  --device cuda:0 \
  --num-images 100
```

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

## Sources

- Torchvision MobileNetV2 model:
  <https://docs.pytorch.org/vision/stable/models/mobilenetv2.html>
- MobileNetV2 paper:
  <https://arxiv.org/abs/1801.04381>
