# EfficientNet-B0 Classifier

This folder documents the Torchvision EfficientNet-B0 classifier used in the
repository:

| Repo name | Checkpoint location | Parameters | FLOPs | ImageNet-1K acc@1 / acc@5 |
| --- | --- | --- | --- | --- |
| `efficientnet_b0` | `models/cnn/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth` | 5.29M | 0.386B | 77.692 / 93.532 |

EfficientNet-B0 is an ImageNet-1K pretrained image classifier with 1000 output
classes. It is included as a strong lightweight CNN baseline for inference
demos and system-performance benchmarks.

The parameter count, FLOPs, accuracy, preprocessing transform, and class labels
come from the metadata attached to Torchvision's default
`EfficientNet_B0_Weights`.

For definitions of terms such as compound scaling, MBConv, depthwise
convolution, squeeze-and-excitation, SiLU, stochastic depth, logits, softmax,
top-k, parameters, and FLOPs, see [definitions.md](definitions.md).

The checkpoint path is registered in `src/constants.py`:

```python
EFFICIENTNET_MODEL_CHECKPOINTS = {
    "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
}
```

## What Makes EfficientNet Different

EfficientNet was designed around compound scaling. Instead of making a CNN
larger along only one dimension, the family scales network depth, channel
width, and input resolution together.

EfficientNet-B0 is the baseline architecture from which the larger B1-B7
variants are scaled. This repository starts with B0 because it offers a useful
lightweight accuracy baseline: its Torchvision weights report higher
ImageNet-1K top-1 accuracy than the local MobileNet variants while remaining a
small CNN.

EfficientNet-B0 is not an object detector. It returns one 1000-class score
vector for the whole image and does not return bounding boxes. Its top-k output
should be read as "which ImageNet labels best describe the full image crop?"

## Internal Flow

At inference time, the model follows this broad path:

```text
image file
  -> Torchvision preprocessing
  -> convolutional stem
  -> MBConv stages
  -> squeeze-and-excitation inside each MBConv
  -> final 1x1 convolution
  -> global average pooling
  -> dropout and linear classifier
  -> logits
  -> softmax probabilities
  -> top-k labels
```

### 1. Preprocessing

The demo loads an image as RGB and applies the exact transform bundled with the
pretrained weight:

```python
weights = EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()
input_tensor = transform(image).unsqueeze(0)
```

The default transform:

- resizes the image to 256 using bicubic interpolation;
- center crops it to 224x224;
- converts it to a tensor;
- normalizes RGB with ImageNet mean `[0.485, 0.456, 0.406]`;
- normalizes RGB with ImageNet standard deviation
  `[0.229, 0.224, 0.225]`.

The model receives a batched tensor shaped `1 x 3 x 224 x 224`.

### 2. Convolutional Stem

The stem uses a stride-2 3x3 convolution to create 32 feature channels while
reducing the spatial resolution. Batch normalization and SiLU activation
follow the convolution.

### 3. MBConv Stages

The main feature extractor contains mobile inverted bottleneck convolution
blocks, usually shortened to MBConv.

A typical MBConv block:

1. expands the channel width with a 1x1 convolution;
2. applies a 3x3 or 5x5 depthwise convolution;
3. uses squeeze-and-excitation to reweight channels;
4. projects features to the output width with a 1x1 convolution;
5. adds a residual connection when input and output shapes match.

The first stage omits the expansion step because its expansion ratio is one.
Later stages use wider intermediate representations. Stride-2 blocks reduce
spatial resolution between selected stages.

### 4. Stochastic Depth

Residual MBConv blocks include stochastic depth during training. It randomly
skips some residual branches as regularization. The skip probability increases
through the network.

The loaded model is put in evaluation mode, so stochastic depth is disabled
during demos and benchmarks.

### 5. Classification Head

After the MBConv stages, a 1x1 convolution expands the final features to 1280
channels. Global average pooling reduces the spatial feature map to one vector.
The classifier applies dropout and maps that vector to 1000 ImageNet logits.

The demo converts logits to probabilities and selects the top-k classes:

```python
probabilities = logits.softmax(dim=1)[0]
scores, class_ids = probabilities.topk(TOP_K)
```

## How This Repo Loads It

`src/load_model.py` returns the pretrained Torchvision module:

```python
from src.load_model import load_model

model = load_model("efficientnet_b0")
print(type(model))
```

Before loading, the repository sets `TORCH_HOME` to `models/cnn`. Torchvision
therefore reads or downloads the weight under:

```text
models/cnn/hub/checkpoints/
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model("efficientnet_b0")
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> EfficientNet_B0_Weights.DEFAULT.transforms()
  -> model(input_tensor)
```

## Inference Example

```python
from pathlib import Path

import torch
from torchvision.models import EfficientNet_B0_Weights

from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image

image_path = Path("datasets/coco/images/000000047585.jpg")
weights = EfficientNet_B0_Weights.DEFAULT
model = load_model("efficientnet_b0").eval()
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

EfficientNet-B0 returns logits shaped:

```text
batch_size, 1000
```

For one image, the shape is `1 x 1000`. Each value corresponds to one
ImageNet-1K class. The demo prints rank, label, softmax probability, and
ImageNet class ID.

COCO images may contain several objects or scenes that do not map cleanly to
one ImageNet label. The result is whole-image classification, not detection.

## Quick Demo

Run directly:

```bash
uv run python docs/models/efficientnet/demo.py
```

Or use the shared demo script:

```bash
./scripts/inference_demo.sh efficientnet
```

The demo is configured through constants at the top of `demo.py`:

```python
MODEL_NAMES = ["efficientnet_b0"]
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

Set `USE_RANDOM_IMAGE = False` to use `IMAGE_PATH`. Set `RANDOM_SEED` to an
integer for reproducible random selection.

When saving is enabled, the selected image and
`efficientnet_b0_topk.json` are written under
`docs/models/efficientnet/outputs/`. Generated outputs are ignored by Git.

## Performance Benchmark

Run a small CPU benchmark:

```bash
uv run python experiments/performance/system_performance.py \
  --model efficientnet_b0 \
  --device cpu \
  --num-images 100
```

Use `--device cuda:0` on a CUDA system. Benchmark rows are appended to
`results/system_metrics/system_performance_results.csv`.

## Sources

- [Torchvision EfficientNet documentation](https://docs.pytorch.org/vision/stable/models/efficientnet.html)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
