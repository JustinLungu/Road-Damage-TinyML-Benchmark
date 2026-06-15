# InceptionV3 Classifier

This folder documents the Torchvision InceptionV3 classifier used in the
repository:

| Repo name | Checkpoint location | Parameters | FLOPs | ImageNet-1K acc@1 / acc@5 |
| --- | --- | --- | --- | --- |
| `inception_v3` | `models/cnn/hub/checkpoints/inception_v3_google-0cc3c7bd.pth` | 27.16M | 5.713B | 77.294 / 93.450 |

InceptionV3 is an ImageNet-1K pretrained classifier with 1000 output classes.
It is included as a higher-compute CNN reference with a different architecture
from the edge-oriented MobileNetV2 and EfficientNet-B0 models.

The parameter count, FLOPs, accuracy, preprocessing transform, and class labels
come from Torchvision's default `Inception_V3_Weights`. The parameter count
includes the auxiliary classifier stored in the pretrained model, although
that classifier is inactive during evaluation.

For definitions of Inception module, parallel branch, factorized convolution,
grid reduction, auxiliary classifier, logits, softmax, top-k, parameters, and
FLOPs, see [definitions.md](definitions.md).

The checkpoint path is registered in `src/constants.py`:

```python
INCEPTION_MODEL_CHECKPOINTS = {
    "inception_v3": "inception_v3_google-0cc3c7bd.pth",
}
```

## What Makes InceptionV3 Different

InceptionV3 processes features through several parallel branches within each
Inception module. Different branches use different convolution and pooling
paths, then concatenate their outputs along the channel dimension.

This lets the model capture patterns at several effective receptive-field
sizes in the same stage. It also uses factorized convolutions to reduce the
cost of larger filters.

InceptionV3 is substantially more computationally expensive than the other
new classifiers in this repository. Its default input is 299x299 rather than
224x224, and Torchvision reports 5.713 GFLOPs. This makes it a useful comparison
model, but it is less aligned with constrained Jetson Nano deployment.

Like the other classifiers, InceptionV3 returns one ImageNet score vector for
the whole image. It does not return bounding boxes.

## Internal Flow

The inference path is:

```text
image file
  -> Torchvision preprocessing
  -> 299x299 normalized tensor
  -> convolutional stem
  -> Inception-A modules
  -> grid-reduction Inception-B module
  -> Inception-C modules
  -> grid-reduction Inception-D module
  -> Inception-E modules
  -> global average pooling
  -> dropout and fully connected classifier
  -> logits
  -> softmax probabilities
  -> top-k labels
```

### 1. Preprocessing

The demo applies the exact transform attached to the pretrained weight:

```python
weights = Inception_V3_Weights.DEFAULT
transform = weights.transforms()
input_tensor = transform(image).unsqueeze(0)
```

The default transform:

- resizes the image to 342 using bilinear interpolation;
- center crops it to 299x299;
- converts it to a tensor;
- normalizes RGB with ImageNet mean `[0.485, 0.456, 0.406]`;
- normalizes RGB with ImageNet standard deviation
  `[0.229, 0.224, 0.225]`.

The resulting tensor is shaped `1 x 3 x 299 x 299`.

### 2. Convolutional Stem

The stem uses several 3x3 convolutions, two max-pooling operations, and one
1x1 convolution. It reduces spatial resolution while building the initial
feature channels before the parallel Inception modules.

### 3. Parallel Inception Branches

An Inception module sends the same input through several branches. Depending on
the module type, branches include:

- a 1x1 convolution;
- a larger convolution path preceded by a 1x1 channel-reduction convolution;
- a deeper sequence of spatial convolutions;
- a pooling operation followed by a 1x1 convolution.

The branch outputs are concatenated rather than added. This differs from
ResNet18, where a residual branch is added to a shortcut.

### 4. Factorized Convolutions

InceptionV3 replaces some expensive large square filters with sequences of
smaller or asymmetric filters.

Examples include:

```text
5x5 convolution -> two 3x3 convolutions
7x7 convolution -> 1x7 convolution followed by 7x1 convolution
3x3 convolution -> parallel 1x3 and 3x1 convolutions
```

Factorization reduces computation while preserving a large effective
receptive field and adding extra non-linear transformations.

### 5. Grid Reduction

The `Mixed_6a` and `Mixed_7a` modules reduce spatial resolution. They use
parallel stride-2 convolution and pooling branches so downsampling also learns
richer feature transformations.

### 6. Auxiliary Classifier

The pretrained Torchvision model contains an auxiliary classifier after the
middle Inception stages. During training, this extra head can provide another
learning signal.

This repository calls `.eval()` in the loader, demo, and benchmark adapter. In
evaluation mode, InceptionV3 returns only the main `1 x 1000` logits tensor.
The auxiliary classifier is not executed and no auxiliary output needs to be
handled.

### 7. Classification Head

After the final Inception modules, adaptive global average pooling produces a
2048-value feature vector. Dropout and a fully connected layer map it to 1000
ImageNet logits.

The demo applies softmax and selects the top-k classes:

```python
probabilities = logits.softmax(dim=1)[0]
scores, class_ids = probabilities.topk(TOP_K)
```

## How This Repo Loads It

`src/load_model.py` returns the pretrained Torchvision model in evaluation
mode:

```python
from src.load_model import load_model

model = load_model("inception_v3")
print(type(model))
print(model.training)  # False
```

The loader sets `TORCH_HOME` to `models/cnn`, so the checkpoint is stored
under:

```text
models/cnn/hub/checkpoints/
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model("inception_v3")
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> Inception_V3_Weights.DEFAULT.transforms()
  -> model(input_tensor)
```

## Inference Example

```python
from pathlib import Path

import torch
from torchvision.models import Inception_V3_Weights

from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image

image_path = Path("datasets/coco/images/000000047585.jpg")
weights = Inception_V3_Weights.DEFAULT
model = load_model("inception_v3").eval()
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

In evaluation mode, InceptionV3 returns logits shaped `batch_size x 1000`. For
one image, the shape is `1 x 1000`.

The demo prints rank, ImageNet label, softmax probability, and class ID. COCO
images can contain multiple objects, so the result should be interpreted as
whole-image classification rather than detection.

## Quick Demo

Run directly:

```bash
uv run python docs/models/inception/demo.py
```

Or use the shared script:

```bash
./scripts/inference_demo.sh inception
```

The demo is configured through constants at the top of `demo.py`:

```python
MODEL_NAMES = ["inception_v3"]
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

Generated images and JSON files are written under
`docs/models/inception/outputs/`, which is ignored by Git.

## Performance Benchmark

```bash
uv run python experiments/performance/system_performance.py \
  --model inception_v3 \
  --device cpu \
  --num-images 100
```

Use `--device cuda:0` on a CUDA system. Results are appended to
`results/system_metrics/system_performance_results.csv`.

## Sources

- [Torchvision InceptionV3 documentation](https://docs.pytorch.org/vision/stable/models/inception.html)
- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
