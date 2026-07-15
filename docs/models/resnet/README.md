# ResNet Classifiers

This folder documents the ResNet-family classifiers used in the repository:

| Repo name | Checkpoint location | Family | Parameters | FLOPs | ImageNet-1K acc@1 / acc@5 | Size bucket |
| --- | --- | --- | ---: | ---: | --- | --- |
| `resnet18` | `models/cnn/hub/checkpoints/resnet18-f37072fd.pth` | Torchvision ResNet18 | 11.69M | 1.814B | 69.758 / 89.078 | medium |
| `resnet8` | none; source-defined model | custom residual CNN | 111,928 | not measured | not pretrained | tiny |

ResNet18 is an ImageNet-1K pretrained classifier with 1000 output classes. It
is included as a conventional CNN baseline against which the lighter
MobileNetV2 and EfficientNet-B0 models can be compared.

ResNet8 is a custom local residual CNN. It has no pretrained checkpoint and is
instantiated directly from `src/custom_models.py`. It exists as a true tiny
model candidate for RDD binary pothole classification.

The ResNet18 parameter count, FLOPs, accuracy, preprocessing transform, and
class labels come from Torchvision's default `ResNet18_Weights`. The ResNet8
parameter count is computed from the local architecture.

For definitions of residual learning, skip connection, BasicBlock, identity
mapping, downsampling, logits, softmax, top-k, parameters, and FLOPs, see
[definitions.md](definitions.md).

The ResNet18 checkpoint path is registered in `src/constants.py`:

```python
RESNET_MODEL_CHECKPOINTS = {
    "resnet18": "resnet18-f37072fd.pth",
}
```

The custom ResNet8 model is registered in:

```python
CUSTOM_IMAGE_CLASSIFICATION_MODELS = frozenset(
    {
        "resnet8",
        "tiny_cnn",
    }
)
```

## Why ResNet18 Is A Useful Baseline

ResNet introduced residual learning: instead of requiring every stack of
layers to learn a complete transformation, a block learns a residual update
that is added to its input.

This architecture is widely understood, broadly supported, and easy to compare
with results from other projects. ResNet18 is the smallest standard ResNet in
the original family. It is heavier than the local MobileNetV2 and
EfficientNet-B0 models in parameter count and FLOPs, but provides a clean
reference architecture without mobile-specific operators.

ResNet18 is not an object detector. It returns one 1000-class score vector for
the whole image and does not return bounding boxes.

## Why ResNet8 Is Included

ResNet8 provides a tiny residual alternative to `tiny_cnn`. It is below 1 MB
before quantization and therefore satisfies the tiny-model bucket. Unlike
ResNet18, it is not pretrained. Its useful test is whether a very small
residual model can learn from the cleaned and balanced RDD2022 binary pothole
dataset.

The ResNet8 flow is:

```text
image tensor
  -> convolutional stem
  -> residual block at 16 channels
  -> downsample to 32 channels
  -> residual block at 32 channels
  -> downsample to 64 channels
  -> adaptive average pooling
  -> fully connected classifier
  -> logits
```

## Internal Flow

The ResNet18 inference path is:

```text
image file
  -> Torchvision preprocessing
  -> 7x7 convolutional stem
  -> max pooling
  -> four residual stages
  -> eight BasicBlocks
  -> global average pooling
  -> fully connected classifier
  -> logits
  -> softmax probabilities
  -> top-k labels
```

### 1. Preprocessing

The demo applies the exact transform attached to the pretrained weight:

```python
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()
input_tensor = transform(image).unsqueeze(0)
```

The default transform:

- resizes the image to 256 using bilinear interpolation;
- center crops it to 224x224;
- converts it to a tensor;
- normalizes RGB with ImageNet mean `[0.485, 0.456, 0.406]`;
- normalizes RGB with ImageNet standard deviation
  `[0.229, 0.224, 0.225]`.

The model receives a batched tensor shaped `1 x 3 x 224 x 224`.

### 2. Stem

The stem starts with a stride-2 7x7 convolution that produces 64 channels.
Batch normalization, ReLU, and a stride-2 max-pooling layer reduce the spatial
resolution before the residual stages.

### 3. Residual Stages

ResNet18 has four stages with two BasicBlocks per stage:

| Stage | Blocks | Output channels |
| --- | --- | --- |
| `layer1` | 2 | 64 |
| `layer2` | 2 | 128 |
| `layer3` | 2 | 256 |
| `layer4` | 2 | 512 |

Each BasicBlock contains two 3x3 convolutions. Its input is added to the output
of the second convolution before the final ReLU:

```text
output = ReLU(residual_branch(input) + shortcut(input))
```

When the spatial size and channel count stay unchanged, the shortcut is the
identity. The first block of stages 2-4 downsamples with stride 2 and uses a
1x1 convolution in the shortcut so both branches have matching shapes.

### 4. Classification Head

Adaptive global average pooling reduces the final 512-channel feature map to
one value per channel. A fully connected layer maps the resulting vector to
1000 ImageNet logits.

The demo converts logits to probabilities and selects the top-k classes:

```python
probabilities = logits.softmax(dim=1)[0]
scores, class_ids = probabilities.topk(TOP_K)
```

## How This Repo Loads It

`src/load_model.py` returns either the pretrained Torchvision model or the
source-defined custom model:

```python
from src.load_model import load_model

model = load_model("resnet18")
print(type(model))

tiny_model = load_model("resnet8")
print(type(tiny_model))
```

The loader sets `TORCH_HOME` to `models/cnn`, so Torchvision stores the
checkpoint under:

```text
models/cnn/hub/checkpoints/
```

ResNet8 does not create a file under `models/`; it is created from source code.
After RDD training, its learned checkpoint is saved under
`results/rdd_trained_models/<experiment_name>/resnet8/best.pt`.

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model("resnet18")
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> ResNet18_Weights.DEFAULT.transforms()
  -> model(input_tensor)
```

## Inference Example

```python
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights

from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image

image_path = Path("datasets/coco/images/000000047585.jpg")
weights = ResNet18_Weights.DEFAULT
model = load_model("resnet18").eval()
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

Both ResNet demos return logits shaped `batch_size x 1000`. For one image, the shape is
`1 x 1000`.

For ResNet18, the demo prints rank, ImageNet label, softmax probability, and
class ID. COCO images may contain several objects, so the top label should be
interpreted as whole-image classification rather than detection.

For ResNet8, the logits are random until the model is fine-tuned. The demo can
still run to confirm shape and inference plumbing, but the printed labels should
not be interpreted as meaningful predictions.

## Quick Demo

Run directly:

```bash
uv run python docs/models/resnet/demo.py
```

Or use the shared script:

```bash
./scripts/inference_demo.sh resnet
```

The demo is configured through constants at the top of `demo.py`:

```python
MODEL_NAMES = ["resnet18"]
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

Available demo options:

```python
MODEL_NAMES = ["resnet18"]            # pretrained Torchvision ResNet18
MODEL_NAMES = ["resnet8"]             # custom tiny ResNet8, untrained
MODEL_NAMES = ["resnet18", "resnet8"] # run both
```

Generated images and JSON files are written under
`docs/models/resnet/outputs/`, which is ignored by Git.

## Performance Benchmark

```bash
uv run python experiments/performance/system_performance.py \
  --model resnet18 \
  --device cpu \
  --num-images 100
```

Use `--device cuda:0` on a CUDA system. Results are appended to
`results/system_metrics/system_performance_results.csv`.

## Sources

- [Torchvision ResNet documentation](https://docs.pytorch.org/vision/stable/models/resnet.html)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
