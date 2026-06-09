# EfficientFormer Classifiers

This folder documents the timm EfficientFormer classifiers used in the repository:

| Repo name | timm model ID | Parameters | Input size | Stage block counts |
| --- | --- | --- | --- | --- |
| `efficientformer_l1` | `efficientformer_l1.snap_dist_in1k` | 12.29M | 224x224 | `[3, 2, 6, 5]` |
| `efficientformer_l3` | `efficientformer_l3.snap_dist_in1k` | 31.41M | 224x224 | `[4, 4, 12, 7]` |
| `efficientformer_l7` | `efficientformer_l7.snap_dist_in1k` | 82.23M | 224x224 | `[6, 6, 18, 9]` |

All three are ImageNet-1K pretrained image classifiers with 1000 output classes. In this project they are useful efficient-transformer baselines for system-performance benchmarks.

The parameter counts and stage block counts above come from the locally loaded timm models.

For definitions of terms such as timm, stem, stage, `MetaBlock2d`, token mixer, pooling token mixer, layer scale, `MetaBlock1d`, attention, logits, softmax, and top-k, see [definitions.md](definitions.md).

The repo registry names are defined in `src/constants.py`:

```python
EFFICIENTFORMER_MODEL_IDS = {
    "efficientformer_l1": "efficientformer_l1.snap_dist_in1k",
    "efficientformer_l3": "efficientformer_l3.snap_dist_in1k",
    "efficientformer_l7": "efficientformer_l7.snap_dist_in1k",
}
```

## What Makes EfficientFormer Different

EfficientFormer is designed to get transformer-like representation quality while keeping much of the speed profile of efficient CNNs.

The timm models used here are not pure Vision Transformers. They keep most processing in 2D feature maps and use efficient pooling-based token mixing through most of the network. Attention appears later, after the image has been reduced to a smaller representation.

The local variants mainly differ by size and depth:

- `efficientformer_l1` is the smallest and cheapest;
- `efficientformer_l3` is larger and deeper;
- `efficientformer_l7` is the largest local variant.

Like MobileNetV3 and MobileViT, EfficientFormer is not an object detector. It does not return boxes. It returns one 1000-class ImageNet score vector for the whole image. The top-k output should be read as "which ImageNet labels best describe the full image crop?", not "which objects were detected and where are they?"

## Internal Flow

At inference time, the demo follows this broad path:

```text
image file
  -> timm preprocessing transform
  -> 224x224 normalized tensor
  -> Stem4 convolutional stem
  -> four EfficientFormer stages
  -> mostly 2D MetaBlocks with pooling token mixers
  -> Flat transition to 1D tokens
  -> later 1D attention MetaBlocks
  -> classification head
  -> logits
  -> softmax probabilities
  -> top-k labels
```

### 1. Preprocessing

The demo loads the image as RGB and asks timm for the exact preprocessing config associated with the checkpoint:

```python
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
input_tensor = transform(image).unsqueeze(0)
```

For the locally loaded EfficientFormer models, timm resolves:

- input size: `3 x 224 x 224`;
- interpolation: `bicubic`;
- crop percentage: `0.95`;
- mean: `[0.485, 0.456, 0.406]`;
- std: `[0.229, 0.224, 0.225]`.

The original COCO image can have any size. The transform converts it into the normalized 224x224 tensor expected by the checkpoint.

### 2. Stem4

The first module is `Stem4`.

In the local timm model, this stem uses two stride-2 convolutions. That reduces spatial resolution by 4x before the main stages, while increasing the channel count.

### 3. EfficientFormer Stages

The model then runs four stages.

Each stage contains multiple blocks. Between stages, downsampling reduces spatial resolution and increases channels. This lets later blocks work with richer features at lower spatial cost.

The stage depths grow with model size:

| Repo name | Stage block counts |
| --- | --- |
| `efficientformer_l1` | `[3, 2, 6, 5]` |
| `efficientformer_l3` | `[4, 4, 12, 7]` |
| `efficientformer_l7` | `[6, 6, 18, 9]` |

### 4. 2D Meta Blocks

Most blocks are `MetaBlock2d` blocks.

In the local timm implementation, these blocks keep the representation as a 2D feature map. Each block contains:

- a pooling token mixer;
- residual connections;
- layer scale;
- a convolutional MLP with normalization.

The pooling token mixer uses average pooling to mix nearby spatial information cheaply. This is less expressive than full attention, but it is much cheaper and fits the EfficientFormer goal of fast inference.

### 5. Flat and 1D Attention Blocks

Near the end, the model uses a `Flat` transition to move from 2D feature maps into 1D tokens.

After that, it uses `MetaBlock1d` blocks with attention:

| Repo name | `MetaBlock1d` count |
| --- | --- |
| `efficientformer_l1` | 1 |
| `efficientformer_l3` | 4 |
| `efficientformer_l7` | 8 |

This puts the more expensive attention work late in the network, after spatial resolution has already been reduced.

### 6. Classification Head

The final representation goes through normalization and a classification head.

The local timm models include both `head` and `head_dist`, reflecting the pretrained checkpoint style. During normal evaluation, the model call returns one logits tensor for 1000 ImageNet classes.

The demo applies softmax to convert logits into probabilities, then prints the top-k classes:

```python
probabilities = logits.softmax(dim=1)[0]
scores, class_ids = probabilities.topk(TOP_K)
```

## How This Repo Loads Them

`src/load_model.py` loads EfficientFormer through timm:

```python
from src.load_model import load_model

model = load_model("efficientformer_l1")
print(type(model))
```

Internally the loader calls:

```python
timm.create_model(
    "efficientformer_l1.snap_dist_in1k",
    pretrained=True,
).eval()
```

Before creating the model, the repo sets:

```python
os.environ["HF_HOME"] = str(VIT_DIR / local_name)
os.environ["TIMM_HOME"] = str(VIT_DIR / local_name)
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model(model_name)
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> timm.data.resolve_model_data_config(model)
  -> timm.data.create_transform(...)
  -> model(input_tensor)
```

## Inference Example

```python
from pathlib import Path

import timm
import torch
from timm.data import ImageNetInfo

from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image

model_name = "efficientformer_l1"
image_path = Path("datasets/coco/images/000000047585.jpg")
device = torch.device("cpu")

model = load_model(model_name).to(device).eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
labels = [ImageNetInfo().index_to_description(i) for i in range(1000)]

image = load_rgb_image(image_path)
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.inference_mode():
    logits = model(input_tensor)
    probabilities = logits.softmax(dim=1)[0]
    scores, class_ids = probabilities.topk(5)

for score, class_id in zip(scores, class_ids):
    class_id_int = int(class_id)
    print(labels[class_id_int], float(score))
```

## Reading The Output

EfficientFormer returns logits shaped:

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
uv run python docs/models/efficientformer/demo.py
```

The demo is intentionally configured by constants at the top of `demo.py`, so edit these values before running:

```python
MODEL_NAMES = ["efficientformer_l1", "efficientformer_l3", "efficientformer_l7"]
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

Set `MODEL_NAMES` to one or more supported names. For example, `["efficientformer_l1"]` runs only the smallest model, while `["efficientformer_l1", "efficientformer_l3", "efficientformer_l7"]` runs all three models on the same selected image.

Set `USE_RANDOM_IMAGE = False` to use `IMAGE_PATH`. Set `USE_RANDOM_IMAGE = True` to ignore `IMAGE_PATH` and sample a random `.jpg` from `RANDOM_IMAGE_DIR`. Set `RANDOM_SEED` to an integer if you want the same random image across runs.

When `SAVE_RESULTS_JSON = True`, each model writes a top-k JSON file under:

```text
docs/models/efficientformer/outputs/
```

When `SAVE_INPUT_IMAGE = True`, the selected source image is also copied to:

```text
docs/models/efficientformer/outputs/selected_image.jpg
```

This image is the visual reference for the top-k whole-image classifications. It is not annotated with boxes because EfficientFormer does not predict object locations.

## Benchmark Commands

The system-performance benchmark measures latency, FPS, RAM, GPU metrics, power, and energy per inference. It does not report classification accuracy.

```bash
uv run python experiments/performance/system_performance.py \
  --model efficientformer_l1 \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model efficientformer_l3 \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model efficientformer_l7 \
  --device cuda:0 \
  --num-images 100
```
