# MobileViT Classifiers

This folder documents the Hugging Face MobileViT classifiers used in the repository:

| Repo name | Hugging Face model ID | Local cache | Approx. local cache size | Hidden sizes |
| --- | --- | --- | --- | --- |
| `mobilevit_xxs` | `apple/mobilevit-xx-small` | `models/vit/mobilevit_xxs/` | 11M | `[64, 80, 96]` |
| `mobilevit_xs` | `apple/mobilevit-x-small` | `models/vit/mobilevit_xs/` | 19M | `[96, 120, 144]` |
| `mobilevit_s` | `apple/mobilevit-small` | `models/vit/mobilevit_s/` | 43M | `[144, 192, 240]` |

All three are ImageNet-1K pretrained image classifiers with 1000 output classes. In this project they are useful lightweight vision-transformer baselines for system-performance benchmarks.

For definitions of terms such as CNN, transformer, self-attention, patch, `pixel_values`, logits, softmax, top-k, hidden size, and classifier head, see [definitions.md](definitions.md).

The repo registry names are defined in `src/constants.py`:

```python
MOBILEVIT_MODEL_IDS = {
    "mobilevit_xxs": "apple/mobilevit-xx-small",
    "mobilevit_xs": "apple/mobilevit-x-small",
    "mobilevit_s": "apple/mobilevit-small",
}
```

## What Makes MobileViT Different

MobileViT is a hybrid between a CNN and a Vision Transformer.

Pure CNNs are efficient and strong at local visual patterns, but their direct view is mostly local. Pure Vision Transformers can mix information globally, but they are often more expensive. MobileViT tries to combine both:

- convolutional layers build efficient local features;
- MobileViT blocks unfold feature maps into patch-like sequences;
- transformer layers mix information across those patches;
- the result is folded back into a feature map for later convolutional processing.

The local variants mainly differ by width:

- `mobilevit_xxs` is the smallest and cheapest;
- `mobilevit_xs` is a middle option;
- `mobilevit_s` is wider and heavier.

Like MobileNetV3, MobileViT is not an object detector. It does not return boxes. It returns one 1000-class ImageNet score vector for the whole image. The top-k output should be read as "which ImageNet labels best describe the full image crop?", not "which objects were detected and where are they?"

## Internal Flow

At inference time, the demo follows this broad path:

```text
image file
  -> Hugging Face image processor
  -> pixel_values tensor
  -> convolutional stem and local CNN blocks
  -> MobileViT global representation blocks
  -> final feature/neck layers
  -> classifier head
  -> logits
  -> softmax probabilities
  -> top-k labels
```

### 1. Preprocessing

The demo loads the image as RGB and passes it through the checkpoint's Hugging Face image processor:

```python
processor = AutoImageProcessor.from_pretrained(...)
inputs = processor(images=image, return_tensors="pt")
```

The local MobileViT processors resize the image to 288 pixels, center crop to 256x256, and convert the image into a batched tensor named `pixel_values`.

This means the original COCO image can have any size. The model still receives a fixed-size tensor. In the local processors, `do_flip_channels = true`, so the processor also applies the channel order expected by these checkpoints.

### 2. Convolutional Stem

The first part of MobileViT uses convolutional layers.

These layers build local visual features such as edges, textures, corners, and small object parts. This keeps the model efficient and preserves the inductive bias that CNNs have for images.

### 3. MobileViT Blocks

A MobileViT block mixes CNN and transformer behavior.

Conceptually, it:

1. uses convolutions to build local features;
2. unfolds the feature map into patch-like tokens;
3. applies transformer layers so patches can exchange information globally;
4. folds the sequence back into a spatial feature map.

This is the main architectural idea that makes MobileViT different from MobileNetV3.

### 4. Neck and Final Features

After several stages, the model builds a compact image representation for classification.

The local configs expose `neck_hidden_sizes`, which describe the channel widths through the later feature-processing layers:

| Repo name | `neck_hidden_sizes` |
| --- | --- |
| `mobilevit_xxs` | `[16, 16, 24, 48, 64, 80, 320]` |
| `mobilevit_xs` | `[16, 32, 48, 64, 80, 96, 384]` |
| `mobilevit_s` | `[16, 32, 64, 96, 128, 160, 640]` |

Wider channels usually increase model capacity, but they also increase memory use and compute.

### 5. Classifier Head

The classifier head maps the final image representation to ImageNet logits.

The demo applies softmax to convert logits into probabilities, then prints the top-k classes:

```python
probabilities = logits.softmax(dim=-1)[0]
scores, class_ids = probabilities.topk(TOP_K)
```

## How This Repo Loads Them

`src/load_model.py` loads MobileViT through Hugging Face Transformers:

```python
from src.load_model import load_model

model = load_model("mobilevit_xxs")
print(type(model))
```

Internally the loader calls:

```python
AutoModelForImageClassification.from_pretrained(
    model_id,
    cache_dir=str(VIT_DIR / local_name),
).eval()
```

The image processor is loaded separately:

```python
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(
    "apple/mobilevit-xx-small",
    cache_dir="models/vit/mobilevit_xxs",
    use_fast=False,
)
```

The benchmark path is:

```text
experiments/performance/system_performance.py
  -> src.load_model.load_model(model_name)
  -> src.performance_benchmark.PerformanceBenchmark
  -> src.performance_benchmark.model_inference_adapter.ModelInferenceAdapter
  -> AutoImageProcessor.from_pretrained(...)
  -> processor(images=image, return_tensors="pt")
  -> model(**inputs)
```

## Inference Example

```python
from pathlib import Path

import torch
from transformers import AutoImageProcessor

from src.constants import MOBILEVIT_MODEL_IDS, VIT_DIR
from src.load_model import load_model
from src.performance_benchmark.utils import load_rgb_image, move_inputs_to_device

model_name = "mobilevit_xxs"
image_path = Path("datasets/coco/images/000000047585.jpg")
device = torch.device("cpu")

processor = AutoImageProcessor.from_pretrained(
    MOBILEVIT_MODEL_IDS[model_name],
    cache_dir=str(VIT_DIR / model_name),
    use_fast=False,
)
model = load_model(model_name).to(device).eval()

image = load_rgb_image(image_path)
inputs = processor(images=image, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**move_inputs_to_device(inputs, device))
    probabilities = outputs.logits.softmax(dim=-1)[0]
    scores, class_ids = probabilities.topk(5)

for score, class_id in zip(scores, class_ids):
    class_id_int = int(class_id)
    print(model.config.id2label[class_id_int], float(score))
```

## Reading The Output

MobileViT returns logits shaped:

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
uv run python docs/models/mobilevit/demo.py
```

The demo is intentionally configured by constants at the top of `demo.py`, so edit these values before running:

```python
MODEL_NAMES = ["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"]
IMAGE_PATH = REPO_ROOT / "datasets" / "coco" / "images" / "000000047585.jpg"
USE_RANDOM_IMAGE = True
RANDOM_IMAGE_DIR = REPO_ROOT / "datasets" / "coco" / "images"
RANDOM_SEED = None
DEVICE_NAME = "cpu"
TOP_K = 5
LOCAL_FILES_ONLY = True
SAVE_RESULTS_JSON = True
SAVE_INPUT_IMAGE = True
OUTPUT_DIR = MODEL_DOCS_DIR / "outputs"
```

Set `MODEL_NAMES` to one or more supported names. For example, `["mobilevit_xxs"]` runs only the smallest model, while `["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"]` runs all three models on the same selected image.

Set `USE_RANDOM_IMAGE = False` to use `IMAGE_PATH`. Set `USE_RANDOM_IMAGE = True` to ignore `IMAGE_PATH` and sample a random `.jpg` from `RANDOM_IMAGE_DIR`. Set `RANDOM_SEED` to an integer if you want the same random image across runs.

When `SAVE_RESULTS_JSON = True`, each model writes a top-k JSON file under:

```text
docs/models/mobilevit/outputs/
```

When `SAVE_INPUT_IMAGE = True`, the selected source image is also copied to:

```text
docs/models/mobilevit/outputs/selected_image.jpg
```

This image is the visual reference for the top-k whole-image classifications. It is not annotated with boxes because MobileViT does not predict object locations.

## Benchmark Commands

The system-performance benchmark measures latency, FPS, RAM, GPU metrics, power, and energy per inference. It does not report classification accuracy.

```bash
uv run python experiments/performance/system_performance.py \
  --model mobilevit_xxs \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model mobilevit_xs \
  --device cuda:0 \
  --num-images 100
```

```bash
uv run python experiments/performance/system_performance.py \
  --model mobilevit_s \
  --device cuda:0 \
  --num-images 100
```
