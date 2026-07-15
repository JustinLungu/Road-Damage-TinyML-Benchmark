# ShuffleNet Classifiers

This folder documents the ShuffleNet models used in the repository.

| Repo name | Checkpoint location | Family | Parameters | FLOPs | ImageNet-1K acc@1 / acc@5 | Size bucket |
| --- | --- | --- | ---: | ---: | --- | --- |
| `shufflenet_v2_x0_5` | `models/cnn/hub/checkpoints/shufflenetv2_x0.5-f707e7126e.pth` | Torchvision ShuffleNetV2 0.5x | 1.37M | 0.040B | not listed in local metadata | small |

`shufflenet_v2_x0_5` is a Torchvision ImageNet-1K pretrained image
classifier. Unlike the custom tiny models, it has a real pretrained checkpoint.
Running:

```bash
./scripts/download_models.sh shufflenet_v2_x0_5
```

downloads/caches the checkpoint under:

```text
models/cnn/hub/checkpoints/
```

The checkpoint path is registered in `src/constants.py`:

```python
SHUFFLENET_MODEL_CHECKPOINTS = {
    "shufflenet_v2_x0_5": "shufflenetv2_x0.5-f707e7126e.pth",
}
```

For definitions of terms such as channel shuffle, grouped convolution,
depthwise convolution, pointwise convolution, logits, top-k, parameters, and
FLOPs, see [definitions.md](definitions.md).

## Why ShuffleNetV2-0.5x Is Included

ShuffleNetV2 is designed for efficient inference on mobile and edge hardware.
The 0.5x variant is a small pretrained classifier, so it is useful as a
comparison point against:

- custom tiny models trained from scratch;
- small pretrained models such as MobileViT-XS;
- larger high-performing models such as EfficientFormer.

It is not below 1 MB, so it is not a tiny model. It fits the small-model bucket.

## Internal Flow

At inference time, ShuffleNetV2 follows this broad path:

```text
image file
  -> Torchvision preprocessing
  -> convolutional stem
  -> ShuffleNetV2 stages
  -> channel split and channel shuffle operations
  -> global pooling
  -> fully connected classifier
  -> logits
```

ShuffleNetV2 improves efficiency by carefully balancing channel widths,
minimizing memory fragmentation, and using channel shuffle to exchange
information across channel groups.

## Project Usage

List available models:

```bash
./scripts/download_models.sh --list
```

Download/load the model:

```bash
./scripts/download_models.sh shufflenet_v2_x0_5
```

Run the documentation demo:

```bash
./scripts/inference_demo.sh --model shufflenet
```

Use it in the RDD benchmark by setting:

```python
RDD_MODEL_MODE = "single"
RDD_SINGLE_MODEL = "shufflenet_v2_x0_5"
```

## Output Interpretation

Before RDD fine-tuning, the model returns ImageNet-1K logits. After RDD
fine-tuning, the benchmark adapter replaces its `fc` classifier with a 2-class
binary pothole head.
