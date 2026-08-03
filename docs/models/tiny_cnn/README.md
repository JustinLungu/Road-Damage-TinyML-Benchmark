# TinyCNN Classifier

This folder documents `tiny_cnn`, a custom local CNN added as the first true
tiny image-classification baseline for the RDD2022 binary pothole benchmark.

| Repo name | Checkpoint location | Family | Parameters | FP32 parameter size | Size bucket |
| --- | --- | --- | ---: | ---: | --- |
| `tiny_cnn` | none; source-defined model | custom CNN | 88,696 | approx. 0.34 MiB / 0.35 MB | tiny |

`tiny_cnn` has no pretrained checkpoint. It is instantiated directly from
`src/custom_models.py`, so `./scripts/download_models.sh tiny_cnn` only checks
that the model can be constructed. Nothing is downloaded.

The model is registered in:

```python
CUSTOM_IMAGE_CLASSIFICATION_MODELS = frozenset(
    {
        "tiny_cnn",
    }
)
```

and loaded through:

```python
load_tiny_cnn()
```

For definitions of terms such as logits, feature extractor, global average
pooling, classifier head, parameters, and FP32 size, see
[definitions.md](definitions.md).

## Why TinyCNN Exists

The previous RDD image-classification candidates included small models such as
`mobilevit_xxs` and `mobilevit_xs`, but none of them were below 1 MB. The
supervisor requirement asks for at least one tiny model below 1 MB and one
small model between 1-10 MB.

`tiny_cnn` fills the tiny-model slot. It is intentionally simple and compact:

```text
image tensor
  -> 3 convolution blocks
  -> adaptive average pooling
  -> linear classifier
  -> logits
```

Because it starts from random weights, it should be treated differently from
the pretrained models. Its value is not ImageNet transfer learning; its value
is testing whether a very small custom model can learn the cleaned and balanced
RDD binary pothole task.

## Internal Flow

At inference time, `tiny_cnn` expects a normalized RGB image tensor shaped:

```text
batch x 3 x 224 x 224
```

The model applies three convolutional blocks. Each block uses:

```text
Conv2d -> BatchNorm2d -> ReLU
```

The spatial resolution is reduced by stride-2 convolutions. After the final
block, adaptive average pooling collapses the feature map to one 64-channel
feature vector per image. The classifier then maps those 64 features to the
configured number of output classes.

For normal model loading, `tiny_cnn` is created with 1000 output classes so it
behaves like the other image-classification models. For RDD fine-tuning, the
benchmark adapter replaces the classifier with a 2-class head:

```text
non_pothole / pothole
```

## Project Usage

List available models:

```bash
./scripts/download_models.sh --list
```

Load the model:

```bash
./scripts/download_models.sh tiny_cnn
```

Run the documentation demo:

```bash
./scripts/inference_demo.sh --model tiny_cnn
```

Use it in the RDD benchmark by setting:

```python
RDD_MODEL_NAMES = ("tiny_cnn",)
```

## Output Interpretation

Before RDD fine-tuning, the output logits are random and should not be used as
meaningful predictions. After RDD fine-tuning, the two logits represent the
binary pothole classes used throughout the benchmark.
