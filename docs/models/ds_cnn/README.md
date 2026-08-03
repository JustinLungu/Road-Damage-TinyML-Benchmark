# DS-CNN Classifiers

This folder documents `ds_cnn_small`, a custom local depthwise-separable CNN
added as a tiny image-classification baseline for the RDD2022 binary pothole
benchmark.

| Repo name | Checkpoint location | Family | Parameters | FP32 parameter size | Size bucket |
| --- | --- | --- | ---: | ---: | --- |
| `ds_cnn_small` | none; source-defined model | custom depthwise-separable CNN | 53,152 | approx. 0.20 MiB / 0.21 MB | tiny |

`ds_cnn_small` has no pretrained checkpoint. It is instantiated directly from
`src/custom_models.py`, so `./scripts/download_models.sh ds_cnn_small` only
checks that the model can be constructed. Nothing is downloaded.

The model is registered in:

```python
CUSTOM_IMAGE_CLASSIFICATION_MODELS = frozenset(
    {
        "ds_cnn_small",
        "resnet8",
        "tiny_cnn",
    }
)
```

and loaded through:

```python
load_ds_cnn_small()
```

For definitions of terms such as depthwise convolution, pointwise convolution,
depthwise-separable block, classifier head, logits, and FP32 size, see
[definitions.md](definitions.md).

## Why DS-CNN-Small Exists

`ds_cnn_small` is the smallest custom classifier currently in the repository.
It tests whether a very lightweight depthwise-separable CNN can learn the
cleaned and balanced RDD binary pothole task.

It gives a different tiny-model hypothesis than the other custom models:

- `tiny_cnn`: plain compact convolutional baseline;
- `resnet8`: compact residual CNN;
- `ds_cnn_small`: compact depthwise-separable CNN.

The architecture is intentionally small:

```text
image tensor
  -> convolutional stem
  -> depthwise-separable block, 16 -> 24 channels
  -> depthwise-separable block, 24 -> 32 channels
  -> depthwise-separable block, 32 -> 48 channels
  -> adaptive average pooling
  -> linear classifier
  -> logits
```

Because it starts from random weights, it should be treated differently from
the pretrained models. Its useful test is whether the RDD benchmark can train
a very small edge-friendly classifier from scratch.

## Internal Flow

At inference time, `ds_cnn_small` expects a normalized RGB image tensor shaped:

```text
batch x 3 x 224 x 224
```

Each depthwise-separable block first applies one spatial convolution per input
channel, then uses a 1x1 pointwise convolution to mix channels. This is much
cheaper than a standard convolution with the same input/output channels.

After feature extraction, adaptive average pooling collapses the final feature
map to one 48-channel vector per image. The classifier maps those 48 features
to class logits.

For normal model loading, `ds_cnn_small` is created with 1000 output classes so
it behaves like the other image-classification models. For RDD fine-tuning, the
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
./scripts/download_models.sh ds_cnn_small
```

Run the documentation demo:

```bash
./scripts/inference_demo.sh --model ds_cnn
```

Use it in the RDD benchmark by setting:

```python
RDD_MODEL_NAMES = ("ds_cnn_small",)
```

## Output Interpretation

Before RDD fine-tuning, the output logits are random and should not be used as
meaningful predictions. After RDD fine-tuning, the two logits represent the
binary pothole classes used throughout the benchmark.
