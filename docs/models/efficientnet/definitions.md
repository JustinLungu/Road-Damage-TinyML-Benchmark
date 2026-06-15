# EfficientNet Definitions

This glossary explains the terms used in the EfficientNet-B0 model notes.

## Image Classification

Image classification assigns a label to a whole image. It does not return
bounding boxes or object locations.

EfficientNet-B0 produces one score for each of 1000 ImageNet classes. This is
different from YOLO, which detects individual objects and returns their boxes.

## ImageNet-1K

ImageNet-1K is an image-classification dataset with 1000 classes. The local
EfficientNet-B0 checkpoint is pretrained on ImageNet-1K, so its output vector
contains 1000 values.

## Compound Scaling

Compound scaling increases model depth, channel width, and input resolution
together using a coordinated scaling rule.

The EfficientNet family starts from B0 and creates larger variants by scaling
these dimensions together instead of enlarging only one dimension.

## MBConv

MBConv means mobile inverted bottleneck convolution. It is the main block used
by EfficientNet-B0.

A typical MBConv expands the channels, applies a depthwise spatial convolution,
uses squeeze-and-excitation, and projects back to a narrower output. A residual
connection is used when the input and output shapes match.

## Inverted Bottleneck

An inverted bottleneck is narrow at its input and output but wider in the
middle. The expanded middle representation gives the depthwise convolution
more channels in which to process features.

## Depthwise Convolution

A depthwise convolution applies one spatial filter per input channel. A
standard convolution mixes spatial and channel information together, while a
depthwise convolution handles the spatial part more cheaply.

MBConv combines depthwise convolution with 1x1 pointwise convolutions that
expand, mix, and project channels.

## Pointwise Convolution

A pointwise convolution is a 1x1 convolution. It changes and mixes feature
channels without using a larger spatial kernel.

EfficientNet uses pointwise convolutions to expand and project features inside
MBConv blocks.

## Squeeze-and-Excitation

Squeeze-and-excitation, or SE, reweights feature channels using global context.

It averages each channel over the spatial dimensions, learns channel
importance values, and multiplies the original features by those values. This
lets the model emphasize useful channels and suppress less useful ones.

## SiLU

SiLU is the sigmoid linear unit activation, also known as swish:

```text
SiLU(x) = x * sigmoid(x)
```

Unlike ReLU, SiLU is smooth and can keep small negative values. EfficientNet-B0
uses SiLU throughout its stem, MBConv blocks, and final convolution.

## Residual Connection

A residual connection adds a block's input to its output when their shapes
match. It gives information and gradients a direct path through the network.

## Stochastic Depth

Stochastic depth randomly skips residual branches during training. It acts as
regularization and can improve generalization.

It is disabled when the model is in evaluation mode, which is the mode used by
the repository's loader, demo, and benchmark.

## Global Average Pooling

Global average pooling averages each feature channel across all spatial
positions. It converts the final feature map into one feature vector before
classification.

This removes explicit location information, which is why the model classifies
the whole crop rather than locating objects.

## Logits

Logits are the raw classifier outputs before probabilities are computed. A
larger logit means the model prefers that class more strongly.

## Softmax

Softmax converts logits into probabilities that sum to one:

```python
probabilities = logits.softmax(dim=1)
```

## Top-K

Top-k means the `k` highest-scoring classes. With `TOP_K = 5`, the demo prints
the five ImageNet classes with the largest probabilities.

## Parameters

Parameters are learned model weights. Torchvision reports about 5.29 million
parameters for EfficientNet-B0.

## FLOPs

FLOPs estimates the number of floating-point operations required for one
forward pass. Torchvision reports about 0.386 billion operations for
EfficientNet-B0 at its default input size.

Fewer FLOPs often helps inference efficiency, but actual speed also depends on
hardware, operator support, memory movement, and the inference library.
