# InceptionV3 Definitions

This glossary explains the terms used in the InceptionV3 model notes.

## Image Classification

Image classification assigns a label to a whole image. InceptionV3 returns
scores for 1000 ImageNet classes but does not return bounding boxes.

## ImageNet-1K

ImageNet-1K is an image-classification dataset with 1000 classes. The local
InceptionV3 checkpoint is pretrained on this dataset.

## Inception Module

An Inception module processes one input through several branches in parallel.
Each branch uses a different sequence of convolutions or pooling operations.
Their output feature maps are concatenated along the channel dimension.

## Parallel Branch

A parallel branch is one independent path through a module. Inception branches
can focus on different spatial scales and transformations at the same stage.

Unlike ResNet shortcuts, Inception branch outputs are concatenated rather than
added.

## Concatenation

Concatenation joins tensors along one dimension. In an Inception module, branch
outputs have matching spatial dimensions and are joined along the channel
dimension.

If branches produce 64, 64, 96, and 32 channels, concatenation produces 256
output channels.

## Channel Reduction

Channel reduction uses a 1x1 convolution to reduce channel count before a more
expensive spatial convolution. This limits computation while still letting the
network learn how channels should be combined.

## Factorized Convolution

Factorized convolution replaces one expensive convolution with multiple
cheaper convolutions.

InceptionV3 examples include replacing a 5x5 convolution with two 3x3
convolutions and replacing a 7x7 convolution with a 1x7 convolution followed
by a 7x1 convolution.

## Asymmetric Convolution

An asymmetric convolution has different height and width, such as 1x7 or 7x1.
Combining the two orientations approximates a square 7x7 receptive field with
less computation.

## Receptive Field

The receptive field is the region of the original input that can affect a
feature value. Larger or stacked convolutions give deeper features a larger
receptive field.

## Grid Reduction

Grid reduction decreases feature-map width and height. InceptionV3 performs
downsampling through modules with parallel stride-2 convolution and pooling
branches.

The term "grid" refers to the spatial feature grid.

## Auxiliary Classifier

An auxiliary classifier is an extra classification head attached to an
intermediate feature map. It can provide another training signal to earlier
layers.

Torchvision's pretrained InceptionV3 contains this head. It is inactive when
the model is in evaluation mode, so repository inference returns only the main
logits.

## Evaluation Mode

Calling `model.eval()` switches modules such as dropout and batch normalization
to inference behavior. For InceptionV3, evaluation mode also disables the
auxiliary output.

## Global Average Pooling

Global average pooling averages each final feature channel across its spatial
positions. InceptionV3 uses it to create a 2048-value vector before the final
classifier.

## Dropout

Dropout randomly sets some feature values to zero during training as
regularization. It is disabled in evaluation mode.

## Logits

Logits are raw classifier output values before probabilities are computed.

## Softmax

Softmax converts logits into class probabilities that sum to one:

```python
probabilities = logits.softmax(dim=1)
```

## Top-K

Top-k means the `k` highest-scoring classes. With `TOP_K = 5`, the demo prints
the five largest ImageNet probabilities.

## Parameters

Parameters are learned model weights. Torchvision reports about 27.16 million
parameters for the pretrained InceptionV3 model, including its auxiliary
classifier.

## FLOPs

FLOPs estimates the floating-point operations required for one forward pass.
Torchvision reports about 5.713 billion operations for InceptionV3 at its
default 299x299 input size.

Actual speed also depends on hardware, memory movement, and operator
implementation.
