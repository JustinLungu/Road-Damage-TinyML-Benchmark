# ResNet Definitions

This glossary explains the terms used in the ResNet model notes.

## Image Classification

Image classification assigns a label to a whole image. The ResNet demos output
one score for each of 1000 classes. They do not return bounding boxes or object
locations.

## ImageNet-1K

ImageNet-1K is an image-classification dataset with 1000 classes. The local
ResNet18 checkpoint is pretrained on this dataset. ResNet8 is not pretrained.

## Residual Learning

Residual learning asks a block to learn a change to its input rather than an
entire new representation.

If the desired transformation is `H(x)`, the residual branch learns:

```text
F(x) = H(x) - x
```

The block reconstructs the desired output by adding the input:

```text
H(x) = F(x) + x
```

This makes deep networks easier to optimize because information and gradients
can travel through the direct shortcut path.

## Residual Connection

A residual connection adds a block's shortcut branch to its learned residual
branch before activation.

It is also called a skip connection because the shortcut lets information skip
the block's convolutional layers.

## Identity Mapping

An identity mapping returns its input unchanged. ResNet18 uses an identity
shortcut when a block's input and output have the same shape.

## BasicBlock

BasicBlock is the residual block used by ResNet18 and ResNet34. It contains two
3x3 convolutions with batch normalization. The shortcut is added after the
second normalization, followed by ReLU.

Larger ResNets such as ResNet50 use a different three-convolution bottleneck
block.

ResNet8 uses a smaller custom residual block with the same broad idea: two
3x3 convolutions are added back to the input feature map.

## Downsampling

Downsampling reduces spatial resolution. The first block of ResNet18 stages
2-4 uses stride 2 in its residual branch.

Its shortcut also uses a stride-2 1x1 convolution to reduce spatial size and
increase the channel count so the two branches can be added.

## Convolution

A convolution applies learned filters across an image or feature map.
ResNet18's BasicBlocks use standard 3x3 convolutions rather than the depthwise
convolutions used by MobileNet and EfficientNet.

## Batch Normalization

Batch normalization normalizes intermediate channel values and applies learned
scale and shift parameters. ResNet18 places batch normalization after each
convolution.

## ReLU

ReLU is the rectified linear unit activation:

```text
ReLU(x) = max(0, x)
```

It replaces negative values with zero while keeping positive values.

## Global Average Pooling

Global average pooling averages each final feature channel over its spatial
positions. ResNet18 uses it to produce a 512-value vector before its fully
connected classifier. ResNet8 uses adaptive average pooling to produce a
64-value vector.

## Fully Connected Layer

A fully connected, or linear, layer maps every input feature to every output.
ResNet18's final layer maps 512 pooled features to 1000 ImageNet logits.
ResNet8's final layer maps 64 pooled features to 1000 logits before RDD
fine-tuning.

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

Parameters are learned model weights. Torchvision reports about 11.69 million
parameters for ResNet18. The local ResNet8 architecture has 111,928
parameters.

## Tiny Model

In this project, a tiny model is below 1 MB. ResNet8 is about 0.43 MiB in FP32
parameter size, so it satisfies the tiny bucket before quantization.

## FLOPs

FLOPs estimates the floating-point operations required for one forward pass.
Torchvision reports about 1.814 billion operations for ResNet18 at its default
input size.

Actual speed also depends on hardware, memory movement, and how efficiently
the inference library implements each operation.
