# ShuffleNet Definitions

## ShuffleNetV2

ShuffleNetV2 is an efficient CNN architecture designed for mobile and edge
inference. It uses lightweight blocks and channel shuffle operations to improve
information flow while keeping computation low.

## Width Multiplier

A width multiplier scales the number of channels in a model. The 0.5x variant
uses fewer channels than the full-width model, reducing parameters and FLOPs.

## Channel Shuffle

Channel shuffle rearranges feature channels so information can move between
groups after grouped or split operations.

## Depthwise Convolution

A depthwise convolution applies one spatial filter per input channel. It is
cheaper than a standard convolution because it does not mix channels.

## Pointwise Convolution

A pointwise convolution is a 1x1 convolution. It mixes information across
channels.

## Global Pooling

Global pooling collapses spatial dimensions into one feature vector per image.

## Classifier Head

The classifier head is the final linear layer. In ShuffleNetV2, this is the
`fc` layer. In RDD fine-tuning, it is replaced with a 2-class pothole/no-pothole
head.

## Logits

Logits are raw model scores before softmax. Higher logits mean the model prefers
that class more strongly.

## FLOPs

FLOPs estimates the floating-point operations required for one forward pass.
Torchvision reports about 0.040 billion operations for ShuffleNetV2-0.5x.
