# DS-CNN Definitions

## DS-CNN

DS-CNN means depthwise-separable convolutional neural network. It replaces many
standard convolutions with cheaper depthwise and pointwise convolutions.

## Depthwise Convolution

A depthwise convolution applies one spatial filter per input channel. It does
not mix information between channels.

## Pointwise Convolution

A pointwise convolution is a 1x1 convolution. It mixes information across
channels after the depthwise spatial filtering step.

## Depthwise-Separable Block

A depthwise-separable block combines:

```text
depthwise convolution -> pointwise convolution
```

This usually requires far fewer parameters and operations than a standard
convolution.

## Tiny Model

In this project, a tiny model is below 1 MB. `ds_cnn_small` is about 0.20 MiB
in FP32 parameter size, so it satisfies the tiny bucket before quantization.

## Parameter Size

Parameter size is the storage needed for model weights. FP32 weights use four
bytes per parameter. `ds_cnn_small` has 53,152 parameters:

```text
53,152 x 4 bytes = 212,608 bytes
```

That is about 0.20 MiB.

## Adaptive Average Pooling

Adaptive average pooling converts the final spatial feature map into one fixed
feature vector per image. This lets the classifier head consume a stable feature
size.

## Classifier Head

The classifier head is the final linear layer. In RDD fine-tuning, it is
replaced with a 2-class pothole/no-pothole head.

## Logits

Logits are raw model scores before softmax. Higher logits mean the model prefers
that class more strongly.
