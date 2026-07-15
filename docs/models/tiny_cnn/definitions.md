# TinyCNN Definitions

## Tiny Model

In this project, a tiny model is a model below 1 MB. `tiny_cnn` is about
0.34 MiB in FP32 parameter size, so it satisfies the tiny bucket before any
quantization.

## Parameter Size

Parameter size is the storage needed for model weights. FP32 weights use four
bytes per parameter. `tiny_cnn` has 88,696 parameters:

```text
88,696 x 4 bytes = 354,784 bytes
```

That is about 0.34 MiB.

## Convolution Block

A convolution block extracts local visual patterns such as edges, texture, or
small road-surface structures. `tiny_cnn` uses:

```text
Conv2d -> BatchNorm2d -> ReLU
```

## Adaptive Average Pooling

Adaptive average pooling converts the final spatial feature map into one fixed
feature vector per image. This lets the classifier head consume a stable feature
size.

## Classifier Head

The classifier head is the final linear layer. It maps the extracted image
features to class logits. In RDD fine-tuning, this head is replaced with a
2-class pothole/no-pothole head.

## Logits

Logits are raw model scores before softmax. Higher logits mean the model prefers
that class more strongly.
