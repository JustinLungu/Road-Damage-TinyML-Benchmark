# EfficientFormer Definitions

This glossary explains the terms used in the EfficientFormer model notes.

## Image Classification

Image classification assigns labels to a whole image.

EfficientFormer does not return bounding boxes or generated text. It returns a score vector over ImageNet classes, then the demo converts that vector into top-k labels.

## ImageNet-1K

ImageNet-1K is a common image-classification benchmark with 1000 classes.

The local EfficientFormer timm checkpoints are ImageNet-1K classifiers, so their output vector has 1000 values.

## timm

`timm` is the PyTorch Image Models library.

This repo loads EfficientFormer through:

```python
timm.create_model("efficientformer_l1.snap_dist_in1k", pretrained=True)
```

`timm` also provides the preprocessing configuration used by the demo and benchmark.

## EfficientFormer

EfficientFormer is a vision model designed to bring transformer-style modeling closer to mobile or efficient CNN speed.

The timm implementation used here keeps most computation in efficient 2D feature-map blocks and uses attention only in the later 1D token blocks.

## Stem

The stem is the first part of the network.

In the local timm model, the stem is `Stem4`: two stride-2 convolutions that reduce the input image resolution by 4x while increasing channels.

## Stage

A stage is a group of blocks operating at a related feature-map resolution and channel width.

The local EfficientFormer models have four `EfficientFormerStage`s. Later stages usually have lower spatial resolution and more channels.

## Downsampling

Downsampling reduces spatial resolution.

EfficientFormer uses downsampling between stages so later blocks work on smaller feature maps with more channels. This reduces compute while allowing deeper processing.

## Feature Map

A feature map is an intermediate image-like tensor inside the model.

It usually has shape:

```text
channels, height, width
```

EfficientFormer keeps many blocks in this 2D feature-map form because it is efficient for image hardware and convolutional operations.

## MetaBlock2d

`MetaBlock2d` is the main 2D EfficientFormer block in the local timm implementation.

It contains:

- a token mixer;
- residual connections;
- layer scale;
- a convolutional MLP with normalization.

In the local models, most `MetaBlock2d` token mixers use pooling rather than full self-attention.

## Token Mixer

A token mixer is the part of a block that lets spatial positions exchange information.

In CNNs, convolutions mix nearby pixels. In transformers, self-attention can mix distant tokens. EfficientFormer uses efficient token mixing choices to keep inference fast.

## Pooling Token Mixer

The local EfficientFormer `MetaBlock2d` blocks use an average-pooling token mixer.

This is cheaper than full self-attention and keeps much of the model in efficient 2D operations.

## Conv MLP

A convolutional MLP is an MLP-like channel-mixing block implemented with convolutions.

In the local EfficientFormer 2D blocks, `ConvMlpWithNorm` expands channels, applies an activation, then projects channels back down.

## Layer Scale

Layer scale is a small learned multiplier applied to residual branch outputs.

It can help stabilize deep transformer-like networks by controlling how strongly each block changes the running representation.

## Flat

`Flat` is the transition from 2D feature maps to a 1D token sequence.

After this step, the model can use token-sequence operations such as attention.

## MetaBlock1d

`MetaBlock1d` is the later token-sequence block in the local timm EfficientFormer implementation.

Unlike `MetaBlock2d`, this block uses layer normalization, attention, and a standard MLP over 1D tokens.

## Attention

Attention lets each token decide which other tokens are relevant.

EfficientFormer uses attention near the end of the model, after earlier stages have already reduced spatial size.

## Distillation Head

The local timm models include both `head` and `head_dist`.

This reflects the pretrained checkpoint style. During normal evaluation, timm returns one 1000-class logits tensor for classification.

## Data Config

The data config tells the demo how to preprocess images for the checkpoint.

For these local EfficientFormer models, timm resolves a 224x224 input size, bicubic interpolation, ImageNet mean/std normalization, and a crop percentage of 0.95.

## Logits

Logits are raw model output scores before probabilities are computed.

For a single image, EfficientFormer returns logits shaped like:

```text
1, 1000
```

Each value corresponds to one ImageNet class.

## Softmax

Softmax converts logits into probabilities.

The demo applies softmax before selecting the top-k labels:

```python
probabilities = logits.softmax(dim=1)[0]
```

## Top-K

Top-k means the `k` highest-scoring predictions.

If `TOP_K = 5`, the demo prints the five most likely ImageNet labels for the whole image.
