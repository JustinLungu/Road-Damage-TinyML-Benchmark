# MobileNet Definitions

This glossary explains the terms used in the MobileNet model notes.

## Image Classification

Image classification assigns one or more labels to a whole image. It does not return bounding boxes or object locations.

For example, a classifier may say an image is most likely `groom`, `suit`, or `bow tie`, but it does not identify where those things are in the image.

This is different from YOLO object detection, which returns object boxes and class labels for each detected object.

## ImageNet-1K

ImageNet-1K is a common image-classification benchmark with 1000 object/category classes.

The local MobileNetV2 and MobileNetV3 checkpoints are pretrained on
ImageNet-1K, so their output vector has 1000 values. Each output position
corresponds to one ImageNet class.

## Logits

Logits are the raw model output values before probabilities are computed.

A larger logit means the model prefers that class more strongly, but logits are not probabilities yet. They can be positive, negative, and do not need to sum to 1.

## Softmax

Softmax converts logits into probabilities. After softmax, all class probabilities sum to 1.

In the demo:

```python
probabilities = logits.softmax(dim=1)[0]
```

The top probabilities are the model's most likely ImageNet classes for the whole image.

## Top-K

Top-k means the `k` highest-scoring predictions.

If `TOP_K = 5`, the demo prints the five most likely ImageNet classes. This is useful because classification can be uncertain, and several related labels may be plausible.

## Preprocessing

Preprocessing prepares the original image for the model.

Torchvision MobileNet weights include the exact transforms expected by each
checkpoint. Those transforms usually:

1. resize the image;
2. crop the center region;
3. convert it to a tensor;
4. normalize RGB channels with ImageNet mean and standard deviation.

This matters because pretrained models expect images to be processed the same way they were processed during training.

## Resize and Center Crop

Resize changes the image scale. Center crop takes a fixed-size crop from the center of the resized image.

For the local default weights:

- `mobilenet_v2`: resize to 232, center crop to 224;
- `mobilenet_v3_small`: resize to 256, center crop to 224;
- `mobilenet_v3_large`: resize to 232, center crop to 224.

The model receives a 224x224 tensor even if the source image has a different size.

## Normalization

Normalization shifts and scales pixel values. Torchvision ImageNet models commonly use:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

This makes input values match the distribution the model saw during training.

## Channel

An RGB image has three input channels: red, green, and blue.

Inside a CNN, feature channels are learned representations. Early channels may respond to simple patterns like edges or colors. Later channels may respond to object parts or high-level visual concepts.

## Convolution

A convolution applies small learned filters across the image or feature map. It is the basic operation used by CNNs to detect local patterns.

Early convolutions often learn simple patterns. Deeper convolutions combine those patterns into more meaningful visual features.

## Depthwise Separable Convolution

Depthwise separable convolution is a cheaper alternative to a standard convolution.

It splits the operation into two parts:

1. **Depthwise convolution**: applies one spatial filter per channel.
2. **Pointwise convolution**: uses 1x1 convolutions to mix information across channels.

This greatly reduces computation, which is one reason MobileNet models are efficient.

## Inverted Residual Block

An inverted residual block is a MobileNet block that expands channels, applies an efficient depthwise convolution, then projects channels back down.

The "inverted" part means the block is wide in the middle and narrow at the input/output. This is the opposite of older bottleneck blocks that are narrow in the middle.

## Linear Bottleneck

A linear bottleneck is the narrow projection at the end of a MobileNetV2
inverted residual block.

The block expands features, processes them with a depthwise convolution, then
projects them back to fewer channels. MobileNetV2 does not apply another
non-linear activation after this final projection. The intent is to avoid
destroying useful information when the representation is compressed.

## Residual Connection

A residual connection adds a block's input to its output when their shapes
match.

This gives information and gradients a direct path through the network.
MobileNetV2 inverted residual blocks use this connection when their stride is
one and the input/output channel counts match.

## ReLU6

ReLU6 is an activation that clamps values to the range from zero to six:

```text
ReLU6(x) = min(max(x, 0), 6)
```

MobileNetV2 uses ReLU6 inside its stem and expanded block representations. The
bounded activation was designed to behave predictably in lower-precision mobile
inference.

## Squeeze-and-Excitation

Squeeze-and-excitation, often shortened to SE, lets the network reweight channels based on global image context.

It helps the model emphasize useful feature channels and suppress less useful
ones. The local MobileNetV3 models use SE in selected blocks; MobileNetV2 does
not.

## h-swish

h-swish means hard swish, an efficient approximation of the swish activation function.

MobileNetV3 uses hardware-aware activation choices like h-swish to keep inference efficient while preserving accuracy.

## Parameters

Parameters are learned weights in the model.

More parameters can increase model capacity, but they also increase model size
and memory use. MobileNetV3 Small has the fewest parameters locally,
MobileNetV2 is in the middle, and MobileNetV3 Large has the most.

## FLOPs

FLOPs means floating-point operations. It approximates how much computation a model needs for one forward pass.

Lower FLOPs usually means faster and more energy-efficient inference, although actual speed also depends on hardware and implementation details.

Torchvision reports MobileNetV3 Small as the cheapest local variant by FLOPs.
MobileNetV2 uses more operations than both V3 variants with the selected
pretrained weights, but it remains an important simple edge-oriented baseline.
