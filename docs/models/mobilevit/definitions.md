# MobileViT Definitions

This glossary explains the terms used in the MobileViT model notes.

## Image Classification

Image classification assigns labels to a whole image.

MobileViT does not return bounding boxes or generated text. It returns a score vector over ImageNet classes, then the demo converts that vector into top-k labels.

## ImageNet-1K

ImageNet-1K is a common image-classification benchmark with 1000 classes.

The local MobileViT checkpoints are pretrained for ImageNet-1K classification, so their output vector has 1000 values.

## MobileViT

MobileViT is a lightweight hybrid image model.

It combines CNN-style local feature extraction with transformer-style global feature mixing. The goal is to keep the model small enough for mobile or edge-style inference while still giving it a broader view of the image than a purely local convolution stack.

## CNN

A convolutional neural network, or CNN, uses learned filters that slide across an image or feature map.

CNNs are efficient and good at local patterns such as edges, textures, and object parts. MobileViT keeps this strength by using convolutional layers around its transformer blocks.

## Transformer

A transformer is a neural network block built around self-attention.

In vision models, transformers can mix information across distant image regions. This gives the model a way to reason about global context, not only nearby pixels.

## Self-Attention

Self-attention lets each token compare itself with other tokens and decide which ones matter.

For images, the tokens usually represent patches or patch-like feature chunks. Self-attention helps the model connect distant regions, such as a wheel and the body of a vehicle.

## Local Representation

Local representation means features computed from nearby pixels or nearby feature positions.

Convolutions naturally build local representations because each filter sees a small spatial neighborhood at a time.

## Global Representation

Global representation means features that can combine information from far-apart image regions.

Transformer self-attention helps MobileViT build this kind of representation.

## Patch

A patch is a small spatial chunk of an image or feature map.

MobileViT blocks unfold feature maps into patch-like sequences, process them with transformer layers, then fold them back into feature maps.

## Feature Map

A feature map is the tensor representation produced inside a vision model.

Early feature maps look closer to image structure. Deeper feature maps contain more abstract information about objects and scenes.

## Hidden Size

Hidden size is the number of feature channels used inside part of the model.

The three local MobileViT variants mainly differ by these widths. Larger hidden sizes usually improve capacity but increase compute and memory use.

## Neck

The neck is the later feature-processing part that connects the feature extractor to the classification head.

In the local configs, `neck_hidden_sizes` describe the channel widths used through this part of the network.

## Processor

The processor is the Hugging Face object that prepares images for the model.

For MobileViT, the processor handles resizing, center cropping, channel order, and tensor conversion before returning `pixel_values`.

## `pixel_values`

`pixel_values` are the preprocessed image tensors passed to the model.

For the local MobileViT processors, the source image is resized, center cropped to 256x256, and converted into a batched tensor.

## Logits

Logits are raw model output scores before probabilities are computed.

For a single image, MobileViT returns logits shaped like:

```text
1, 1000
```

Each value corresponds to one ImageNet class.

## Softmax

Softmax converts logits into probabilities.

The demo applies softmax before selecting the top-k labels:

```python
probabilities = logits.softmax(dim=-1)[0]
```

## Top-K

Top-k means the `k` highest-scoring predictions.

If `TOP_K = 5`, the demo prints the five most likely ImageNet labels for the whole image.

## Classifier Head

The classifier head maps the model's final image representation into class logits.

For MobileViT ImageNet checkpoints, the classifier head outputs one score for each of the 1000 ImageNet classes.
