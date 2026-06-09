# YOLO Definitions

This glossary explains the terms used in the YOLO model notes. The goal is practical understanding: what the term means, why it exists, and how it affects inference.

## Anchor

An anchor is a predefined box shape used by older object detectors to help predict bounding boxes. Instead of predicting every box from nothing, the model starts from several template boxes at each image location and learns how to adjust them.

For example, a detector might have anchor templates for tall boxes, wide boxes, and square boxes. If the image contains a person, the model can adjust a tall anchor to fit the person.

The downside is that anchors are a design choice. They must be selected to match the dataset reasonably well, and they add extra prediction logic.

## Anchor-Free

Anchor-free detection removes predefined anchor templates. The model predicts boxes directly from feature-map locations.

In practical terms:

- old anchor-based style: "at this location, adjust these preset boxes";
- anchor-free style: "at this location, predict the object box directly."

The local `yolov5nu` and `yolov8n` checkpoints both use the modern Ultralytics anchor-free detection head.

## Objectness

Objectness is a score used by older YOLO heads to estimate whether a predicted box contains any object at all, independent of which class it is.

Modern Ultralytics heads used by these checkpoints are objectness-free. They predict box localization and class scores in a split head, without a separate objectness score.

## C3

C3 is a YOLOv5 building block. It is a compact convolutional block based on CSP ideas.

The high-level idea is:

1. Split features into paths.
2. Run bottleneck transformations on one path.
3. Keep another path lighter.
4. Concatenate/merge them again.

This helps the model learn useful features without making the network too expensive.

`yolov5nu` uses C3/CSP-style blocks in its backbone and neck.

## CSP

CSP means Cross Stage Partial. It is a network design pattern where feature channels are split so only part of the features pass through heavier computation while another part bypasses it.

This is useful because it:

- reduces repeated computation;
- keeps gradient flow healthy during training;
- preserves information while saving compute.

In YOLO explanations, CSP is usually discussed as part of YOLOv5-style feature blocks.

## C3 / CSP Bottleneck Blocks

A bottleneck block compresses features, processes them, and expands or merges them again. This saves computation compared with doing every operation at full channel width.

In C3/CSP bottleneck blocks, only part of the channels go through these bottleneck transformations, while another path carries features forward more directly.

For inference, you can think of these blocks as efficient feature extractors. They help the detector recognize edges, textures, parts of objects, and larger object patterns while keeping the model small enough for real-time use.

## C2f Blocks

C2f is the YOLOv8 replacement/evolution of older C3-style blocks. It is still designed to be efficient, but it changes how intermediate features are split, reused, and concatenated.

The practical purpose is similar to C3:

- extract useful visual features;
- keep computation low;
- preserve feature flow through the network.

`yolov8n` uses C2f blocks in its backbone and neck.

## Conv-BatchNorm-SiLU

This is a common sequence inside YOLO convolution blocks:

1. **Conv**: a convolution learns local visual patterns such as edges, corners, textures, and object parts.
2. **BatchNorm**: batch normalization stabilizes feature values so the model is easier to train and behaves more predictably.
3. **SiLU**: SiLU is the activation function. It adds non-linearity so the model can learn more than simple linear filters.

Many YOLO layers are built from this pattern.

## Feature Map

A feature map is an intermediate tensor inside the model. Instead of raw pixels, it stores learned visual features.

Early feature maps usually preserve more spatial detail and represent simple patterns. Deeper feature maps are smaller spatially but represent more semantic concepts, such as object parts or whole-object cues.

YOLO combines multiple feature-map scales because small objects need spatial detail, while large objects benefit from deeper semantic features.

## SPPF

SPPF means Spatial Pyramid Pooling - Fast. It is a module near the deepest part of YOLO.

It applies pooling in a way that lets the model "see" a larger context without greatly increasing computation. This helps the detector reason about larger objects and surrounding context.

You can think of SPPF as a cheap way to widen the model's field of view.

## Neck

The neck is the part of the detector between the backbone and the detection head.

- The backbone extracts features from the image.
- The neck mixes features from different scales.
- The head turns those mixed features into boxes and class scores.

YOLO necks commonly upsample deeper features and concatenate them with earlier, higher-resolution features. This lets the model combine semantic understanding with spatial precision.

## Detection Head

The detection head is the final prediction part of the model. It receives feature maps from the neck and predicts:

- bounding box coordinates;
- class scores.

The local YOLO checkpoints use a split head: one branch focuses on box localization, and another branch focuses on class prediction.

## Stride

Stride describes how much smaller a feature map is compared with the input image.

For these models, the detection strides are 8, 16, and 32:

- stride 8 means one feature-map cell corresponds to an 8x8 region of the input image;
- stride 16 means one feature-map cell corresponds to a 16x16 region;
- stride 32 means one feature-map cell corresponds to a 32x32 region.

Smaller stride keeps more spatial detail and is better for small objects. Larger stride has more context and is better for larger objects.

## FLOPs

FLOPs means floating-point operations. It is an approximate measure of how much computation a model needs for one forward pass.

Lower FLOPs usually means faster inference and lower power use, although real speed also depends on hardware, memory movement, implementation details, image size, and batching.

In the Ultralytics tables, FLOPs are reported for a standard input size, commonly 640.

## Non-Maximum Suppression

Non-maximum suppression, often shortened to NMS, removes duplicate detections.

A detector may predict several overlapping boxes around the same object. NMS keeps the strongest box and suppresses other boxes that overlap too much with it.

The `NMS_IOU_THRESHOLD` constant in `demo.py` controls how much overlap is tolerated during this cleanup step.
