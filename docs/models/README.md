# Model Notes

This directory documents the model families used in the repository. Use one subdirectory per model family, and keep the model explanations, demos, and generated output paths together in that folder.

Each model-family README should explain:

- where the model is registered and loaded;
- what happens inside the model at a high level;
- how inference is called in this project;
- how to read the returned outputs;
- which demo scripts can be run from the repository root.

## Object Detection

- [YOLO detectors](yolo/README.md): Ultralytics YOLOv5nu and YOLOv8n object detectors loaded from `models/cnn/`.

## Image Classification

- [TinyCNN classifier](tiny_cnn/README.md): custom local tiny CNN loaded from source code with no pretrained checkpoint.
- [DS-CNN classifiers](ds_cnn/README.md): custom local depthwise-separable CNN loaded from source code with no pretrained checkpoint.
- [EfficientNet-B0 classifier](efficientnet/README.md): Torchvision EfficientNet-B0 image classifier loaded from `models/cnn/hub/checkpoints/`.
- [EfficientFormer classifiers](efficientformer/README.md): timm EfficientFormer L1, L3, and L7 image classifiers loaded from `models/vit/`.
- [InceptionV3 classifier](inception/README.md): Torchvision InceptionV3 image classifier loaded from `models/cnn/hub/checkpoints/`.
- [MobileNet classifiers](mobilenet/README.md): Torchvision MobileNetV2 and MobileNetV3 Small/Large classifiers loaded from `models/cnn/hub/checkpoints/`.
- [MobileViT classifiers](mobilevit/README.md): Hugging Face MobileViT XXS, XS, and Small image classifiers loaded from `models/vit/`.
- [ResNet classifiers](resnet/README.md): Torchvision ResNet18 plus custom local ResNet8 residual classifier.

## Vision-Language

- [SmolVLM models](smolvlm/README.md): Hugging Face SmolVLM instruct models loaded from `models/vlm/` for image-plus-text prompting.
