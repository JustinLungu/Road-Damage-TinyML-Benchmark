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

- [MobileNetV3 classifiers](mobilenet/README.md): Torchvision MobileNetV3 Small and Large classifiers loaded from `models/cnn/hub/checkpoints/`.

## Vision-Language

- [SmolVLM models](smolvlm/README.md): Hugging Face SmolVLM instruct models loaded from `models/vlm/` for image-plus-text prompting.
