# Scripts

This directory contains standalone scripts for dataset downloads, model
downloads, inference demos, and benchmark automation.

## Dataset Downloads

Download the COCO 2017 validation images and annotations:

```bash
./scripts/download_coco_val.sh
```

Download the labeled Imagenette validation split used by the classification
accuracy benchmark:

```bash
./scripts/download_imagenette_val.sh
```

Download the annotated training split for one RDD2022 country. India is the
default initial subset:

```bash
./scripts/download_rdd2022_subset.sh
```

List or select the other official country-specific archives:

```bash
./scripts/download_rdd2022_subset.sh --list
./scripts/download_rdd2022_subset.sh china-motorbike
```

Download every available country-specific annotated training split:

```bash
./scripts/download_rdd2022_subset.sh all
```

Create binary pothole/non-pothole image-classification manifests from the
downloaded RDD2022 XML annotations:

```bash
uv run python -m src.rdd_benchmark.main
```

See `datasets/rdd2022/README.md` for the dataset layout, classes, and source
information.

## Model Downloads

Load/download selected pretrained models through the repository model registry:

```bash
./scripts/download_models.sh mobilenet_v2 resnet18 efficientnet_b0
```

List supported model names:

```bash
./scripts/download_models.sh --list
```

Load/download every supported model:

```bash
./scripts/download_models.sh --all
```

For the RDD2022 binary pothole fine-tuning work, start with the image
classification models, for example:

```bash
./scripts/download_models.sh \
  mobilenet_v2 \
  mobilenet_v3_small \
  mobilenet_v3_large \
  efficientnet_b0 \
  resnet18 \
  inception_v3
```

## Inference Demos

Use `inference_demo.sh` to run any `demo.py` under `docs/models/<model-family>/` from one common entry point.

List available demos:

```bash
./scripts/inference_demo.sh --list
```

Run a demo by directory name:

```bash
./scripts/inference_demo.sh yolo
```

You can also use the explicit flag:

```bash
./scripts/inference_demo.sh --model mobilenet
./scripts/inference_demo.sh --model resnet
./scripts/inference_demo.sh --model inception
./scripts/inference_demo.sh -m smolvlm
```

The argument must match a directory under `docs/models/` that contains a `demo.py`, for example:

```text
efficientformer
efficientnet
inception
mobilenet
mobilevit
resnet
smolvlm
yolo
```

Each individual demo is configured by constants at the top of its own `docs/models/<model-family>/demo.py` file.
