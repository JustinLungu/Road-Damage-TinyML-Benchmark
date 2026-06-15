# Scripts

This directory contains standalone scripts for training, benchmarking, deployment, inference demos, and data preprocessing.

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

See `datasets/rdd2022/README.md` for the dataset layout, classes, and source
information.

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
