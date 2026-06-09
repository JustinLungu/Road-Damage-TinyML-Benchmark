# Scripts

This directory contains standalone scripts for training, benchmarking, deployment, inference demos, and data preprocessing.

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
./scripts/inference_demo.sh -m smolvlm
```

The argument must match a directory under `docs/models/` that contains a `demo.py`, for example:

```text
efficientformer
mobilenet
mobilevit
smolvlm
yolo
```

Each individual demo is configured by constants at the top of its own `docs/models/<model-family>/demo.py` file.
