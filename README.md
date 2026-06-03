# Cascaded-TinyVLM

Cascaded TinyVLMs for real-time edge AI and object detection on Jetson Nano.

## Features
- Real-time object detection
- Lightweight VLM inference
- Edge AI deployment
- Hierarchical inference
- Jetson Nano optimization
- Uncertainty-aware inference
- Conformal prediction

## Hardware
- NVIDIA Jetson Nano

## Planned Models
- YOLO
- MobileNet
- EfficientNet
- SmolVLM
- Qwen2-VL

## Research Goals
- Reduce latency
- Reduce energy consumption
- Improve edge reliability
- Enable adaptive edge/cloud inference



# Repository Structure

```text
Cascaded-TinyVLM/
│
├── docs/
├── src/
├── models/
├── configs/
├── scripts/
├── experiments/
├── jetson/
├── results/
└── tests/
```

## Folder Overview

### `docs/`
Contains documentation, architecture diagrams, research notes, benchmark summaries, and implementation details.

Examples:
- system architecture
- experiment reports
- paper notes
- deployment guides
- figures for presentations

---

### `src/`
Main source code for the project.

Will contain:
- inference pipelines
- object detection logic
- VLM integration
- uncertainty estimation
- conformal prediction
- hierarchical inference logic
- utilities

Example future structure:

```text
src/
├── inference/
├── vlm/
├── detection/
├── uncertainty/
├── conformal/
├── hierarchy/
└── utils/
```

---

### `models/`
Stores model-related files and wrappers.

Examples:
- YOLO checkpoints
- MobileNet/EfficientNet models
- TinyVLM adapters
- TensorRT engines
- ONNX exports

Large files should NOT be committed directly to Git.

---

### `configs/`
Configuration files for experiments and deployment.

Examples:
- model settings
- dataset paths
- inference thresholds
- Jetson deployment configs
- uncertainty parameters
- conformal calibration configs

Recommended format:
- YAML
- JSON

---

### `scripts/`
Standalone scripts for automation and execution.

Examples:
- benchmarking scripts
- deployment scripts
- training scripts
- inference launchers
- dataset preprocessing

---

### `experiments/`
Experiment tracking and reproducibility folder.

Examples:
- latency benchmarks
- energy measurements
- uncertainty experiments
- hierarchical inference evaluations
- VLM comparisons

Each experiment should ideally contain:
- configuration
- logs
- metrics
- notes

---

### `jetson/`
Jetson Nano specific code and deployment utilities.

Examples:
- TensorRT optimization
- CUDA benchmarking
- FPS profiling
- power measurements
- deployment scripts
- Jetson-specific inference pipelines

---

### `results/`
Stores generated outputs and evaluation artifacts.

Examples:
- plots
- graphs
- benchmark tables
- confusion matrices
- screenshots
- evaluation summaries

Large outputs should not be tracked directly in Git.

---

### `tests/`
Unit tests and validation scripts.

Examples:
- inference tests
- pipeline checks
- model loading tests
- latency validation
- regression testing