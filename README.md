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

---

## Local SonarQube Analysis

This project can be checked locally with SonarQube for code smells,
duplication, security findings, and imported test coverage.

Start the local SonarQube server with Docker:

```bash
docker run --name sonarqube -p 9000:9000 sonarqube:community
```

For later sessions, restart the same container:

```bash
docker start sonarqube
```

Wait until SonarQube is ready:

```bash
docker logs -f sonarqube
```

You can also check the status endpoint:

```bash
curl http://localhost:9000/api/system/status
```

The status should be `UP`.

Open SonarQube in the browser:

```text
http://localhost:9000
```

Create the local project with this key:

```text
cascaded-tinyvlm
```

Generate a user token in SonarQube, copy the example environment file, and add
the token to `.env`:

```bash
cp .env.example .env
```

```bash
SONAR_HOST_URL=http://localhost:9000
SONAR_TOKEN=sqp_your_token_here
```

Do not commit `.env`. It contains the local SonarQube token.

Install the scanner dependencies with uv:

```bash
uv sync --group dev
```

Load the environment variables and run the scanner:

```bash
set -a
source .env
set +a

uv run pysonar
```

The scanner uses `sonar-project.properties` for the project settings and sends
the analysis to:

```text
http://localhost:9000/dashboard?id=cascaded-tinyvlm
```

SonarQube does not run tests by itself. When tests are available, generate the
coverage and test reports before running `pysonar`:

```bash
mkdir -p reports

uv run pytest \
  -p pytest_cov \
  --cov=src \
  --cov=experiments \
  --cov-report=xml:coverage.xml \
  --junitxml=reports/pytest.xml

uv run pysonar
```

The generated `coverage.xml`, `reports/`, and `.sonar/` scanner output are local
artifacts and should not be committed.

If pytest tries to load unrelated system plugins, such as ROS pytest plugins,
run the coverage command with plugin autoload disabled while explicitly loading
`pytest-cov`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  -p pytest_cov \
  --cov=src \
  --cov=experiments \
  --cov-report=xml:coverage.xml \
  --junitxml=reports/pytest.xml
```
