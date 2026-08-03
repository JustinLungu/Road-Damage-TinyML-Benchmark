# Tests

This directory contains unit tests and regression checks grouped by subsystem.

The suite groups tests by subsystem so failures are easy to locate. It covers
model loading, vision and performance benchmarks, plotting, RDD training, and
hardware metrics without downloading real models or requiring CUDA.

Run the fast unit test suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

Run tests with coverage reports for SonarQube:

```bash
mkdir -p reports

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  -p pytest_cov \
  --cov=src \
  --cov=experiments \
  --cov-report=xml:coverage.xml \
  --junitxml=reports/pytest.xml
```
