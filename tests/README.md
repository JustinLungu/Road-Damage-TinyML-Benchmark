# Tests

This directory will contain unit tests, validation scripts, pipeline checks, and regression tests.

The suite uses a small number of broader scenario tests instead of many tiny
tests. The goal is to cover the model-loading, benchmark, plotting, and hardware
metric paths without downloading real models or requiring CUDA.

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
