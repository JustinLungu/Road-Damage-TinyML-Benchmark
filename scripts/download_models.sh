#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export UV_CACHE_DIR

cd "${REPO_ROOT}"
mkdir -p "${UV_CACHE_DIR}"

if command -v uv >/dev/null 2>&1; then
    python_runner=(uv run python)
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    python_runner=("${REPO_ROOT}/.venv/bin/python")
else
    python_runner=(python)
fi

list_models() {
    "${python_runner[@]}" - <<'PY'
from src.load_model import MODEL_LOADERS

for model_name in sorted(MODEL_LOADERS):
    print(model_name)
PY
}

usage() {
    cat <<EOF
Usage:
  ./scripts/download_models.sh <model> [model ...]
  ./scripts/download_models.sh --all
  ./scripts/download_models.sh --list

Examples:
  ./scripts/download_models.sh mobilenet_v2 resnet18 efficientnet_b0
  ./scripts/download_models.sh --all
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --list)
        list_models
        exit 0
        ;;
    --all|-all|all)
        mapfile -t model_names < <(list_models)
        ;;
    "")
        echo "Error: provide at least one model name, or use --all." >&2
        echo >&2
        usage >&2
        exit 1
        ;;
    *)
        model_names=("$@")
        ;;
esac

if [[ "${#model_names[@]}" -eq 0 ]]; then
    echo "Error: no models selected." >&2
    exit 1
fi

available_models="$(list_models)"
failed_models=()

for model_name in "${model_names[@]}"; do
    if ! grep -Fxq "${model_name}" <<< "${available_models}"; then
        echo "Error: unsupported model '${model_name}'." >&2
        echo >&2
        echo "Available models:" >&2
        printf "%s\n" "${available_models}" >&2
        exit 1
    fi

    echo
    echo "Loading/downloading ${model_name}..."
    if ! "${python_runner[@]}" -m src.load_model --model "${model_name}"; then
        failed_models+=("${model_name}")
    fi
done

if [[ "${#failed_models[@]}" -gt 0 ]]; then
    echo >&2
    echo "Failed models: ${failed_models[*]}" >&2
    exit 1
fi

echo
echo "All selected models loaded successfully."
