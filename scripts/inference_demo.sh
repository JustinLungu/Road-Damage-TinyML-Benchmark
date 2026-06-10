#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DOCS_DIR="$ROOT_DIR/docs/models"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export UV_CACHE_DIR

list_demos() {
    find "$MODEL_DOCS_DIR" \
        -mindepth 2 \
        -maxdepth 2 \
        -name demo.py \
        -type f \
        -printf '%h\n' \
        | xargs -r -n 1 basename \
        | sort
}

usage() {
    cat <<EOF
Usage:
  ./scripts/inference_demo.sh <model-doc-directory>
  ./scripts/inference_demo.sh --model <model-doc-directory>
  ./scripts/inference_demo.sh -m <model-doc-directory>
  ./scripts/inference_demo.sh --list

Examples:
  ./scripts/inference_demo.sh yolo
  ./scripts/inference_demo.sh --model mobilenet
  ./scripts/inference_demo.sh -m smolvlm

Available demos:
$(list_demos)
EOF
}

model_name=""

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --list)
        list_demos
        exit 0
        ;;
    -m|--model)
        if [[ $# -ne 2 ]]; then
            echo "Error: $1 requires a model-doc-directory value." >&2
            echo >&2
            usage >&2
            exit 1
        fi
        model_name="$2"
        ;;
    "")
        echo "Error: missing model-doc-directory." >&2
        echo >&2
        usage >&2
        exit 1
        ;;
    *)
        if [[ $# -ne 1 ]]; then
            echo "Error: expected one model-doc-directory argument." >&2
            echo >&2
            usage >&2
            exit 1
        fi
        model_name="$1"
        ;;
esac

if [[ "$model_name" == *"/"* || "$model_name" == *".."* ]]; then
    echo "Error: model-doc-directory must be a directory name under docs/models." >&2
    exit 1
fi

demo_path="$MODEL_DOCS_DIR/$model_name/demo.py"

if [[ ! -f "$demo_path" ]]; then
    echo "Error: no demo.py found for '$model_name'." >&2
    echo >&2
    echo "Available demos:" >&2
    list_demos >&2
    exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$UV_CACHE_DIR"
uv run python "$demo_path"
