#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGENETTE_DIR="${REPO_ROOT}/datasets/imagenette"
VALIDATION_DIR="${IMAGENETTE_DIR}/validation"
IMAGES_DIR="${VALIDATION_DIR}/images"
MANIFEST_PATH="${IMAGENETTE_DIR}/validation_labels.csv"
ARCHIVE_URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
ARCHIVE_NAME="imagenette2-160.tgz"

# Imagenette directory synsets mapped to the standard ImageNet-1K output index.
declare -A CLASS_IDS=(
    [n01440764]=0
    [n02102040]=217
    [n02979186]=482
    [n03000684]=491
    [n03028079]=497
    [n03394916]=566
    [n03417042]=569
    [n03425413]=571
    [n03445777]=574
    [n03888257]=701
)
SYNSETS=(
    n01440764
    n02102040
    n02979186
    n03000684
    n03028079
    n03394916
    n03417042
    n03425413
    n03445777
    n03888257
)

usage() {
    cat <<EOF
Usage:
  ./scripts/download_imagenette_val.sh

Downloads Imagenette-160 and keeps its labeled validation split for evaluating
the repository's ImageNet-pretrained classification models.
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    "")
        ;;
    *)
        echo "Error: this script does not accept arguments." >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

if [[ -e "${VALIDATION_DIR}" || -e "${MANIFEST_PATH}" ]]; then
    echo "Error: Imagenette validation data already exists under:" >&2
    echo "${IMAGENETTE_DIR}" >&2
    echo "Remove or move it before downloading again." >&2
    exit 1
fi

for command in wget tar find sort; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Error: required command is not installed: ${command}" >&2
        exit 1
    fi
done

temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/imagenette.XXXXXX")"
trap 'rm -rf "${temp_dir}"' EXIT

archive_path="${temp_dir}/${ARCHIVE_NAME}"
extract_dir="${temp_dir}/extracted"

echo "Downloading Imagenette-160..."
wget --output-document="${archive_path}" "${ARCHIVE_URL}"

echo "Extracting the labeled validation split..."
mkdir -p "${extract_dir}"
tar -xzf "${archive_path}" -C "${extract_dir}" imagenette2-160/val

extracted_validation_dir="${extract_dir}/imagenette2-160/val"
if [[ ! -d "${extracted_validation_dir}" ]]; then
    echo "Error: the archive did not contain imagenette2-160/val." >&2
    exit 1
fi

mkdir -p "${IMAGES_DIR}"

synset=""
for synset in "${SYNSETS[@]}"; do
    source_dir="${extracted_validation_dir}/${synset}"
    if [[ ! -d "${source_dir}" ]]; then
        echo "Error: missing Imagenette class directory: ${synset}" >&2
        exit 1
    fi
    mv "${source_dir}" "${IMAGES_DIR}/${synset}"
done

{
    printf "image_path,class_id\n"
    for synset in "${SYNSETS[@]}"; do
        while IFS= read -r image_path; do
            relative_path="${image_path#"${IMAGENETTE_DIR}/"}"
            printf "%s,%s\n" "${relative_path}" "${CLASS_IDS[$synset]}"
        done < <(
            find "${IMAGES_DIR}/${synset}" \
                -maxdepth 1 \
                -type f \
                \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) \
                | sort
        )
    done
} > "${MANIFEST_PATH}"

image_count="$(find "${IMAGES_DIR}" -type f | wc -l)"
manifest_count="$(tail -n +2 "${MANIFEST_PATH}" | wc -l)"
if [[ "${image_count}" -eq 0 || "${manifest_count}" -ne "${image_count}" ]]; then
    echo "Error: Imagenette image and manifest counts do not match." >&2
    exit 1
fi

echo "Imagenette validation dataset is ready."
echo "Images: ${image_count}"
echo "Manifest: ${MANIFEST_PATH}"
