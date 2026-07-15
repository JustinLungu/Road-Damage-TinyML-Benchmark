#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RDD2022_DIR="${REPO_ROOT}/datasets/rdd2022"
BASE_URL="https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022"

# Map user-facing CLI names to the exact directory/archive names used upstream.
declare -A COUNTRY_NAMES=(
    [china-drone]="China_Drone"
    [china-motorbike]="China_MotorBike"
    [czech]="Czech"
    [india]="India"
    [japan]="Japan"
    [norway]="Norway"
    [united-states]="United_States"
)

declare -A ARCHIVE_SIZES=(
    [china-drone]="152.8 MB"
    [china-motorbike]="183.1 MB"
    [czech]="245.2 MB"
    [india]="502.3 MB"
    [japan]="1022.9 MB"
    [norway]="9.9 GB"
    [united-states]="423.8 MB"
)

usage() {
    cat <<EOF
Usage:
  ./scripts/download_rdd2022_subset.sh [country]
  ./scripts/download_rdd2022_subset.sh all
  ./scripts/download_rdd2022_subset.sh --list

Downloads one official RDD2022 country archive, or all available country
archives, and keeps only annotated training splits. The default country is
India.

Examples:
  ./scripts/download_rdd2022_subset.sh
  ./scripts/download_rdd2022_subset.sh all
  ./scripts/download_rdd2022_subset.sh china-motorbike
  ./scripts/download_rdd2022_subset.sh united-states
EOF
}

list_countries() {
    printf "%-18s %-18s %s\n" "argument" "dataset directory" "archive size"
    printf "%-18s %-18s %s\n" "--------" "-----------------" "------------"

    local country_key
    for country_key in \
        china-drone \
        china-motorbike \
        czech \
        india \
        japan \
        norway \
        united-states
    do
        printf "%-18s %-18s %s\n" \
            "${country_key}" \
            "${COUNTRY_NAMES[$country_key]}" \
            "${ARCHIVE_SIZES[$country_key]}"
    done
}

normalize_country() {
    # Accept common spelling variants while keeping one internal country key.
    local country="${1,,}"
    country="${country//_/-}"
    country="${country// /-}"

    case "${country}" in
        china-drone|china-motorbike|czech|india|japan|norway|united-states)
            printf "%s\n" "${country}"
            ;;
        us|usa|unitedstates)
            printf "united-states\n"
            ;;
        *)
            return 1
            ;;
    esac
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --list)
        list_countries
        exit 0
        ;;
esac

download_country() {
    local country_key="$1"
    local country_name="${COUNTRY_NAMES[$country_key]}"
    local archive_name="RDD2022_${country_name}.zip"
    local archive_url="${BASE_URL}/${archive_name}"
    local country_dir="${RDD2022_DIR}/${country_name}"
    local target_train_dir="${country_dir}/train"

    # Do not overwrite an existing dataset because it may contain user changes.
    if [[ -e "${target_train_dir}" ]]; then
        echo "Skipping RDD2022 ${country_name}: target already exists."
        echo "Dataset directory: ${target_train_dir}"
        return 0
    fi

    # Download and extract in temporary storage. The trap cleans it on success,
    # failure, or interruption, so archives are not left in the repository.
    local temp_dir
    temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/rdd2022.XXXXXX")"
    local archive_path="${temp_dir}/${archive_name}"
    local extract_dir="${temp_dir}/extracted"

    trap "rm -rf '${temp_dir}'" EXIT

    echo "Downloading RDD2022 ${country_name} archive (${ARCHIVE_SIZES[$country_key]})..."
    wget --output-document="${archive_path}" "${archive_url}"

    # Country archives also contain an unannotated challenge test split.
    # Extracting only <country>/train keeps the useful labeled subset and
    # reduces disk usage.
    echo "Extracting the annotated training split..."
    mkdir -p "${extract_dir}"
    unzip -q "${archive_path}" "${country_name}/train/*" -d "${extract_dir}"

    local extracted_train_dir="${extract_dir}/${country_name}/train"
    local extracted_images_dir="${extracted_train_dir}/images"
    local extracted_annotations_dir="${extracted_train_dir}/annotations/xmls"

    # Validate the upstream structure before moving anything into datasets/.
    if [[ ! -d "${extracted_images_dir}" || ! -d "${extracted_annotations_dir}" ]]; then
        echo "Error: the archive did not contain the expected RDD2022 train layout." >&2
        exit 1
    fi

    # Count the extracted files for both validation and the completion summary.
    local image_count
    image_count="$(
        find "${extracted_images_dir}" -maxdepth 1 -type f -name '*.jpg' | wc -l
    )"
    local annotation_count
    annotation_count="$(
        find "${extracted_annotations_dir}" -maxdepth 1 -type f -name '*.xml' | wc -l
    )"

    # Install only after every validation passes, avoiding half-populated targets.
    mkdir -p "${country_dir}"
    mv "${extracted_train_dir}" "${target_train_dir}"
    rm -rf "${temp_dir}"
    trap - EXIT

    echo "RDD2022 ${country_name} training subset is ready."
    echo "Images: ${image_count}"
    echo "Annotations: ${annotation_count}"
    echo "Dataset directory: ${target_train_dir}"
}

# Parse one optional country argument. No argument means the India subset.
if [[ $# -gt 1 ]]; then
    echo "Error: expected at most one country argument." >&2
    echo >&2
    usage >&2
    exit 1
fi

selected_country="${1:-india}"

if [[ "${selected_country,,}" != "all" ]] && ! country_key="$(normalize_country "${selected_country}")"; then
    echo "Error: unsupported RDD2022 country '${1:-}'." >&2
    echo >&2
    list_countries >&2
    exit 1
fi

for command in wget unzip find; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Error: required command is not installed: ${command}" >&2
        exit 1
    fi
done

if [[ "${selected_country,,}" == "all" ]]; then
    for country_key in \
        china-drone \
        china-motorbike \
        czech \
        india \
        japan \
        norway \
        united-states
    do
        download_country "${country_key}"
    done
else
    download_country "${country_key}"
fi
