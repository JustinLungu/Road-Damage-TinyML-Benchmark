#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COCO_DIR="${SCRIPT_DIR}/../datasets/coco"

cd "${COCO_DIR}"

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

mv val2017 images

rm val2017.zip annotations_trainval2017.zip
