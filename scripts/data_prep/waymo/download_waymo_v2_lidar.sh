#!/bin/bash
# Download Waymo v2 lidar component (range images).
# Needed together with lidar_camera_projection to create depth maps.
# WARNING: This is large (~100GB+ per split).
#
# Usage:
#   bash data_conversion/waymo/download_waymo_v2_lidar.sh

set -e

BUCKET="gs://waymo_open_dataset_v_2_0_0"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/waymo_v2}"

SPLITS="validation training"

echo "Downloading lidar (range images)..."

for SPLIT in ${SPLITS}; do
    SRC="${BUCKET}/${SPLIT}/lidar/"
    DST="${OUTPUT_ROOT}/${SPLIT}/lidar/"
    mkdir -p "${DST}"
    echo "--- ${SPLIT}/lidar ---"
    gsutil -m cp -n "${SRC}*.parquet" "${DST}"
    echo "  Done."
done

echo "Download complete!"
