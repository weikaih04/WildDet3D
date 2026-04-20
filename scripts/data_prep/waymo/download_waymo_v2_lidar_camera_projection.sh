#!/bin/bash
# Download Waymo v2 lidar_camera_projection component.
# Contains lidar points projected onto camera images (sparse depth map).
#
# Usage:
#   bash data_conversion/waymo/download_waymo_v2_lidar_camera_projection.sh

set -e

BUCKET="gs://waymo_open_dataset_v_2_0_0"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/waymo_v2}"

SPLITS="validation training"

echo "Downloading lidar_camera_projection..."

for SPLIT in ${SPLITS}; do
    SRC="${BUCKET}/${SPLIT}/lidar_camera_projection/"
    DST="${OUTPUT_ROOT}/${SPLIT}/lidar_camera_projection/"
    mkdir -p "${DST}"
    echo "--- ${SPLIT}/lidar_camera_projection ---"
    gsutil -m cp -n "${SRC}*.parquet" "${DST}"
    echo "  Done."
done

echo "Download complete!"
