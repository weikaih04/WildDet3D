"""Generate sparse LiDAR depth maps for Waymo FRONT camera images.

Combines lidar (range image) + lidar_camera_projection to create
per-frame depth maps saved as uint16 PNG (pixel = depth_m * 256).

These depth maps are used by the COCO3D dataloader when with_depth=True.

Usage:
    # Quick test (2 segments)
    python data_conversion/waymo/generate_waymo_depth_maps.py \
        --split validation --max_segments 2

    # Full validation (64 workers)
    python data_conversion/waymo/generate_waymo_depth_maps.py \
        --split validation --frame_interval 5 --num_workers 64

    # Full training (64 workers)
    python data_conversion/waymo/generate_waymo_depth_maps.py \
        --split training --frame_interval 5 --num_workers 64

Output:
    data/waymo/depth/{split}/{segment}_{timestamp}.png

Depth format:
    - dtype: uint16
    - encoding: pixel_value = depth_meters * 256
    - no-depth pixels: 0
    - loaded by COCO3D as: depth = im_decode(bytes).astype(float32) / 256.0

Requires:
    data/waymo_v2/{split}/lidar/
    data/waymo_v2/{split}/lidar_camera_projection/
    data/waymo_v2/{split}/camera_image/  (for timestamp/segment info)
"""

import argparse
import multiprocessing
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

FRONT_CAMERA = 1
DEPTH_SCALE = 256
SEG_COL = "key.segment_context_name"
TS_COL = "key.frame_timestamp_micros"
CAM_NAME_COL = "key.camera_name"


def process_segment(args):
    """Generate depth maps for one segment."""
    seg_file, split_dir, split, depth_output_dir, frame_interval, img_w, img_h = args

    seg_name = os.path.splitext(os.path.basename(seg_file))[0]

    # Load camera image timestamps (to know which frames to process)
    cam_path = os.path.join(split_dir, "camera_image", seg_file)
    cam_df = pq.read_table(cam_path, columns=[SEG_COL, TS_COL, CAM_NAME_COL]).to_pandas()
    cam_df = cam_df[cam_df[CAM_NAME_COL] == FRONT_CAMERA].reset_index(drop=True)
    if len(cam_df) == 0:
        return 0

    cam_df = cam_df.sort_values(TS_COL).reset_index(drop=True)
    if frame_interval > 1:
        cam_df = cam_df.iloc[::frame_interval].reset_index(drop=True)
    frame_timestamps = set(cam_df[TS_COL].tolist())

    lidar_path = os.path.join(split_dir, "lidar", seg_file)
    proj_path = os.path.join(split_dir, "lidar_camera_projection", seg_file)
    if not os.path.exists(lidar_path) or not os.path.exists(proj_path):
        return 0

    out_dir = os.path.join(depth_output_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    # Process one frame at a time to keep memory low.
    # Each read loads ~5 rows (~13MB) instead of entire segment (~2.7GB).
    ts_list = sorted(frame_timestamps)
    count = 0
    for ts in ts_list:
        out_path = os.path.join(out_dir, f"{seg_name}_{ts}.png")
        if os.path.exists(out_path):
            count += 1
            continue

        depth_map = np.zeros((img_h, img_w), dtype=np.float32)

        lidar_ts = pq.read_table(
            lidar_path, filters=[(TS_COL, "=", ts)]
        ).to_pandas()
        proj_ts = pq.read_table(
            proj_path, filters=[(TS_COL, "=", ts)]
        ).to_pandas()

        # Process all 5 lidars
        for laser_name in [1, 2, 3, 4, 5]:
            lidar_rows = lidar_ts[
                lidar_ts["key.laser_name"] == laser_name
            ]
            proj_rows = proj_ts[
                proj_ts["key.laser_name"] == laser_name
            ]
            if len(lidar_rows) == 0 or len(proj_rows) == 0:
                continue

            lidar_row = lidar_rows.iloc[0]
            proj_row = proj_rows.iloc[0]

            # Range image: [H, W, 4], channel 0 = range
            ri_shape = np.array(
                lidar_row["[LiDARComponent].range_image_return1.shape"]
            )
            ri_vals = np.array(
                lidar_row["[LiDARComponent].range_image_return1.values"]
            )
            range_img = ri_vals.reshape(ri_shape)
            ranges = range_img[:, :, 0]

            # Projection: [H, W, 6]
            pj_shape = np.array(
                proj_row[
                    "[LiDARCameraProjectionComponent]"
                    ".range_image_return1.shape"
                ]
            )
            pj_vals = np.array(
                proj_row[
                    "[LiDARCameraProjectionComponent]"
                    ".range_image_return1.values"
                ]
            )
            proj_img = pj_vals.reshape(pj_shape)

            # First return projection
            for ch_offset in [0, 3]:
                cam = proj_img[:, :, ch_offset]
                px = proj_img[:, :, ch_offset + 1]
                py = proj_img[:, :, ch_offset + 2]

                mask = (cam == FRONT_CAMERA) & (ranges > 0)
                xs = px[mask].astype(np.int32)
                ys = py[mask].astype(np.int32)
                ds = ranges[mask]

                valid = (xs >= 0) & (xs < img_w) & (ys >= 0) & (ys < img_h)
                xs, ys, ds = xs[valid], ys[valid], ds[valid]

                # Keep closer depth when multiple points hit same pixel
                for x, y, d in zip(xs, ys, ds):
                    if depth_map[y, x] == 0 or d < depth_map[y, x]:
                        depth_map[y, x] = d

        # Save as uint16 PNG
        depth_uint16 = (depth_map * DEPTH_SCALE).astype(np.uint16)
        cv2.imwrite(out_path, depth_uint16)
        count += 1

        del lidar_ts, proj_ts, depth_map, depth_uint16

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate Waymo LiDAR depth maps"
    )
    parser.add_argument(
        "--waymo_root", type=str, default="data/waymo_v2",
    )
    parser.add_argument(
        "--depth_output_dir", type=str, default="data/waymo/depth",
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        choices=["training", "validation"],
    )
    parser.add_argument(
        "--frame_interval", type=int, default=1,
    )
    parser.add_argument(
        "--max_segments", type=int, default=None,
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
    )
    args = parser.parse_args()

    split_dir = os.path.join(args.waymo_root, args.split)

    # Get image dimensions from calibration
    calib_dir = os.path.join(split_dir, "camera_calibration")
    calib_file = sorted(
        [f for f in os.listdir(calib_dir) if f.endswith(".parquet")]
    )[0]
    calib_df = pq.read_table(os.path.join(calib_dir, calib_file)).to_pandas()
    front_calib = calib_df[calib_df[CAM_NAME_COL] == FRONT_CAMERA].iloc[0]
    img_w = int(front_calib["[CameraCalibrationComponent].width"])
    img_h = int(front_calib["[CameraCalibrationComponent].height"])
    print(f"Image size: {img_w}x{img_h}")

    # List segments
    cam_dir = os.path.join(split_dir, "camera_image")
    seg_files = sorted(
        [f for f in os.listdir(cam_dir) if f.endswith(".parquet")]
    )
    print(f"Found {len(seg_files)} segments for {args.split}")

    if args.max_segments is not None and args.max_segments > 0:
        seg_files = seg_files[:args.max_segments]
        print(f"Limiting to {args.max_segments} segments")

    seg_args = [
        (sf, split_dir, args.split, args.depth_output_dir,
         args.frame_interval, img_w, img_h)
        for sf in seg_files
    ]

    if args.num_workers <= 1:
        results = []
        for a in tqdm(seg_args, desc="Generating depth"):
            results.append(process_segment(a))
    else:
        print(f"Using {args.num_workers} workers")
        with multiprocessing.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_segment, seg_args),
                total=len(seg_args),
                desc="Generating depth",
            ))

    total = sum(results)
    print(f"\nGenerated {total} depth maps -> {args.depth_output_dir}/{args.split}/")


if __name__ == "__main__":
    main()
