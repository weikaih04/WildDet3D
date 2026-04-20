"""Convert Waymo Open Dataset v2 (Parquet) to Omni3D JSON format.

Reads Waymo v2 Parquet files directly (no TensorFlow needed).
Only processes FRONT camera images.
Processes per-segment to avoid loading all data into memory.

Requires these Waymo v2 components (download with download_waymo_v2_all.sh):
    - camera_image:       FRONT camera JPEG images
    - camera_calibration: intrinsics + extrinsics (per-segment)
    - lidar_box:          3D bounding box annotations (vehicle frame)
    - camera_box:         Human-annotated 2D tight boxes (for bbox2D_tight)
    - camera_to_lidar_box_association: links camera_box to lidar_box (ped/cyclist only)

Coordinate transform chain:
    Vehicle frame -> Waymo camera frame (extrinsic inverse)
                  -> OpenCV camera frame (R_WAYMO_CAM_TO_OPENCV)
    Box rotation:  R_cam = R_waymo2cv @ R_v2c @ R_heading @ P_WAYMO_BOX_TO_VIS4D

bbox2D_tight matching:
    - Pedestrian/Cyclist: exact match via association table (laser_object_id)
    - Vehicle: greedy IoU matching (same type, IoU descending, one-to-one)
    - Sign: no camera_box annotation, set to [-1,-1,-1,-1]

No filtering is applied here -- all annotations are kept exhaustively.
Filtering (truncation, visibility, depth, etc.) is handled by the COCO3D dataloader.

Usage:
    # Quick test (2 segments)
    python data_conversion/waymo/convert_waymo_v2_to_omni3d.py \
        --split validation --max_segments 2

    # Full validation (every 5th frame, ~8K images, 64 workers)
    python data_conversion/waymo/convert_waymo_v2_to_omni3d.py \
        --split validation --frame_interval 5 --num_workers 64

    # Full training (every 5th frame, ~32K images, 64 workers)
    python data_conversion/waymo/convert_waymo_v2_to_omni3d.py \
        --split training --frame_interval 5 --num_workers 64

Output:
    data/waymo/annotations/Waymo_val.json   (or Waymo_train.json)
    data/waymo/images/{split}/*.jpg

Expected Waymo v2 directory structure (after download_waymo_v2_all.sh):
    data/waymo_v2/
    ├── training/
    │   ├── camera_image/
    │   ├── camera_calibration/
    │   ├── camera_box/
    │   ├── camera_to_lidar_box_association/
    │   └── lidar_box/
    └── validation/
        ├── camera_image/
        ├── camera_calibration/
        ├── camera_box/
        ├── camera_to_lidar_box_association/
        └── lidar_box/
"""

import argparse
import json
import math
import multiprocessing
import os
from collections import Counter
from functools import partial

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


# -- Waymo camera names --
FRONT_CAMERA = 1

# -- Waymo label types --
WAYMO_TYPE_TO_CATEGORY = {
    1: "vehicle",
    2: "pedestrian",
    3: "sign",
    4: "cyclist",
}

WAYMO_CATEGORIES = [
    {"id": 0, "name": "vehicle", "supercategory": "vehicle"},
    {"id": 1, "name": "pedestrian", "supercategory": "person"},
    {"id": 2, "name": "cyclist", "supercategory": "person"},
    {"id": 3, "name": "sign", "supercategory": "object"},
]

CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in WAYMO_CATEGORIES}

# -- Column names (Waymo v2 Parquet) --
SEG_COL = "key.segment_context_name"
TS_COL = "key.frame_timestamp_micros"
CAM_NAME_COL = "key.camera_name"
IMAGE_COL = "[CameraImageComponent].image"

FU_COL = "[CameraCalibrationComponent].intrinsic.f_u"
FV_COL = "[CameraCalibrationComponent].intrinsic.f_v"
CU_COL = "[CameraCalibrationComponent].intrinsic.c_u"
CV_COL = "[CameraCalibrationComponent].intrinsic.c_v"
EXTRINSIC_COL = "[CameraCalibrationComponent].extrinsic.transform"
WIDTH_COL = "[CameraCalibrationComponent].width"
HEIGHT_COL = "[CameraCalibrationComponent].height"

BOX_TYPE_COL = "[LiDARBoxComponent].type"
CX_COL = "[LiDARBoxComponent].box.center.x"
CY_COL = "[LiDARBoxComponent].box.center.y"
CZ_COL = "[LiDARBoxComponent].box.center.z"
SIZE_X_COL = "[LiDARBoxComponent].box.size.x"
SIZE_Y_COL = "[LiDARBoxComponent].box.size.y"
SIZE_Z_COL = "[LiDARBoxComponent].box.size.z"
HEADING_COL = "[LiDARBoxComponent].box.heading"
NUM_LIDAR_COL = "[LiDARBoxComponent].num_lidar_points_in_box"

CAMBOX_CX_COL = "[CameraBoxComponent].box.center.x"
CAMBOX_CY_COL = "[CameraBoxComponent].box.center.y"
CAMBOX_SX_COL = "[CameraBoxComponent].box.size.x"
CAMBOX_SY_COL = "[CameraBoxComponent].box.size.y"
CAMBOX_TYPE_COL = "[CameraBoxComponent].type"

LASER_OBJ_ID_COL = "key.laser_object_id"
CAMERA_OBJ_ID_COL = "key.camera_object_id"

# Waymo camera frame: X-forward, Y-left, Z-up
# OpenCV camera frame: X-right, Y-down, Z-forward
R_WAYMO_CAM_TO_OPENCV = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

# Waymo box-local axes: X=length(heading), Y=width, Z=height(up)
# vis4d canonical OPENCV: X=length, Y=height(down), Z=width
#
# Mapping:
#   canonical X (length)       -> Waymo X (length)         : [1,0,0]
#   canonical Y (height, +Y=down) -> Waymo -Z (height, -Z=down) : [0,0,-1]
#   canonical Z (width)        -> Waymo Y (width)          : [0,1,0]
#
# det = +1 (proper rotation, not reflection)
P_WAYMO_BOX_TO_VIS4D = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def compute_box3d_corners(center, dimensions, R):
    """Compute 8 corners of 3D bounding box.

    vis4d canonical (OPENCV): X=length, Y=height, Z=width.
    dimensions = [w, h, l] in Omni3D format.
    Corner ordering matches vis4d's boxes3d_to_corners().
    """
    w, h, l = dimensions
    # X = length, Y = height, Z = width
    corners_local = np.array(
        [
            [l / 2, h / 2, -w / 2],
            [l / 2, h / 2, w / 2],
            [-l / 2, h / 2, -w / 2],
            [-l / 2, h / 2, w / 2],
            [l / 2, -h / 2, -w / 2],
            [l / 2, -h / 2, w / 2],
            [-l / 2, -h / 2, -w / 2],
            [-l / 2, -h / 2, w / 2],
        ]
    )
    corners_cam = (R @ corners_local.T).T + np.array(center)
    return corners_cam.tolist()


def project_box3d_to_bbox2d(corners_3d, K, img_w, img_h):
    """Project 8 3D corners to 2D and compute both unclipped and clipped boxes.

    Returns (bbox2d_proj, bbox2d_trunc):
        bbox2d_proj:  [x1, y1, x2, y2] unclipped (full projection)
        bbox2d_trunc: [x1, y1, x2, y2] clipped to image bounds
    Returns (None, None) if any corner is behind camera.
    """
    corners = np.array(corners_3d)
    if np.any(corners[:, 2] <= 0):
        return None, None
    K_np = np.array(K)
    projected = K_np @ corners.T  # 3x8
    projected_2d = projected[:2, :] / projected[2:, :]  # 2x8

    px1 = float(projected_2d[0].min())
    py1 = float(projected_2d[1].min())
    px2 = float(projected_2d[0].max())
    py2 = float(projected_2d[1].max())
    bbox2d_proj = [px1, py1, px2, py2]

    tx1 = max(0.0, px1)
    ty1 = max(0.0, py1)
    tx2 = min(float(img_w), px2)
    ty2 = min(float(img_h), py2)
    if tx2 <= tx1 or ty2 <= ty1:
        return bbox2d_proj, None
    bbox2d_trunc = [tx1, ty1, tx2, ty2]

    return bbox2d_proj, bbox2d_trunc


def box_area(bbox):
    """Compute area of [x1, y1, x2, y2] box."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def box_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_camera_boxes_greedy(lidar_entries, cam_boxes_at_ts):
    """Greedy match lidar projected boxes with camera_box by IoU.

    One-to-one matching: each camera_box can only be matched once.
    Sorted by IoU descending so best matches are assigned first.

    Args:
        lidar_entries: list of (index, bbox2d_trunc, box_type)
        cam_boxes_at_ts: list of camera_box rows for this timestamp

    Returns dict: lidar_index -> [x1, y1, x2, y2] matched tight box.
    """
    cam_entries = []
    for cb in cam_boxes_at_ts:
        cx = float(cb[CAMBOX_CX_COL])
        cy = float(cb[CAMBOX_CY_COL])
        sx = float(cb[CAMBOX_SX_COL])
        sy = float(cb[CAMBOX_SY_COL])
        cb_box = [cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2]
        cb_type = int(cb[CAMBOX_TYPE_COL])
        cam_entries.append((cb_box, cb_type))

    # Compute all pairwise IoU (same type only)
    pairs = []
    for li, (l_idx, l_box, l_type) in enumerate(lidar_entries):
        for ci, (c_box, c_type) in enumerate(cam_entries):
            if l_type != c_type:
                continue
            iou = box_iou(l_box, c_box)
            if iou > 0:
                pairs.append((iou, li, ci))

    # Greedy: sort by IoU descending, assign one-to-one
    pairs.sort(reverse=True)
    matched_lidar = set()
    matched_cam = set()
    result = {}
    for iou, li, ci in pairs:
        if li in matched_lidar or ci in matched_cam:
            continue
        l_idx = lidar_entries[li][0]
        result[l_idx] = cam_entries[ci][0]
        matched_lidar.add(li)
        matched_cam.add(ci)

    return result


def heading_to_rotation_matrix(heading):
    """Convert heading angle to 3x3 rotation matrix in vehicle frame."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.array(
        [[cos_h, -sin_h, 0.0], [sin_h, cos_h, 0.0], [0.0, 0.0, 1.0]]
    )


def convert_box_to_opencv_cam(box_center, box_dims, heading, T_cam_to_vehicle):
    """Convert a 3D box from Waymo vehicle frame to OpenCV camera frame.

    Pipeline:
      1. Vehicle frame -> Waymo camera frame (extrinsic inverse)
      2. Waymo camera frame -> OpenCV camera frame (R_WAYMO_CAM_TO_OPENCV)

    For rotation, we also apply P_WAYMO_BOX_TO_VIS4D because:
      - Waymo box-local: X=length, Y=width, Z=height
      - vis4d canonical:  X=length, Y=height, Z=width
    So R_cam must include the Y<->Z swap so that vis4d's canonical Y
    (height axis) maps to the vertical direction in camera frame.

    Full rotation: R_cam = R_waymo2cv @ R_v2c @ R_heading @ P_swap
    """
    T_v2c = np.linalg.inv(T_cam_to_vehicle)
    R_v2c = T_v2c[:3, :3]
    t_v2c = T_v2c[:3, 3]

    # Vehicle -> Waymo camera -> OpenCV camera
    center_waymo = R_v2c @ np.array(box_center) + t_v2c
    center_cv = R_WAYMO_CAM_TO_OPENCV @ center_waymo

    R_obj_v = heading_to_rotation_matrix(heading)
    R_obj_cv = R_WAYMO_CAM_TO_OPENCV @ R_v2c @ R_obj_v @ P_WAYMO_BOX_TO_VIS4D

    # Waymo: [length, width, height] -> Omni3D: [w, h, l]
    length, width, height = box_dims
    dimensions = [width, height, length]

    return center_cv.tolist(), dimensions, R_obj_cv.tolist()


def process_segment(
    seg_file, split_dir, split, image_output_dir,
    calib_lookup, dataset_id, frame_interval,
    img_id_start=0, ann_id_start=0,
):
    """Process a single segment.

    Uses local IDs starting from img_id_start/ann_id_start.
    Returns (images, annotations, skipped_types, next_img_id, next_ann_id).
    """
    seg_name = os.path.splitext(os.path.basename(seg_file))[0]

    # Load camera images for this segment
    cam_path = os.path.join(split_dir, "camera_image", seg_file)
    cam_df = pq.read_table(cam_path).to_pandas()
    cam_df = cam_df[cam_df[CAM_NAME_COL] == FRONT_CAMERA].reset_index(drop=True)

    if len(cam_df) == 0:
        return [], [], Counter(), img_id_start, ann_id_start

    # Load boxes for this segment
    box_path = os.path.join(split_dir, "lidar_box", seg_file)
    if not os.path.exists(box_path):
        return [], [], Counter(), img_id_start, ann_id_start
    box_df = pq.read_table(box_path).to_pandas()

    # Build box lookup by timestamp
    box_by_ts = {}
    for _, row in box_df.iterrows():
        ts = row[TS_COL]
        if ts not in box_by_ts:
            box_by_ts[ts] = []
        box_by_ts[ts].append(row)

    # Load camera_box for this segment
    # - keyed by (ts, camera_object_id) for association lookup
    # - also grouped by ts for IoU fallback
    cambox_by_cam_id = {}
    cambox_by_ts = {}
    cambox_path = os.path.join(split_dir, "camera_box", seg_file)
    if os.path.exists(cambox_path):
        cambox_df = pq.read_table(cambox_path).to_pandas()
        cambox_df = cambox_df[cambox_df[CAM_NAME_COL] == FRONT_CAMERA]
        for _, row in cambox_df.iterrows():
            cam_obj_id = row[CAMERA_OBJ_ID_COL]
            ts = row[TS_COL]
            cambox_by_cam_id[(ts, cam_obj_id)] = row
            if ts not in cambox_by_ts:
                cambox_by_ts[ts] = []
            cambox_by_ts[ts].append(row)

    # Load association: (ts, laser_object_id) -> camera_object_id
    # Only covers pedestrian/cyclist in Waymo v2
    laser_to_cam = {}
    assoc_path = os.path.join(
        split_dir, "camera_to_lidar_box_association", seg_file
    )
    if os.path.exists(assoc_path):
        assoc_df = pq.read_table(assoc_path).to_pandas()
        assoc_df = assoc_df[assoc_df[CAM_NAME_COL] == FRONT_CAMERA]
        for _, row in assoc_df.iterrows():
            ts = row[TS_COL]
            laser_id = row[LASER_OBJ_ID_COL]
            cam_id = row[CAMERA_OBJ_ID_COL]
            laser_to_cam[(ts, laser_id)] = cam_id

    # Get calibration
    if seg_name not in calib_lookup:
        return [], [], Counter(), img_id_start, ann_id_start
    calib = calib_lookup[seg_name]

    img_w = calib["width"]
    img_h = calib["height"]
    K = [
        [calib["fu"], 0.0, calib["cu"]],
        [0.0, calib["fv"], calib["cv"]],
        [0.0, 0.0, 1.0],
    ]
    T_cam_to_vehicle = np.array(calib["extrinsic"]).reshape(4, 4)

    # Sort by timestamp and apply frame interval
    cam_df = cam_df.sort_values(TS_COL).reset_index(drop=True)
    if frame_interval > 1:
        cam_df = cam_df.iloc[::frame_interval].reset_index(drop=True)

    images = []
    annotations = []
    skipped_types = Counter()
    img_id = img_id_start
    ann_id = ann_id_start

    for _, cam_row in cam_df.iterrows():
        ts = cam_row[TS_COL]
        image_bytes = cam_row[IMAGE_COL]
        if image_bytes is None:
            continue

        # Save image
        img_filename = f"{seg_name}_{ts}.jpg"
        rel_img_path = os.path.join("waymo", "images", split, img_filename)
        abs_img_path = os.path.join(image_output_dir, split, img_filename)
        os.makedirs(os.path.dirname(abs_img_path), exist_ok=True)
        with open(abs_img_path, "wb") as f:
            f.write(image_bytes)

        images.append({
            "width": img_w,
            "height": img_h,
            "file_path": rel_img_path,
            "K": K,
            "src_90_rotate": 0,
            "src_flagged": False,
            "incomplete": False,
            "id": img_id,
            "dataset_id": dataset_id,
        })

        # Process 3D boxes - no filtering, let the dataloader decide.
        # Two passes: 1) convert all boxes, 2) greedy match tight boxes.
        frame_boxes = box_by_ts.get(ts, [])
        frame_anns = []
        greedy_entries = []  # (index, bbox2d_trunc, box_type)

        for box_row in frame_boxes:
            box_type = int(box_row[BOX_TYPE_COL])
            if box_type not in WAYMO_TYPE_TO_CATEGORY:
                skipped_types[box_type] += 1
                continue

            cat_name = WAYMO_TYPE_TO_CATEGORY[box_type]
            cat_id = CATEGORY_NAME_TO_ID[cat_name]
            lidar_pts = int(box_row[NUM_LIDAR_COL])
            laser_obj_id = box_row[LASER_OBJ_ID_COL]

            box_center_v = [
                float(box_row[CX_COL]),
                float(box_row[CY_COL]),
                float(box_row[CZ_COL]),
            ]
            box_dims_v = [
                float(box_row[SIZE_X_COL]),
                float(box_row[SIZE_Y_COL]),
                float(box_row[SIZE_Z_COL]),
            ]
            heading = float(box_row[HEADING_COL])

            center_cam, dimensions, R_cam = convert_box_to_opencv_cam(
                box_center_v, box_dims_v, heading, T_cam_to_vehicle
            )

            is_behind = center_cam[2] <= 0
            bbox3D_cam = compute_box3d_corners(center_cam, dimensions, R_cam)

            bbox2d_proj = [-1, -1, -1, -1]
            bbox2d_trunc = [-1, -1, -1, -1]
            bbox2d_tight = [-1, -1, -1, -1]
            truncation = -1.0
            visibility = -1.0

            if not is_behind:
                proj, trunc = project_box3d_to_bbox2d(
                    bbox3D_cam, K, img_w, img_h
                )
                if proj is not None:
                    bbox2d_proj = proj
                if trunc is not None:
                    bbox2d_trunc = trunc

                    # Compute truncation from proj vs trunc areas
                    area_proj = box_area(bbox2d_proj)
                    area_trunc = box_area(bbox2d_trunc)
                    if area_proj > 0:
                        truncation = 1.0 - area_trunc / area_proj
                    else:
                        truncation = 0.0

                    # Try association first (exact for ped/cyclist)
                    cam_obj_id = laser_to_cam.get(
                        (ts, laser_obj_id)
                    )
                    if cam_obj_id is not None:
                        cam_row = cambox_by_cam_id.get(
                            (ts, cam_obj_id)
                        )
                        if cam_row is not None:
                            cx = float(cam_row[CAMBOX_CX_COL])
                            cy = float(cam_row[CAMBOX_CY_COL])
                            sx = float(cam_row[CAMBOX_SX_COL])
                            sy = float(cam_row[CAMBOX_SY_COL])
                            bbox2d_tight = [
                                cx - sx / 2, cy - sy / 2,
                                cx + sx / 2, cy + sy / 2,
                            ]

                    # Queue for greedy matching if no association
                    if bbox2d_tight == [-1, -1, -1, -1]:
                        greedy_entries.append(
                            (len(frame_anns), bbox2d_trunc, box_type)
                        )

            frame_anns.append({
                "behind_camera": is_behind,
                "truncation": truncation,
                "bbox2D_tight": bbox2d_tight,
                "visibility": visibility,
                "segmentation_pts": -1,
                "lidar_pts": lidar_pts,
                "valid3D": True,
                "depth_error": -1,
                "category_id": cat_id,
                "category_name": cat_name,
                "id": ann_id,
                "image_id": img_id,
                "dataset_id": dataset_id,
                "bbox2D_proj": bbox2d_proj,
                "bbox2D_trunc": bbox2d_trunc,
                "center_cam": center_cam,
                "dimensions": dimensions,
                "R_cam": R_cam,
                "bbox3D_cam": bbox3D_cam,
            })
            ann_id += 1

        # Greedy match remaining boxes by IoU
        if greedy_entries:
            cam_boxes = cambox_by_ts.get(ts, [])
            matched = match_camera_boxes_greedy(
                greedy_entries, cam_boxes
            )
            for local_idx, tight_box in matched.items():
                frame_anns[local_idx]["bbox2D_tight"] = tight_box

        # Compute visibility for annotations that have tight box
        for ann in frame_anns:
            if ann["bbox2D_tight"] != [-1, -1, -1, -1]:
                area_tight = box_area(ann["bbox2D_tight"])
                area_trunc = box_area(ann["bbox2D_trunc"])
                if area_trunc > 0:
                    ann["visibility"] = min(1.0, area_tight / area_trunc)

        annotations.extend(frame_anns)

        img_id += 1

    return images, annotations, skipped_types, img_id, ann_id


def _process_segment_wrapper(args):
    """Wrapper for multiprocessing (must be top-level picklable)."""
    return process_segment(*args)


def convert(waymo_root, output_dir, split, image_output_dir, dataset_id=100,
            max_segments=None, frame_interval=1, num_workers=1):
    """Main conversion: Waymo v2 Parquet -> Omni3D JSON.

    Supports parallel processing with num_workers > 1.
    Each segment is processed independently, then global IDs are reassigned.
    """
    split_dir = os.path.join(waymo_root, split)

    # List segment files
    cam_dir = os.path.join(split_dir, "camera_image")
    seg_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(".parquet")])
    print(f"Found {len(seg_files)} segments for {split}")

    if max_segments is not None and max_segments > 0:
        seg_files = seg_files[:max_segments]
        print(f"Limiting to {max_segments} segments")

    # Load all calibration (small: ~200 rows)
    print("Loading camera calibration...")
    calib_dir = os.path.join(split_dir, "camera_calibration")
    calib_files = sorted(
        [os.path.join(calib_dir, f) for f in os.listdir(calib_dir) if f.endswith(".parquet")]
    )
    import pandas as pd
    calib_dfs = [pq.read_table(f).to_pandas() for f in calib_files]
    calib_df = pd.concat(calib_dfs, ignore_index=True)
    calib_df = calib_df[calib_df[CAM_NAME_COL] == FRONT_CAMERA].reset_index(drop=True)
    print(f"  FRONT camera calibration rows: {len(calib_df)}")

    calib_lookup = {}
    for _, row in calib_df.iterrows():
        seg = row[SEG_COL]
        calib_lookup[seg] = {
            "fu": float(row[FU_COL]),
            "fv": float(row[FV_COL]),
            "cu": float(row[CU_COL]),
            "cv": float(row[CV_COL]),
            "extrinsic": [float(x) for x in row[EXTRINSIC_COL]],
            "width": int(row[WIDTH_COL]),
            "height": int(row[HEIGHT_COL]),
        }

    info = {
        "id": dataset_id,
        "source": "Waymo",
        "name": f"Waymo {split.capitalize()}",
        "split": split.capitalize(),
        "version": "0.1",
        "url": "https://waymo.com/open/",
    }

    # Build args for each segment (local IDs start from 0)
    seg_args = [
        (seg_file, split_dir, split, image_output_dir,
         calib_lookup, dataset_id, frame_interval, 0, 0)
        for seg_file in seg_files
    ]

    # Process segments
    if num_workers <= 1:
        results = []
        for args in tqdm(seg_args, desc=f"Converting {split}"):
            results.append(_process_segment_wrapper(args))
    else:
        print(f"Using {num_workers} workers")
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_segment_wrapper, seg_args),
                total=len(seg_args),
                desc=f"Converting {split}",
            ))

    # Merge results and reassign global IDs
    all_images = []
    all_annotations = []
    all_skipped = Counter()
    global_img_id = 0
    global_ann_id = 0

    for images, anns, skipped, _, _ in results:
        # Build local->global image ID mapping for this segment
        local_to_global_img = {}
        for img in images:
            local_img_id = img["id"]
            local_to_global_img[local_img_id] = global_img_id
            img["id"] = global_img_id
            global_img_id += 1

        for ann in anns:
            ann["id"] = global_ann_id
            ann["image_id"] = local_to_global_img[ann["image_id"]]
            global_ann_id += 1

        all_images.extend(images)
        all_annotations.extend(anns)
        all_skipped += skipped

    # Summary
    print(f"\nConversion summary:")
    print(f"  Segments: {len(seg_files)}")
    print(f"  Images: {len(all_images)}")
    print(f"  Annotations: {len(all_annotations)}")
    cat_counts = Counter(a["category_name"] for a in all_annotations)
    print(f"  Category distribution: {dict(cat_counts)}")
    if all_skipped:
        print(f"  Skipped Waymo types: {dict(all_skipped)}")

    # Save JSON
    split_name = "train" if split == "training" else "val"
    output = {
        "info": info,
        "categories": WAYMO_CATEGORIES,
        "images": all_images,
        "annotations": all_annotations,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"Waymo_{split_name}.json")
    print(f"Saving to {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(output, f)
    print("Done!")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Waymo v2 Parquet to Omni3D JSON"
    )
    parser.add_argument(
        "--waymo_root", type=str, default="data/waymo_v2",
        help="Root of Waymo v2 Parquet data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/waymo/annotations",
        help="Output directory for Omni3D JSON",
    )
    parser.add_argument(
        "--image_output_dir", type=str, default="data/waymo/images",
        help="Directory to extract/save JPEG images",
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        choices=["training", "validation"],
        help="Waymo split to convert",
    )
    parser.add_argument(
        "--dataset_id", type=int, default=100,
        help="Dataset ID (avoid Omni3D 0-17)",
    )
    parser.add_argument(
        "--max_segments", type=int, default=None,
        help="Max segments to convert (for testing)",
    )
    parser.add_argument(
        "--frame_interval", type=int, default=1,
        help="Process every Nth frame (10Hz -> set 5 for 2Hz)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers (default: 1, set to e.g. 64 for faster conversion)",
    )
    args = parser.parse_args()

    convert(
        waymo_root=args.waymo_root,
        output_dir=args.output_dir,
        split=args.split,
        image_output_dir=args.image_output_dir,
        dataset_id=args.dataset_id,
        max_segments=args.max_segments,
        frame_interval=args.frame_interval,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
