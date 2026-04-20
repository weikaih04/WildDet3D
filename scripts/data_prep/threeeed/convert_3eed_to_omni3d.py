"""Convert 3EED dataset to Omni3D JSON format.

Produces TWO datasets:
  1. Detection (3EED_det_{split}.json):
     All objects per frame (target + others merged, deduplicated).
     category_name = class label.

  2. Referring (3EED_ref_{split}.json):
     Only target objects (those with captions).
     Includes referring_expression (short phrase from GPT) and original caption.

3EED data format:
  - waymo: bbox_3d = [x, y, z, l, w, h, yaw] (7D, world frame)
  - drone/quad: bbox_3d = [x, y, z, l, w, h, yaw, pitch, roll] (9D, world frame)
  - image_extrinsic: 4x4 world-to-camera (OpenCV: X-right, Y-down, Z-forward)
  - image_intrinsic: 3x3 (waymo has 3x4 with zero 4th column)

Coordinate transform:
  - 3EED world frame: Z-up
  - 3EED extrinsic already maps world -> OpenCV camera frame
  - center_cam = extrinsic @ [x, y, z, 1]
  - R_cam = R_ext @ R_world @ P_3EED_TO_VIS4D

  3EED box-local axes (world frame, before heading rotation):
    X = length, Y = width, Z = height (up)
  vis4d canonical (camera frame):
    X = length, Y = height (down), Z = width

Usage:
    # Quick test (2 sequences per platform)
    python data_conversion/3eed/convert_3eed_to_omni3d.py --max_sequences 2

    # Full conversion
    python data_conversion/3eed/convert_3eed_to_omni3d.py

    # Only waymo
    python data_conversion/3eed/convert_3eed_to_omni3d.py --platforms waymo

Output:
    data/3eed/annotations/3EED_det_train.json
    data/3eed/annotations/3EED_det_val.json
    data/3eed/annotations/3EED_ref_train.json
    data/3eed/annotations/3EED_ref_val.json
"""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm


# -- Axis permutation: vis4d box-local -> 3EED world-local --
# vis4d box-local (OpenCV camera): X=length, Y=height(+Y=down), Z=width
# 3EED world-local (Z-up): X=length, Y=width, Z=height(+Z=up)
#
# Mapping vis4d -> 3EED:
#   3EED X (length) <- vis4d X (length):    [1, 0, 0]
#   3EED Y (width)  <- vis4d Z (width):     [0, 0, 1]
#   3EED -Z (down)  <- vis4d Y (down):      [0, -1, 0]
#
# R_cam = R_ext @ R_world @ P_VIS4D_TO_3EED
# (P maps vis4d local coords to 3EED local coords so that
#  R_world rotates them in the correct world frame)
P_VIS4D_TO_3EED = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)

# Waymo camera frame: X-forward, Y-left, Z-up
# OpenCV camera frame: X-right, Y-down, Z-forward
# Needed ONLY for Waymo (drone/quad extrinsics already output OpenCV)
R_WAYMO_CAM_TO_OPENCV = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)

# Categories
CATEGORIES = [
    {"id": 0, "name": "car", "supercategory": "vehicle"},
    {"id": 1, "name": "pedestrian", "supercategory": "person"},
    {"id": 2, "name": "bus", "supercategory": "vehicle"},
    {"id": 3, "name": "truck", "supercategory": "vehicle"},
    {"id": 4, "name": "othervehicle", "supercategory": "vehicle"},
    {"id": 5, "name": "cyclist", "supercategory": "person"},
]
CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in CATEGORIES}

# Paths are set in main() from CLI args.
DATA_DIR = None
SPLIT_DIR = None
OUTPUT_DIR = None
DEPTH_DIR = None
PHRASES_PATH = None

# Depth map encoding: uint16 PNG, depth_m * DEPTH_SCALE
DEPTH_SCALE = 256


def rotz(yaw):
    """Rotation matrix about Z-axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def roty(pitch):
    """Rotation matrix about Y-axis."""
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(roll):
    """Rotation matrix about X-axis."""
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def bbox_to_rotation(bbox_3d):
    """Build world-frame rotation matrix from bbox yaw angle.

    Only uses yaw (Z-axis rotation), consistent with 3EED's own usage
    which always does bbox[:7] even for 9D drone/quad boxes.
    """
    return rotz(bbox_3d[6])


def compute_box3d_corners(center, dimensions, R):
    """Compute 8 corners of 3D bounding box.

    vis4d canonical (OPENCV): X=length, Y=height, Z=width.
    dimensions = [w, h, l] in Omni3D format.
    """
    w, h, l = dimensions
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
    """Project 8 3D corners to 2D bounding box.

    Returns (bbox2d_proj, bbox2d_trunc).
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


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def convert_single_box(
    bbox_3d, class_name, extrinsic, K, img_w, img_h, platform
):
    """Convert a single 3EED 3D box to Omni3D annotation fields.

    Returns dict with center_cam, dimensions, R_cam, etc. or None if invalid.
    """
    bbox_3d = np.array(bbox_3d, dtype=np.float64)

    # Extract center and dims in world frame
    center_world = bbox_3d[:3]
    length, width, height = bbox_3d[3], bbox_3d[4], bbox_3d[5]

    # World-frame rotation
    R_world = bbox_to_rotation(bbox_3d)

    # Build world-to-OpenCV-camera transform
    E_w2c = build_E_w2c(extrinsic, platform)
    R_ext_cv = E_w2c[:3, :3]
    t_ext_cv = E_w2c[:3, 3]

    # Transform center to OpenCV camera frame
    center_cam = R_ext_cv @ center_world + t_ext_cv

    # Transform rotation to camera frame with vis4d axis permutation
    # R_cam maps vis4d box-local -> OpenCV camera
    # Chain: vis4d_local -> 3EED_local -> world -> camera(OpenCV)
    R_cam = R_ext_cv @ R_world @ P_VIS4D_TO_3EED

    # Dimensions: [width, height, length] in Omni3D format
    dimensions = [float(width), float(height), float(length)]

    # Check if behind camera
    is_behind = center_cam[2] <= 0

    # Compute 3D corners
    bbox3d_cam = compute_box3d_corners(
        center_cam.tolist(), dimensions, R_cam
    )

    # Project to 2D
    bbox2d_proj = [-1, -1, -1, -1]
    bbox2d_trunc = [-1, -1, -1, -1]
    truncation = -1.0

    if not is_behind:
        proj, trunc = project_box3d_to_bbox2d(bbox3d_cam, K, img_w, img_h)
        if proj is not None:
            bbox2d_proj = proj
        if trunc is not None:
            bbox2d_trunc = trunc
            area_proj = box_area(bbox2d_proj)
            area_trunc = box_area(bbox2d_trunc)
            if area_proj > 0:
                truncation = max(0.0, 1.0 - area_trunc / area_proj)
            else:
                truncation = 0.0

    # Category
    cat_name = class_name.lower()
    cat_id = CATEGORY_NAME_TO_ID.get(cat_name, -1)

    return {
        "center_cam": center_cam.tolist(),
        "dimensions": dimensions,
        "R_cam": R_cam.tolist(),
        "bbox3D_cam": bbox3d_cam,
        "bbox2D_proj": bbox2d_proj,
        "bbox2D_trunc": bbox2d_trunc,
        "bbox2D_tight": [-1, -1, -1, -1],
        "behind_camera": bool(is_behind),
        "truncation": truncation,
        "visibility": -1,
        "lidar_pts": -1,
        "segmentation_pts": -1,
        "depth_error": -1,
        "valid3D": not is_behind,
        "category_id": cat_id,
        "category_name": cat_name,
    }


def collect_all_objects(meta, platform):
    """Collect all unique objects from a frame (target + others).

    Returns list of (bbox_3d, class_name, is_target, caption_or_none).
    Deduplicates by center position (rounded to 3 decimals).
    """
    seen_centers = set()
    objects = []

    for obj in meta["ground_info"]:
        bbox = obj["bbox_3d"]
        key = tuple(round(x, 3) for x in bbox[:3])
        if key not in seen_centers:
            seen_centers.add(key)
            objects.append(
                (bbox, obj["class"], True, obj.get("caption", None))
            )

        for other in obj.get("others", []):
            bbox_o = other["bbox_3d_other"]
            key_o = tuple(round(x, 3) for x in bbox_o[:3])
            if key_o not in seen_centers:
                seen_centers.add(key_o)
                objects.append(
                    (bbox_o, other["class_other"], False, None)
                )

    return objects


def get_intrinsic_3x3(intrinsic_raw):
    """Extract 3x3 intrinsic matrix (handles both 3x3 and 3x4)."""
    K = np.array(intrinsic_raw)
    if K.shape == (3, 4):
        K = K[:, :3]
    return K


def get_image_size(platform, image_path):
    """Get image dimensions without loading the full image."""
    if platform == "waymo":
        return 1920, 1280
    else:
        return 1280, 800


def build_E_w2c(extrinsic, platform):
    """Build world-to-OpenCV-camera 4x4 transform."""
    E = np.array(extrinsic, dtype=np.float64)
    if platform == "waymo":
        axis_tf = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        return axis_tf @ np.linalg.inv(E)
    return E


def generate_depth_map(frame_dir, platform, extrinsic, K, img_w, img_h):
    """Project LiDAR to sparse depth map (uint16 PNG, scale=256)."""
    if platform == "waymo":
        lidar_path = os.path.join(frame_dir, "lidar.npy")
        pcd = np.load(lidar_path)
    else:
        lidar_path = os.path.join(frame_dir, "lidar.bin")
        pcd = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    xyz = pcd[:, :3].astype(np.float64)

    E_w2c = build_E_w2c(extrinsic, platform)
    K_np = np.array(K, dtype=np.float64)

    # World -> camera
    xyz_hom = np.hstack([xyz, np.ones((len(xyz), 1))])
    cam = (E_w2c @ xyz_hom.T).T[:, :3]

    # Keep points in front of camera
    mask = cam[:, 2] > 0.1
    cam = cam[mask]

    # Project to image
    proj = K_np @ cam.T
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]
    depth = cam[:, 2]

    # Keep points in image bounds
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[valid].astype(np.int32)
    v = v[valid].astype(np.int32)
    depth = depth[valid]

    # Build depth map (uint16, depth_m * 256)
    depth_map = np.zeros((img_h, img_w), dtype=np.uint16)
    depth_u16 = np.clip(depth * DEPTH_SCALE, 0, 65535).astype(np.uint16)
    # For overlapping pixels, keep the closer one
    for i in range(len(u)):
        if depth_map[v[i], u[i]] == 0 or depth_u16[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = depth_u16[i]

    return depth_map


def process_split(
    platforms, split, short_phrases, max_sequences, dataset_id,
    generate_depth=True,
):
    """Process a single split (train/val) and return (images, det_annos, ref_annos, categories)."""

    images = []
    det_annotations = []
    ref_annotations = []

    image_id = 0
    det_anno_id = 0
    ref_anno_id = 0
    class_counter = Counter()

    for platform in platforms:
        platform_dir = os.path.join(DATA_DIR, platform)
        if not os.path.exists(platform_dir):
            print(f"  Skipping {platform}: directory not found")
            continue

        # Load split file
        split_file = os.path.join(SPLIT_DIR, f"{platform}_{split}.txt")
        if not os.path.exists(split_file):
            print(f"  Skipping {platform}/{split}: split file not found")
            continue

        with open(split_file) as f:
            sequences = [line.strip() for line in f if line.strip()]

        if max_sequences > 0:
            sequences = sequences[:max_sequences]

        print(f"  {platform}/{split}: {len(sequences)} sequences")

        for seq_name in tqdm(
            sequences, desc=f"  {platform}/{split}"
        ):
            seq_dir = os.path.join(platform_dir, seq_name)
            if not os.path.exists(seq_dir):
                continue

            # List frame directories
            frame_dirs = sorted(
                [
                    d
                    for d in os.listdir(seq_dir)
                    if os.path.isdir(os.path.join(seq_dir, d))
                ]
            )

            for frame_name in frame_dirs:
                frame_dir = os.path.join(seq_dir, frame_name)
                meta_path = os.path.join(frame_dir, "meta_info.json")
                image_path = os.path.join(frame_dir, "image.jpg")

                if not os.path.exists(meta_path):
                    continue
                if not os.path.exists(image_path):
                    continue

                with open(meta_path) as f:
                    meta = json.load(f)

                # Camera params
                extrinsic = meta["image_extrinsic"]
                K = get_intrinsic_3x3(meta["image_intrinsic"])
                img_w, img_h = get_image_size(platform, image_path)

                # Relative image path for Omni3D
                rel_image_path = os.path.join(
                    "3eed/3eed_dataset",
                    platform,
                    seq_name,
                    frame_name,
                    "image.jpg",
                )

                # Image entry
                img_entry = {
                    "id": image_id,
                    "file_path": rel_image_path,
                    "height": img_h,
                    "width": img_w,
                    "K": K.tolist(),
                    "src_90_rotate": 0,
                    "src_flagged": False,
                    "dataset_id": dataset_id,
                    "platform": platform,
                    "sequence": seq_name,
                    "frame": frame_name,
                }
                images.append(img_entry)

                # Generate sparse depth map from LiDAR
                if generate_depth:
                    rel_depth_path = os.path.join(
                        "3eed", "depth",
                        platform, seq_name, f"{frame_name}.png",
                    )
                    depth_out_path = os.path.join(
                        DEPTH_DIR,
                        platform, seq_name, f"{frame_name}.png",
                    )
                    if not os.path.exists(depth_out_path):
                        os.makedirs(
                            os.path.dirname(depth_out_path), exist_ok=True
                        )
                        depth_map = generate_depth_map(
                            frame_dir, platform, extrinsic,
                            K.tolist(), img_w, img_h,
                        )
                        cv2.imwrite(depth_out_path, depth_map)

                # Collect all objects for detection
                all_objects = collect_all_objects(meta, platform)

                for bbox_3d, cls_name, is_target, caption in all_objects:
                    ann = convert_single_box(
                        bbox_3d,
                        cls_name,
                        extrinsic,
                        K.tolist(),
                        img_w,
                        img_h,
                        platform,
                    )
                    if ann is None:
                        continue

                    class_counter[ann["category_name"]] += 1

                    # Detection annotation
                    det_ann = dict(ann)
                    det_ann["id"] = det_anno_id
                    det_ann["image_id"] = image_id
                    det_ann["dataset_id"] = dataset_id
                    det_annotations.append(det_ann)
                    det_anno_id += 1

                    # Referring annotation (only for targets)
                    if is_target and caption is not None:
                        ref_ann = dict(ann)
                        ref_ann["id"] = ref_anno_id
                        ref_ann["image_id"] = image_id
                        ref_ann["dataset_id"] = dataset_id
                        ref_ann["caption"] = caption
                        ref_ann["referring_expression"] = (
                            short_phrases.get(caption, ann["category_name"])
                        )
                        ref_annotations.append(ref_ann)
                        ref_anno_id += 1

                image_id += 1

    print(f"\n  Split {split} summary:")
    print(f"    Images: {len(images)}")
    print(f"    Detection annotations: {len(det_annotations)}")
    print(f"    Referring annotations: {len(ref_annotations)}")
    print(f"    Class distribution: {dict(class_counter)}")

    return images, det_annotations, ref_annotations


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3EED to Omni3D format"
    )
    parser.add_argument(
        "--data_root",
        default="data/3eed",
        help="3EED dataset root (contains 3eed_dataset/, depth/, short_phrases.json)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output dir for annotation JSONs (default: <data_root>/annotations)",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["waymo", "drone", "quad"],
        help="Platforms to convert",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to convert",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=0,
        help="Max sequences per platform (0=all)",
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=10,
        help="Dataset ID for Omni3D",
    )
    parser.add_argument(
        "--no_depth",
        action="store_true",
        help="Skip depth map generation",
    )
    args = parser.parse_args()

    # Initialize path globals from CLI args.
    global DATA_DIR, SPLIT_DIR, OUTPUT_DIR, DEPTH_DIR, PHRASES_PATH
    DATA_DIR = os.path.join(args.data_root, "3eed_dataset")
    SPLIT_DIR = os.path.join(args.data_root, "3eed_dataset", "splits")
    OUTPUT_DIR = args.output_dir or os.path.join(args.data_root, "annotations")
    DEPTH_DIR = os.path.join(args.data_root, "depth")
    PHRASES_PATH = os.path.join(args.data_root, "short_phrases.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load short phrases
    short_phrases = {}
    if os.path.exists(PHRASES_PATH):
        with open(PHRASES_PATH) as f:
            short_phrases = json.load(f)
        print(f"Loaded {len(short_phrases)} short phrases")
    else:
        print(
            f"WARNING: {PHRASES_PATH} not found. "
            "Referring expressions will fallback to class names."
        )

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

        images, det_annos, ref_annos = process_split(
            args.platforms,
            split,
            short_phrases,
            args.max_sequences,
            args.dataset_id,
            generate_depth=not args.no_depth,
        )

        # Save detection JSON
        det_output = {
            "images": images,
            "annotations": det_annos,
            "categories": CATEGORIES,
        }
        det_path = os.path.join(OUTPUT_DIR, f"3EED_det_{split}.json")
        with open(det_path, "w") as f:
            json.dump(det_output, f)
        print(f"  Saved detection: {det_path}")

        # Save referring JSON
        ref_output = {
            "images": images,
            "annotations": ref_annos,
            "categories": CATEGORIES,
        }
        ref_path = os.path.join(OUTPUT_DIR, f"3EED_ref_{split}.json")
        with open(ref_path, "w") as f:
            json.dump(ref_output, f)
        print(f"  Saved referring: {ref_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
