"""Fast FoundationPose conversion with gravity correction.

Uses multiprocessing to parallelize JSON loading across NFS.
Skips depth conversion (use --no_depth on original script for that).

Usage:
    python data_conversion/foundationpose/convert_foundationpose_fast.py
    python data_conversion/foundationpose/convert_foundationpose_fast.py --max_images 1000
"""

import argparse
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# Paths are set in main() from CLI args.
BBOX3D_DIR = None
EXTRACTED_DIR = None
OUTPUT_DIR = None

IMG_W, IMG_H = 640, 480

# FP local -> vis4d local axis mapping
P_FP_TO_VIS4D = np.array(
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64
)


def get_gravity_opencv(cam_data):
    """Extract gravity direction in OpenCV camera frame.

    Isaac Sim uses Z-up world. Gravity = [0, 0, -1] in world.
    cameraViewTransform: world-to-OpenGL (row-vector convention).
    """
    V = np.array(cam_data["cameraViewTransform"]).reshape(4, 4)
    R_w2gl = V[:3, :3].T
    grav_gl = R_w2gl @ np.array([0.0, 0.0, -1.0])
    return np.array([grav_gl[0], -grav_gl[1], -grav_gl[2]])


def align_rotation_to_gravity(R_obj, dims_whl, gravity_cam):
    """Correct rotation so local Y aligns with gravity + force W <= L."""
    g = gravity_cam / np.linalg.norm(gravity_cam)
    abs_dots = [
        abs(np.dot(R_obj[:, 0], g)),
        abs(np.dot(R_obj[:, 1], g)),
        abs(np.dot(R_obj[:, 2], g)),
    ]
    best_axis = int(np.argmax(abs_dots))

    R_out = R_obj.copy()
    w, h, l = dims_whl

    if best_axis == 0:
        R_out[:, 0] = R_obj[:, 1]
        R_out[:, 1] = R_obj[:, 0]
        l, h = h, l
        R_out[:, 2] = -R_out[:, 2]
    elif best_axis == 2:
        R_out[:, 2] = R_obj[:, 1]
        R_out[:, 1] = R_obj[:, 2]
        w, h = h, w
        R_out[:, 0] = -R_out[:, 0]

    if np.dot(R_out[:, 1], g) < 0:
        R_out[:, 1] = -R_out[:, 1]
        R_out[:, 2] = -R_out[:, 2]

    if w > l:
        w, l = l, w
        col0 = R_out[:, 0].copy()
        R_out[:, 0] = -R_out[:, 2]
        R_out[:, 2] = col0

    return R_out, [w, h, l]


def get_intrinsic(cam_data):
    """Extract 3x3 intrinsic matrix from camera params."""
    P = np.array(cam_data["cameraProjection"]).reshape(4, 4)
    fx = P[0, 0] * IMG_W / 2
    fy = P[1, 1] * IMG_H / 2
    cx = IMG_W / 2
    cy = IMG_H / 2
    return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]


def compute_box3d_corners(center, dimensions, R_cam):
    """Compute 8 corners of 3D bounding box."""
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
    return ((R_cam @ corners_local.T).T + np.array(center)).tolist()


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def project_box3d_to_bbox2d(corners_3d, K):
    """Project 8 3D corners to 2D bounding box."""
    corners = np.array(corners_3d)
    if np.any(corners[:, 2] <= 0):
        return None, None
    K_np = np.array(K)
    projected = K_np @ corners.T
    projected_2d = projected[:2, :] / projected[2:, :]

    px1 = float(projected_2d[0].min())
    py1 = float(projected_2d[1].min())
    px2 = float(projected_2d[0].max())
    py2 = float(projected_2d[1].max())
    bbox2d_proj = [px1, py1, px2, py2]

    tx1 = max(0.0, px1)
    ty1 = max(0.0, py1)
    tx2 = min(float(IMG_W), px2)
    ty2 = min(float(IMG_H), py2)
    if tx2 <= tx1 or ty2 <= ty1:
        return bbox2d_proj, None
    bbox2d_trunc = [tx1, ty1, tx2, ty2]

    return bbox2d_proj, bbox2d_trunc


def process_single_file(bbox_file):
    """Process one bbox3d file -> (img_entry, [ann_entries], [categories])."""
    bbox_path = os.path.join(BBOX3D_DIR, bbox_file)
    with open(bbox_path) as f:
        data = json.load(f)

    name = data["image_id"]

    # Check image exists
    img_path = os.path.join(EXTRACTED_DIR, "images", "gso", f"{name}.png")
    if not os.path.exists(img_path):
        return None

    # Load camera params
    cam_path = os.path.join(
        EXTRACTED_DIR, "camera_params", "gso", f"{name}.json"
    )
    if not os.path.exists(cam_path):
        return None
    with open(cam_path) as f:
        cam_data = json.load(f)

    K = get_intrinsic(cam_data)
    gravity_cam = get_gravity_opencv(cam_data)

    rel_image_path = os.path.join(
        "foundationpose", "images", "gso", f"{name}.png"
    )
    img_entry = {
        "file_path": rel_image_path,
        "height": IMG_H,
        "width": IMG_W,
        "K": K,
        "src_90_rotate": 0,
        "src_flagged": False,
    }

    anns = []
    cats = []
    for obj_idx in range(len(data["boxes3d"])):
        box3d = data["boxes3d"][obj_idx][0]["box3d"]
        category = data["categories"][obj_idx]
        cats.append(category)

        cx, cy, cz = box3d[0], box3d[1], box3d[2]
        w_box, h_box, d_box = box3d[3], box3d[4], box3d[5]
        qw, qx, qy, qz = box3d[6], box3d[7], box3d[8], box3d[9]

        center_cam = [cx, cy, cz]
        rot = R.from_quat([qx, qy, qz, qw])
        R_cam = rot.as_matrix() @ P_FP_TO_VIS4D
        dimensions = [w_box, h_box, d_box]

        # Gravity correction
        R_cam, dimensions = align_rotation_to_gravity(
            R_cam, dimensions, gravity_cam
        )

        # Check upside-down after gravity alignment
        # R_cam[1,1] * h < 0 means bottom face is above top face
        is_upside_down = R_cam[1, 1] * dimensions[1] < 0

        is_behind = cz <= 0
        bbox3d_cam = compute_box3d_corners(center_cam, dimensions, R_cam)

        bbox2d_proj = [-1, -1, -1, -1]
        bbox2d_trunc = [-1, -1, -1, -1]
        truncation = -1.0

        if not is_behind:
            proj, trunc = project_box3d_to_bbox2d(bbox3d_cam, K)
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

        b2d = data["boxes2d"][obj_idx]

        ann = {
            "center_cam": center_cam,
            "dimensions": dimensions,
            "R_cam": R_cam.tolist(),
            "bbox3D_cam": bbox3d_cam,
            "bbox2D_proj": bbox2d_proj,
            "bbox2D_trunc": bbox2d_trunc,
            "bbox2D_tight": b2d,
            "behind_camera": bool(is_behind),
            "truncation": truncation,
            "visibility": -1,
            "lidar_pts": -1,
            "segmentation_pts": -1,
            "depth_error": -1,
            "valid3D": not is_behind and not is_upside_down,
            "category_name": category,
        }
        anns.append(ann)

    return img_entry, anns, cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bbox3d_dir",
        required=True,
        help="Dir with FoundationPose 3D bbox JSONs (one per object).",
    )
    parser.add_argument(
        "--extracted_dir",
        required=True,
        help="FoundationPose extracted dir with images/ and camera_params/.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/foundationpose/annotations",
        help="Output dir for FoundationPose_{train,val}.json (default: data/foundationpose/annotations).",
    )
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_id", type=int, default=11)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    global BBOX3D_DIR, EXTRACTED_DIR, OUTPUT_DIR
    BBOX3D_DIR = args.bbox3d_dir
    EXTRACTED_DIR = args.extracted_dir
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Listing bbox3d files...")
    bbox_files = sorted(
        [f for f in os.listdir(BBOX3D_DIR) if f.endswith(".json")]
    )
    print(f"Found {len(bbox_files)} bbox3d files")

    if args.max_images > 0:
        bbox_files = bbox_files[: args.max_images]
        print(f"Limited to {len(bbox_files)} files")

    # Process all files in parallel (threads for I/O-bound NFS reads)
    print(f"Processing with {args.workers} threads...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_file, bbox_files),
                total=len(bbox_files),
                desc="Converting",
            )
        )

    # Filter out None results
    results = [r for r in results if r is not None]
    print(f"Successfully processed {len(results)} images")

    # Train/val split
    random.seed(args.seed)
    indices = list(range(len(results)))
    random.shuffle(indices)
    n_val = int(len(indices) * args.val_ratio)
    val_set = set(indices[:n_val])

    cat_set = set()
    class_counter = Counter()
    train_images, train_annos = [], []
    val_images, val_annos = [], []
    train_img_id, val_img_id = 0, 0
    train_anno_id, val_anno_id = 0, 0

    for idx, (img_entry, anns, cats) in enumerate(results):
        is_val = idx in val_set

        if is_val:
            img_id = val_img_id
            anno_id = val_anno_id
        else:
            img_id = train_img_id
            anno_id = train_anno_id

        img_entry["id"] = img_id
        img_entry["dataset_id"] = args.dataset_id

        if is_val:
            val_images.append(img_entry)
        else:
            train_images.append(img_entry)

        for ann in anns:
            ann["id"] = anno_id
            ann["image_id"] = img_id
            ann["dataset_id"] = args.dataset_id
            cat_set.add(ann["category_name"])
            class_counter[ann["category_name"]] += 1

            if is_val:
                val_annos.append(ann)
            else:
                train_annos.append(ann)
            anno_id += 1

        if is_val:
            val_anno_id = anno_id
            val_img_id += 1
        else:
            train_anno_id = anno_id
            train_img_id += 1

    # Build category list
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(sorted(cat_set))
    ]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    for ann in train_annos + val_annos:
        ann["category_id"] = cat_name_to_id.get(
            ann["category_name"], -1
        )

    # Save
    for split_name, images_list, annos_list in [
        ("train", train_images, train_annos),
        ("val", val_images, val_annos),
    ]:
        output = {
            "images": images_list,
            "annotations": annos_list,
            "categories": categories,
        }
        out_path = os.path.join(
            OUTPUT_DIR, f"FoundationPose_{split_name}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f)
        print(
            f"Saved {split_name}: {len(images_list)} images, "
            f"{len(annos_list)} annotations -> {out_path}"
        )

    print(f"\nTotal categories: {len(categories)}")
    print(f"Top 20: {class_counter.most_common(20)}")


if __name__ == "__main__":
    main()
