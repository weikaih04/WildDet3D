"""Convert CA-1M (Cubify Anything) dataset to Omni3D JSON format.

Reads CA-1M WebDataset tar archives and converts to the Omni3D COCO-style
annotation format used by 3D-MOOD.

CA-1M data is already in OpenCV camera coordinates (x-right, y-down, z-forward),
so no coordinate conversion is needed.

No filtering is applied here -- all annotations are kept exhaustively.
Filtering (truncation, visibility, depth, etc.) is handled by the COCO3D
dataloader's is_ignore() method.

Coordinate mapping:
    CA-1M scale = [l, h, w] where l=local_X, h=local_Y, w=local_Z
    Omni3D dimensions = [W, H, L] where vis4d local frame is X=L, Y=H, Z=W
    So: W=scale[2], H=scale[1], L=scale[0]
    R_cam: direct copy (already transforms from local box to camera)

bbox2D mapping:
    box_2d_proj  -> bbox2D_proj  (full 3D->2D projection, unclipped)
    clip(proj)   -> bbox2D_trunc (proj clipped to image bounds, amodal)
    box_2d_rend  -> bbox2D_tight (rendered/modal, accounts for occlusion)

Depth:
    GT depth (laser scanner, 512x384, uint16 mm) is saved as-is.
    In the dataset class, set depth_scale=1000 to decode.

Usage:
    # Test with 3 tars first
    python data_conversion/cubifyanything/convert_ca1m_to_omni3d.py \
        --ca1m_root /path/to/ca1m/data \
        --output_dir data/cubifyanything \
        --split val \
        --max_tars 3

    # Full conversion
    python data_conversion/cubifyanything/convert_ca1m_to_omni3d.py \
        --ca1m_root /path/to/ca1m/data \
        --output_dir data/cubifyanything \
        --split train \
        --num_workers 8
"""

import argparse
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Structural categories to filter out in the non-full version
STRUCTURAL_CATEGORIES = {
    "wall",
    "floor",
    "ceiling",
    "door_frame",
    "door frame",
}

DATASET_ID = 200


def compute_box3d_corners(center, dimensions, R):
    """Compute 8 corners of a 3D bounding box.

    Matches vis4d boxes3d_to_corners() corner ordering (OPENCV mode):
        dimensions = [W, H, L]
        vis4d local frame: X=L, Y=H, Z=W

        Corner ordering:
               (back)
        (6) +---------+. (7)
            | ` .     |  ` .
            | (4) +---+-----+ (5)
            |     |   |     |
        (2) +-----+---+. (3)|
            ` .   |     ` . |
            (0) ` +---------+ (1)
                     (front)

        X (length): L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2, -L/2
        Y (height): H/2, H/2,  H/2,  H/2,-H/2,-H/2, -H/2, -H/2
        Z (width): -W/2, W/2, -W/2,  W/2,-W/2, W/2, -W/2,  W/2
    """
    W, H, L = dimensions
    lx, ly, lz = L / 2.0, H / 2.0, W / 2.0

    corners_local = np.array(
        [
            [lx, ly, -lz],
            [lx, ly, lz],
            [-lx, ly, -lz],
            [-lx, ly, lz],
            [lx, -ly, -lz],
            [lx, -ly, lz],
            [-lx, -ly, -lz],
            [-lx, -ly, lz],
        ]
    )

    R_np = np.array(R)
    corners_cam = (R_np @ corners_local.T).T + np.array(center)
    return corners_cam.tolist()


def box_area(bbox):
    """Compute area of [x1, y1, x2, y2] box."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def clip_box_to_image(bbox, img_w, img_h):
    """Clip [x1, y1, x2, y2] box to image bounds.

    Returns clipped box or None if degenerate after clipping.
    """
    x1 = max(0.0, bbox[0])
    y1 = max(0.0, bbox[1])
    x2 = min(float(img_w), bbox[2])
    y2 = min(float(img_h), bbox[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def process_tar(args):
    """Process a single tar file and return images + annotations.

    No filtering is applied -- all annotations are kept.
    Filtering is handled by the COCO3D dataloader.
    """
    (
        tar_path,
        split,
        output_dir,
        frame_interval,
        max_frames_per_tar,
        img_id_start,
        ann_id_start,
    ) = args

    # Extract numeric video ID from tar filename
    # e.g., "ca1m-train-42444499.tar" -> "42444499"
    tar_stem = Path(tar_path).stem
    parts = tar_stem.split("-")
    video_id = parts[-1] if len(parts) > 1 else tar_stem

    images = []
    annotations = []
    categories_seen = set()
    img_id = img_id_start
    ann_id = ann_id_start

    # Output directories
    img_dir = os.path.join(
        output_dir, "data", "CubifyAnything", split, video_id
    )
    depth_gt_dir = os.path.join(
        output_dir, "depth_gt", split, video_id
    )

    with tarfile.open(tar_path, "r") as tf:
        # Pass 1: Parse member names and group by timestamp
        members_by_ts = defaultdict(dict)
        for member in tf.getmembers():
            if not member.isfile():
                continue

            # Tar path format: {video_num}/{timestamp}.{sensor}/{file}
            name_parts = member.name.split("/", 1)
            if len(name_parts) < 2:
                continue

            remainder = name_parts[1]
            dot_idx = remainder.find(".")
            if dot_idx < 0:
                continue

            timestamp = remainder[:dot_idx]
            rest = remainder[dot_idx + 1:]

            # Skip world entries and metadata (prefixed with _)
            if timestamp == "world" or rest.startswith("_"):
                continue

            members_by_ts[timestamp][rest] = member

        # Sort timestamps numerically and subsample
        sorted_ts = sorted(
            members_by_ts.keys(), key=lambda x: int(x)
        )
        if frame_interval > 1:
            sorted_ts = sorted_ts[::frame_interval]
        if max_frames_per_tar and max_frames_per_tar > 0:
            sorted_ts = sorted_ts[:max_frames_per_tar]

        # Pass 2: Process selected frames
        for ts in sorted_ts:
            mmap = members_by_ts[ts]

            # Check required files exist
            required_keys = [
                "wide/image.png",
                "wide/instances.json",
                "wide/image/K.json",
            ]
            if not all(k in mmap for k in required_keys):
                continue

            # Read camera intrinsics
            k_bytes = tf.extractfile(
                mmap["wide/image/K.json"]
            ).read()
            K = json.loads(k_bytes.decode("utf-8"))
            K = np.array(K).reshape(3, 3).tolist()

            # Read and save RGB image (PNG -> JPEG)
            image_bytes = tf.extractfile(
                mmap["wide/image.png"]
            ).read()
            img_pil = Image.open(io.BytesIO(image_bytes))
            img_w, img_h = img_pil.size

            os.makedirs(img_dir, exist_ok=True)
            img_filename = f"{ts}.jpg"
            img_pil.save(
                os.path.join(img_dir, img_filename),
                "JPEG",
                quality=95,
            )

            rel_img_path = os.path.join(
                "cubifyanything",
                "data",
                "CubifyAnything",
                split,
                video_id,
                img_filename,
            )

            # Save GT depth (laser scanner, uint16 mm) if available
            if "gt/depth.png" in mmap:
                os.makedirs(depth_gt_dir, exist_ok=True)
                depth_data = tf.extractfile(
                    mmap["gt/depth.png"]
                ).read()
                depth_gt_path = os.path.join(
                    depth_gt_dir, f"{ts}.png"
                )
                with open(depth_gt_path, "wb") as f:
                    f.write(depth_data)

            # Build image entry
            images.append(
                {
                    "width": img_w,
                    "height": img_h,
                    "file_path": rel_img_path,
                    "K": K,
                    "src_90_rotate": 0,
                    "src_flagged": False,
                    "incomplete": False,
                    "id": img_id,
                    "dataset_id": DATASET_ID,
                }
            )

            # Parse instances
            inst_bytes = tf.extractfile(
                mmap["wide/instances.json"]
            ).read()
            instances = json.loads(inst_bytes.decode("utf-8"))

            for inst in instances:
                cat = inst["category"]
                pos = inst["position"]
                scale = inst["scale"]
                R = inst["R"]

                categories_seen.add(cat)

                is_behind = pos[2] <= 0

                # CA-1M scale = [l, h, w] (local X, Y, Z)
                # Omni3D dimensions = [W, H, L] (vis4d: X=L, Y=H, Z=W)
                dims_omni3d = [scale[2], scale[1], scale[0]]

                # Compute 8 corners matching vis4d ordering
                bbox3d_cam = compute_box3d_corners(
                    pos, dims_omni3d, R
                )

                # bbox2D fields:
                #   box_2d_proj -> bbox2D_proj (amodal, unclipped)
                #   box_2d_rend -> bbox2D_trunc (amodal, clipped to image)
                #   CA-1M has no modal/tight box -> bbox2D_tight = [-1,-1,-1,-1]
                bbox2d_proj = inst.get(
                    "box_2d_proj", [-1, -1, -1, -1]
                )
                bbox2d_trunc = inst.get(
                    "box_2d_rend", [-1, -1, -1, -1]
                )

                truncation = -1.0
                visibility = -1.0

                if (
                    not is_behind
                    and bbox2d_trunc != [-1, -1, -1, -1]
                    and bbox2d_proj != [-1, -1, -1, -1]
                ):
                    # truncation = 1 - area(trunc) / area(proj)
                    area_proj = box_area(bbox2d_proj)
                    area_trunc = box_area(bbox2d_trunc)
                    if area_proj > 0:
                        truncation = max(
                            0.0, 1.0 - area_trunc / area_proj
                        )
                    else:
                        truncation = 0.0

                annotations.append(
                    {
                        "behind_camera": is_behind,
                        "truncation": truncation,
                        "bbox2D_tight": [-1, -1, -1, -1],
                        "visibility": visibility,
                        "segmentation_pts": -1,
                        "lidar_pts": -1,
                        "valid3D": True,
                        "category_name": cat,
                        "id": ann_id,
                        "image_id": img_id,
                        "dataset_id": DATASET_ID,
                        "bbox2D_proj": bbox2d_proj,
                        "depth_error": -1,
                        "bbox2D_trunc": bbox2d_trunc,
                        "center_cam": pos,
                        "dimensions": dims_omni3d,
                        "R_cam": R,
                        "bbox3D_cam": bbox3d_cam,
                    }
                )
                ann_id += 1

            img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "img_id_next": img_id,
        "ann_id_next": ann_id,
        "category_names": categories_seen,
    }


def verify_projections(images, annotations, num_samples=100):
    """Verify coordinate correctness by projecting center_cam to 2D."""
    print("\nVerification: Projecting center_cam to 2D...")

    img_lookup = {img["id"]: img for img in images}

    errors = []
    checked = 0
    for ann in annotations:
        if checked >= num_samples:
            break
        if ann["behind_camera"]:
            continue
        bbox2d = ann["bbox2D_trunc"]
        if bbox2d == [-1, -1, -1, -1]:
            continue

        img_entry = img_lookup.get(ann["image_id"])
        if img_entry is None:
            continue

        K = np.array(img_entry["K"])
        center = np.array(ann["center_cam"])

        if center[2] <= 0:
            continue

        projected = K @ center
        px = projected[0] / projected[2]
        py = projected[1] / projected[2]

        bbox_cx = (bbox2d[0] + bbox2d[2]) / 2.0
        bbox_cy = (bbox2d[1] + bbox2d[3]) / 2.0

        err = np.sqrt((px - bbox_cx) ** 2 + (py - bbox_cy) ** 2)
        errors.append(err)
        checked += 1

    if errors:
        errors = np.array(errors)
        print(
            "  Projection error (center_cam -> 2D vs bbox center):"
        )
        print(f"    Mean: {errors.mean():.1f} px")
        print(f"    Median: {np.median(errors):.1f} px")
        print(f"    Max: {errors.max():.1f} px")
        print(f"    Samples checked: {len(errors)}")
    else:
        print("  No valid samples for verification.")


def convert(
    ca1m_root,
    output_dir,
    split,
    frame_interval=10,
    max_tars=None,
    max_frames_per_tar=None,
    num_workers=1,
):
    """Main conversion: CA-1M tars -> Omni3D JSON."""
    split_dir = os.path.join(ca1m_root, split)

    tar_files = sorted(
        [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".tar")
        ]
    )

    if max_tars and max_tars > 0:
        tar_files = tar_files[:max_tars]

    print(f"Found {len(tar_files)} tar files in {split_dir}")
    print(f"Frame interval: every {frame_interval}th frame")
    if max_frames_per_tar:
        print(f"Max frames per tar: {max_frames_per_tar}")

    all_images = []
    all_annotations = []
    all_categories = set()

    if num_workers > 1:
        args_list = [
            (
                tar_path,
                split,
                output_dir,
                frame_interval,
                max_frames_per_tar,
                0,
                0,
            )
            for tar_path in tar_files
        ]

        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_tar, args_list),
                    total=len(tar_files),
                    desc=f"Processing {split} tars",
                )
            )

        # Reassign contiguous IDs
        img_id = 0
        ann_id = 0
        for result in results:
            old_to_new_img = {}
            for img in result["images"]:
                old_id = img["id"]
                img["id"] = img_id
                old_to_new_img[old_id] = img_id
                img_id += 1
            for ann in result["annotations"]:
                ann["id"] = ann_id
                ann["image_id"] = old_to_new_img[ann["image_id"]]
                ann_id += 1
            all_images.extend(result["images"])
            all_annotations.extend(result["annotations"])
            all_categories.update(result["category_names"])
    else:
        img_id = 0
        ann_id = 0
        for tar_path in tqdm(
            tar_files, desc=f"Processing {split} tars"
        ):
            result = process_tar(
                (
                    tar_path,
                    split,
                    output_dir,
                    frame_interval,
                    max_frames_per_tar,
                    img_id,
                    ann_id,
                )
            )
            all_images.extend(result["images"])
            all_annotations.extend(result["annotations"])
            all_categories.update(result["category_names"])
            img_id = result["img_id_next"]
            ann_id = result["ann_id_next"]

    # Build category list (sorted for determinism)
    sorted_categories = sorted(all_categories)
    cat_name_to_id = {
        name: i for i, name in enumerate(sorted_categories)
    }
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(sorted_categories)
    ]

    # Assign category_ids
    for ann in all_annotations:
        ann["category_id"] = cat_name_to_id[ann["category_name"]]

    # Print summary
    print(f"\nConversion summary for {split}:")
    print(f"  Images: {len(all_images)}")
    print(f"  Annotations: {len(all_annotations)}")
    print(f"  Categories: {len(categories)}")
    cat_counts = Counter(
        a["category_name"] for a in all_annotations
    )
    print(f"  Top 30 categories:")
    for name, count in cat_counts.most_common(30):
        print(f"    {name}: {count}")

    structural_count = sum(
        1
        for a in all_annotations
        if a["category_name"] in STRUCTURAL_CATEGORIES
    )
    print(
        f"  Structural labels (to be filtered): {structural_count}"
    )

    # Count truncation/visibility stats
    has_trunc = sum(
        1 for a in all_annotations if a["truncation"] >= 0
    )
    has_vis = sum(
        1 for a in all_annotations if a["visibility"] >= 0
    )
    print(f"  Annotations with truncation: {has_trunc}")
    print(f"  Annotations with visibility: {has_vis}")

    info = {
        "id": DATASET_ID,
        "source": "CubifyAnything",
        "name": f"CubifyAnything {split.capitalize()}",
        "split": split.capitalize(),
        "version": "1.0",
        "url": "https://apple.github.io/ml-cubifyanything/",
    }

    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # Version 1: Filtered (structural labels removed)
    filtered_anns = []
    filtered_cat_names = set()
    for ann in all_annotations:
        if ann["category_name"] not in STRUCTURAL_CATEGORIES:
            filtered_anns.append(dict(ann))
            filtered_cat_names.add(ann["category_name"])

    filtered_cat_names_sorted = sorted(filtered_cat_names)
    filtered_cat_name_to_id = {
        name: i
        for i, name in enumerate(filtered_cat_names_sorted)
    }
    filtered_categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(filtered_cat_names_sorted)
    ]
    for ann in filtered_anns:
        ann["category_id"] = filtered_cat_name_to_id[
            ann["category_name"]
        ]

    split_name = split
    filtered_output = {
        "info": info,
        "categories": filtered_categories,
        "images": all_images,
        "annotations": filtered_anns,
    }
    filtered_path = os.path.join(
        ann_dir, f"CubifyAnything_{split_name}.json"
    )
    print(f"\nSaving filtered annotations to {filtered_path}")
    print(
        f"  ({len(filtered_anns)} annotations, "
        f"{len(filtered_categories)} categories)"
    )
    with open(filtered_path, "w") as f:
        json.dump(filtered_output, f)

    # Version 2: Full (all labels)
    full_output = {
        "info": info,
        "categories": categories,
        "images": all_images,
        "annotations": all_annotations,
    }
    full_path = os.path.join(
        ann_dir, f"CubifyAnything_full_{split_name}.json"
    )
    print(f"Saving full annotations to {full_path}")
    print(
        f"  ({len(all_annotations)} annotations, "
        f"{len(categories)} categories)"
    )
    with open(full_path, "w") as f:
        json.dump(full_output, f)

    # Verification
    verify_projections(all_images, all_annotations)

    print("\nDone!")
    return filtered_path, full_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert CA-1M (Cubify Anything) to Omni3D JSON"
    )
    parser.add_argument(
        "--ca1m_root",
        type=str,
        required=True,
        help="Path to CA-1M data directory "
        "(containing train/ and val/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cubifyanything",
        help="Output directory for annotations and data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to convert",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=10,
        help="Process every Nth frame (default: 10)",
    )
    parser.add_argument(
        "--max_tars",
        type=int,
        default=None,
        help="Limit number of tars (for testing)",
    )
    parser.add_argument(
        "--max_frames_per_tar",
        type=int,
        default=None,
        help="Limit frames per tar (for testing)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for tar processing",
    )
    args = parser.parse_args()

    convert(
        ca1m_root=args.ca1m_root,
        output_dir=args.output_dir,
        split=args.split,
        frame_interval=args.frame_interval,
        max_tars=args.max_tars,
        max_frames_per_tar=args.max_frames_per_tar,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
