"""Visualize DROID Omni3D JSON using vis4d's BoundingBox3DVisualizer.

Same pipeline as vis4d's eval — if boxes look correct here, they're correct
for training.

Usage:
    python scripts/data_prep/droid/visualize_droid_vis4d.py \\
        --json_path data/droid/annotations/DROID_train.json \\
        --num_samples 20
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from vis4d.data.const import AxisMode
from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
from vis4d.vis.image.canvas import PillowCanvasBackend


def ann_to_boxes3d(ann):
    """Omni3D annotation -> vis4d boxes3d [10] = [cx,cy,cz, w,l,h, qw,qx,qy,qz]."""
    center = ann["center_cam"]
    width, height, length = ann["dimensions"]
    rot = R.from_matrix(np.array(ann["R_cam"]))
    qx, qy, qz, qw = rot.as_quat()
    return [*center, width, length, height, qw, qx, qy, qz]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path", type=str,
        default="data/droid/annotations/DROID_train.json",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--output_dir", type=str,
        default="scripts/data_prep/droid/vis_output",
    )
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    print(f"Images: {len(data['images'])}, "
          f"Anns: {len(data['annotations'])}, "
          f"Cats: {len(data['categories'])}")

    id_to_img = {img["id"]: img for img in data["images"]}
    id_to_anns = {}
    for ann in data["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_ids = list(id_to_anns.keys())
    random.seed(args.seed)
    random.shuffle(img_ids)
    selected = img_ids[: args.num_samples]

    cat_mapping = {c["name"]: c["id"] for c in data["categories"]}

    visualizer = BoundingBox3DVisualizer(
        axis_mode=AxisMode.OPENCV,
        width=4,
        camera_near_clip=0.01,
        plot_heading=True,
        plot_trajectory=False,
        canvas=PillowCanvasBackend(font_size=12),
        cat_mapping=cat_mapping,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, img_id in enumerate(selected):
        img_info = id_to_img[img_id]
        anns = id_to_anns.get(img_id, [])
        img_path = os.path.join(args.data_root, img_info["file_path"])
        if not os.path.exists(img_path):
            print(f"  [skip] missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        K = np.array(img_info["K"], dtype=np.float32)

        boxes3d_list, class_ids_list, cats_list = [], [], []
        for ann in anns:
            if ann["behind_camera"]:
                continue
            boxes3d_list.append(ann_to_boxes3d(ann))
            class_ids_list.append(ann["category_id"])
            cats_list.append(ann["category_name"])

        if not boxes3d_list:
            continue

        boxes3d = np.array(boxes3d_list, dtype=np.float32)
        class_ids = np.array(class_ids_list, dtype=np.int64)

        visualizer.reset()
        visualizer.process_single_image(
            image=img_rgb,
            image_name=f"droid_{idx:04d}",
            boxes3d=boxes3d,
            intrinsics=K,
            extrinsics=None,
            class_ids=class_ids,
            categories=cats_list,
        )
        visualizer.save_to_disk(cur_iter=0, output_folder=args.output_dir)
        print(f"  [{idx+1}/{len(selected)}] {img_info['file_path']} "
              f"({len(boxes3d_list)} boxes)")

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
