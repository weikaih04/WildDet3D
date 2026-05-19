"""DROID triple-view renderer: wrist + ext1 + ext2 with 3D box overlay.

Reads a vis4d-saved `detect_3D_results.json` (the per-image 3D predictions
that `Detect3DEvaluator` dumps), looks up the matching DROID episode in
the val annotation JSON, and produces a side-by-side comparison panel of
all 3 cameras with the predicted 3D boxes projected onto each.

Usage:
    python demo/embodied/render_triple.py \\
        --episode shard00020_ep004 \\
        --ann data/droid/annotations/DROID_val_unified.json \\
        --pred <eval_output>/eval/droid_dist/3D/detect_3D_results.json \\
        --data_root data \\
        --out_dir demo/embodied/output \\
        --score_thresh 0.2
"""
import argparse
import json
import os
import re

import cv2
import numpy as np


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Saturated colors that read well on RGB.
PALETTE = [
    (0, 200, 255),    # orange-yellow
    (0, 255, 100),    # green
    (255, 100, 100),  # blue (BGR)
    (255, 0, 200),    # purple
    (100, 200, 255),  # peach
    (255, 255, 0),    # cyan
]


def project(corners_cam, K):
    K = np.asarray(K, dtype=np.float64)
    pts = np.asarray(corners_cam, dtype=np.float64)  # (8, 3)
    proj = K @ pts.T
    proj = proj[:2] / np.clip(proj[2:3], 1e-6, None)
    return proj.T


def draw_box(img, corners_2d, color, thickness=2):
    pts = corners_2d.astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]),
                 color, thickness, cv2.LINE_AA)


def draw_label(img, corners_2d, name, color):
    pts = corners_2d.astype(int)
    cx = int(np.mean(pts[:, 0]))
    top_y = max(int(np.min(pts[:, 1])) - 6, 14)
    h, w = img.shape[:2]
    cx = max(2, min(cx - 20, w - 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.55, 1
    (tw, th), _ = cv2.getTextSize(name, font, fs, ft)
    x1, y1 = cx, top_y - th - 4
    x2, y2 = cx + tw + 6, top_y + 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img, name, (cx + 3, top_y - 2),
                font, fs, (0, 0, 0), ft, cv2.LINE_AA)


def view_of(fp):
    if "wrist" in fp:
        return "wrist"
    if "ext1" in fp:
        return "ext1"
    if "ext2" in fp:
        return "ext2"
    return None


def episode_id(fp):
    m = re.search(r"(shard\d+_ep\d+)", fp)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True,
                    help="Episode ID, e.g. shard00020_ep004.")
    ap.add_argument(
        "--ann",
        default="data/droid/annotations/DROID_val_unified.json",
        help="DROID val annotation JSON.",
    )
    ap.add_argument(
        "--pred", required=True,
        help="Path to detect_3D_results.json produced by vis4d eval "
        "(under <output_dir>/eval/droid_*/3D/).",
    )
    ap.add_argument(
        "--data_root", default="data",
        help="Root that the annotation's image file_paths are relative "
        "to (default: data).",
    )
    ap.add_argument(
        "--out_dir", default="demo/embodied/output",
        help="Where to write the rendered panel.",
    )
    ap.add_argument("--score_thresh", type=float, default=0.0)
    ap.add_argument("--max_boxes", type=int, default=10)
    ap.add_argument("--show_gt", action="store_true",
                    help="Also draw GT boxes in white for comparison.")
    ap.add_argument("--layout", choices=["horizontal", "vertical"],
                    default="horizontal")
    ap.add_argument("--out", default=None,
                    help="Override output path (else <out_dir>/<eid>_triple.png).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ann) as f:
        ann = json.load(f)
    cats = {c["id"]: c["name"] for c in ann["categories"]}

    ep_imgs = {}
    for im in ann["images"]:
        if episode_id(im["file_path"]) == args.episode:
            v = view_of(im["file_path"])
            if v:
                ep_imgs[v] = im
    missing = {"wrist", "ext1", "ext2"} - set(ep_imgs.keys())
    if missing:
        raise SystemExit(f"Episode missing views: {missing}")

    with open(args.pred) as f:
        preds = json.load(f)
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)

    panels = []
    for view in ("wrist", "ext1", "ext2"):
        im = ep_imgs[view]
        img_path = os.path.join(args.data_root, im["file_path"])
        rgb = cv2.imread(img_path)
        if rgb is None:
            raise SystemExit(f"Could not read {img_path}")
        K = im["K"]
        boxes = by_img.get(im["id"], [])
        boxes = [b for b in boxes if b["score"] >= args.score_thresh]
        boxes.sort(key=lambda b: -b["score"])
        boxes = boxes[: args.max_boxes]

        if args.show_gt:
            gt_anns = [
                a for a in ann["annotations"]
                if a["image_id"] == im["id"]
                and a.get("valid3D")
                and not a.get("behind_camera", False)
            ]
            for g in gt_anns:
                draw_box(rgb, project(g["bbox3D_cam"], K),
                         (255, 255, 255), thickness=2)

        for i, b in enumerate(boxes):
            corners_2d = project(b["bbox3D"], K)
            color = PALETTE[i % len(PALETTE)]
            draw_box(rgb, corners_2d, color, thickness=2)
            draw_label(
                rgb, corners_2d, cats.get(b["category_id"], "?"), color
            )

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, ft = 0.85, 2
        (tw, th), _ = cv2.getTextSize(view, font, fs, ft)
        cv2.rectangle(rgb, (8, 8), (8 + tw + 12, 8 + th + 12),
                      (0, 0, 0), -1)
        cv2.putText(rgb, view, (14, 8 + th + 4),
                    font, fs, (255, 255, 255), ft, cv2.LINE_AA)
        panels.append(rgb)

    if args.layout == "horizontal":
        target_h = min(p.shape[0] for p in panels)
        resized = [
            cv2.resize(
                p, (int(p.shape[1] * target_h / p.shape[0]), target_h)
            )
            for p in panels
        ]
        sep = np.full((target_h, 4, 3), 255, dtype=np.uint8)
        parts = []
        for i, p in enumerate(resized):
            parts.append(p)
            if i < len(resized) - 1:
                parts.append(sep)
        out = np.hstack(parts)
    else:
        target_w = min(p.shape[1] for p in panels)
        resized = [
            cv2.resize(
                p, (target_w, int(p.shape[0] * target_w / p.shape[1]))
            )
            for p in panels
        ]
        sep = np.full((4, target_w, 3), 255, dtype=np.uint8)
        parts = []
        for i, p in enumerate(resized):
            parts.append(p)
            if i < len(resized) - 1:
                parts.append(sep)
        out = np.vstack(parts)

    out_path = args.out or os.path.join(
        args.out_dir, f"{args.episode}_triple.png"
    )
    cv2.imwrite(out_path, out)
    print(f"Saved {out_path}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
