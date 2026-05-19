"""Convert DROID single-frame 3D box pipeline output to Omni3D JSON.

DROID pipeline output (per episode):
  - step3b1 box JSON: 3D OBB in OpenCV wrist-camera frame at best_frame
  - step2a depth h5: per-frame FoundationStereo depth (uint16 mm) for wrist cam
  - VLM result JSON: object_name + suitable_for_tracking flag
  - Episode dir: <wrist_serial>-stereo.mp4 (left half = rectified left view)

Per episode we take exactly one (image, depth, box) sample at best_frame.

Box-local axes mapping (DROID -> vis4d):
  DROID box-local: (w, h, l) order in `size`, R columns = (w_dir, h_dir, l_dir)
  vis4d canonical: dimensions=[W,H,L], local X=L, Y=H, Z=W
  Same situation as FoundationPose. Apply Ry(90):
    P = [[0,0,1],[0,1,0],[-1,0,0]]   (det=+1)
    R_cam_pre = R_droid @ P
    dimensions = [w, h, l]            # already vis4d [W, H, L]
  Then gravity norm (canonical rotation).

Gravity in OpenCV wrist camera frame:
  Franka world is Z-up, gravity_world = [0, 0, -1].
  R_w2c = inv(wrist_T_c2w)[:3, :3]
  gravity_cam = R_w2c @ [0, 0, -1]

Filter rules:
  - vlm_result.suitable_for_tracking == "YES" (Michael's filter)
  - <eid>_box.json + <eid>.done both exist
  (extrinsics validity already enforced upstream by step3b1)

Usage:
    # Smoke test (10 episodes, single thread, no workers)
    python scripts/data_prep/droid/convert_droid_to_omni3d.py \\
        --box_dir       <pipeline_output>/step3b1_singleframe_box \\
        --depth_h5_dir  <pipeline_output>/step2a_depth \\
        --step2b_dir    <pipeline_output>/step2b_extrinsics \\
        --vlm_dir       <pipeline_output>/step1_vlm_results \\
        --droid_raw_root <droid_raw>/1.0.1 \\
        --out_dir       data/droid \\
        --max_episodes 10 --num_workers 0

Output (under --out_dir):
    annotations/DROID_train.json
    annotations/DROID_val.json
    data/DROID/<eid>.jpg
    depth/<eid>.png
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# Pipeline-output paths. Set from CLI in main(); referenced by the worker
# helpers via module globals so they survive multiprocessing fork.
BOX_DIR: str = ""
DEPTH_H5_DIR: str = ""
STEP2B_DIR: str = ""
VLM_DIR: str = ""
DROID_RAW_ROOT: str = ""

# DROID raw episodes typically come with one of these gs:// / NFS prefixes
# in the VLM result `file_path`. We strip these and rebase under
# `--droid_raw_root` to find the actual episode dir on local disk.
DROID_RAW_PREFIXES = (
    "gs://xembodiment_data/r2d2/r2d2-data-full/",
    "/nfs/kun2/datasets/r2d2/r2d2-data-full/",
)

# Output paths, also set from CLI in main().
OUT_ROOT: str = ""
OUT_ANN_DIR: str = ""
OUT_IMG_DIR: str = ""
OUT_DEPTH_DIR: str = ""

DEPTH_SCALE = 1000.0  # uint16: depth_m * 1000 (already in mm in source h5)
JPEG_QUALITY = 92

# Column rearrangement: DROID box-local (w, h, l) -> vis4d canonical (L, H, W).
# Equivalent to FoundationPose's P_FP_TO_VIS4D. det = +1 preserves handedness.
P_DROID_TO_VIS4D = np.array(
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64
)

SHARD_RE = re.compile(r"^shard(\d+)_ep\d+$")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_box3d_corners(center, dimensions, R_cam):
    """8 corners in vis4d OPENCV convention.

    dimensions = [W, H, L]; local X=L, Y=H, Z=W.
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
    corners_cam = (R_cam @ corners_local.T).T + np.array(center)
    return corners_cam.tolist()


def project_box3d_to_bbox2d(corners_3d, K, img_w, img_h):
    """Project 8 3D corners to (proj, trunc) bbox2d. Returns (None, None) if
    any corner is behind camera."""
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
    tx2 = min(float(img_w), px2)
    ty2 = min(float(img_h), py2)
    if tx2 <= tx1 or ty2 <= ty1:
        return bbox2d_proj, None
    return bbox2d_proj, [tx1, ty1, tx2, ty2]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def gravity_in_wrist_cam(wrist_T_c2w):
    """Gravity vector in OpenCV wrist-camera frame.

    Franka base frame is Z-up, gravity_world = [0, 0, -1].
    """
    T_c2w = np.array(wrist_T_c2w, dtype=np.float64)
    T_w2c = np.linalg.inv(T_c2w)
    return T_w2c[:3, :3] @ np.array([0.0, 0.0, -1.0])


def align_rotation_to_gravity(R_obj, dims_whl, gravity_cam):
    """Canonical-rotation upright fix.

    Same logic as FoundationPose / InTheWild converter:
      1. Axis swap: bring local axis closest to gravity into Y
      2. Y flip: ensure local Y points along gravity
      3. W <= L: enforce width <= length via Ry(90) swap
    """
    g = gravity_cam / np.linalg.norm(gravity_cam)
    dot_x = np.dot(R_obj[:, 0], g)
    dot_y = np.dot(R_obj[:, 1], g)
    dot_z = np.dot(R_obj[:, 2], g)
    abs_dots = [abs(dot_x), abs(dot_y), abs(dot_z)]
    best_axis = int(np.argmax(abs_dots))

    R_out = R_obj.copy()
    w, h, l = dims_whl[0], dims_whl[1], dims_whl[2]

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


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def resolve_episode_dir(file_path: str) -> Path:
    fp = file_path
    for prefix in DROID_RAW_PREFIXES:
        if fp.startswith(prefix):
            fp = fp[len(prefix):]
            break
    return Path(DROID_RAW_ROOT) / Path(fp).parent


def extract_left_frame(mp4_path: str, frame_idx: int):
    """Read frame `frame_idx` from stereo MP4 and return the LEFT half (BGR)."""
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    if w <= 2 * h:
        # Mono MP4 fallback (sanity guard; stereo expected per README L115).
        return frame
    half = w // 2
    return frame[:, :half]


def read_wrist_depth(eid: str, frame_idx: int):
    """Load wrist depth at frame_idx from step2a h5 (uint16 mm)."""
    return read_view_depth(eid, "wrist", frame_idx)


def read_view_depth(eid: str, view: str, frame_idx: int):
    """Load depth at frame_idx from step2a h5 for a given view (wrist/ext1/ext2).

    Returns (depth_array, intrinsic_3x3) or (None, None).
    """
    h5_path = os.path.join(DEPTH_H5_DIR, f"{eid}_depth.h5")
    if not os.path.exists(h5_path):
        return None, None
    with h5py.File(h5_path, "r") as f:
        if view not in f or "depth" not in f[view] or "intrinsic" not in f[view]:
            return None, None
        depth = np.array(f[f"{view}/depth"][frame_idx])
        K = np.array(f[f"{view}/intrinsic"])
    return depth, K


def shard_id_of(eid: str) -> int | None:
    m = SHARD_RE.match(eid)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-episode worker
# ---------------------------------------------------------------------------

_worker_no_image = False
_worker_no_depth = False
_worker_views: tuple[str, ...] = ("wrist",)


def _worker_init(no_image, no_depth, views):
    global _worker_no_image, _worker_no_depth, _worker_views
    _worker_no_image = no_image
    _worker_no_depth = no_depth
    _worker_views = views


def _build_view_record(
    view: str,
    eid: str,
    object_name: str,
    is_val: bool,
    mp4_path: Path,
    best_frame: int,
    K: np.ndarray,
    depth: np.ndarray,
    R_cam_pre_view: np.ndarray,
    dim_pre: list[float],
    center_view: np.ndarray,
    gravity_view: np.ndarray,
):
    """Write JPEG + depth PNG for one view, return (img, ann, is_val)."""
    img_h, img_w = depth.shape

    out_img_rel = os.path.join(
        "droid", "data", "DROID", f"{eid}_{view}.jpg"
    )
    out_img_abs = os.path.join(BASE_DIR, "data", out_img_rel)
    if not _worker_no_image and not os.path.exists(out_img_abs):
        os.makedirs(os.path.dirname(out_img_abs), exist_ok=True)
        frame = extract_left_frame(str(mp4_path), best_frame)
        if frame is None:
            return None
        if frame.shape[0] != img_h or frame.shape[1] != img_w:
            frame = cv2.resize(frame, (img_w, img_h))
        cv2.imwrite(
            out_img_abs, frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

    out_depth_rel = os.path.join("droid", "depth", f"{eid}_{view}.png")
    out_depth_abs = os.path.join(BASE_DIR, "data", out_depth_rel)
    if not _worker_no_depth and not os.path.exists(out_depth_abs):
        os.makedirs(os.path.dirname(out_depth_abs), exist_ok=True)
        cv2.imwrite(out_depth_abs, depth.astype(np.uint16))

    # Gravity normalization in this view's frame
    R_cam, dimensions = align_rotation_to_gravity(
        R_cam_pre_view, list(dim_pre), gravity_view
    )

    cz = float(center_view[2])
    is_behind = cz <= 0
    center_list = center_view.tolist()
    bbox3d_cam = compute_box3d_corners(center_list, dimensions, R_cam)

    bbox2d_proj = [-1.0, -1.0, -1.0, -1.0]
    bbox2d_trunc = [-1.0, -1.0, -1.0, -1.0]
    truncation = -1.0
    if not is_behind:
        proj, trunc = project_box3d_to_bbox2d(
            bbox3d_cam, K.tolist(), img_w, img_h
        )
        if proj is not None:
            bbox2d_proj = proj
        if trunc is not None:
            bbox2d_trunc = trunc
            ap = box_area(bbox2d_proj)
            if ap > 0:
                truncation = max(0.0, 1.0 - box_area(bbox2d_trunc) / ap)
            else:
                truncation = 0.0

    img_entry = {
        "file_path": out_img_rel,
        "height": int(img_h),
        "width": int(img_w),
        "K": K.tolist(),
        "src_90_rotate": 0,
        "src_flagged": False,
    }
    ann = {
        "center_cam": center_list,
        "dimensions": dimensions,
        "R_cam": R_cam.tolist(),
        "bbox3D_cam": bbox3d_cam,
        "bbox2D_proj": bbox2d_proj,
        "bbox2D_trunc": bbox2d_trunc,
        "bbox2D_tight": [-1, -1, -1, -1],
        "behind_camera": bool(is_behind),
        "truncation": float(truncation),
        "visibility": -1,
        "lidar_pts": -1,
        "segmentation_pts": -1,
        "depth_error": -1,
        "valid3D": not is_behind,
        "category_name": object_name,
    }
    return (img_entry, ann, is_val)


def process_one(eid: str):
    """Returns a list of (img_entry, ann, is_val), one per view that succeeded.
    Empty list = filtered or no view succeeded."""
    box_json = os.path.join(BOX_DIR, f"{eid}_box.json")
    done_marker = os.path.join(BOX_DIR, f"{eid}.done")
    if not (os.path.exists(box_json) and os.path.exists(done_marker)):
        return []

    vlm_json = os.path.join(VLM_DIR, f"vlm_result_{eid}.json")
    if not os.path.exists(vlm_json):
        return []
    with open(vlm_json) as f:
        vlm = json.load(f)
    if vlm.get("suitable_for_tracking", "").upper() != "YES":
        return []

    with open(box_json) as f:
        box = json.load(f)

    object_name = (
        box.get("object_name") or vlm.get("object_name") or ""
    ).strip().lower()
    if not object_name:
        return []

    best_frame = int(box["best_frame"])
    wrist_serial = box["wrist_serial"]

    # Resolve episode dir + metadata for ext serials
    ep_dir = resolve_episode_dir(vlm["file_path"])
    metadata_files = list(ep_dir.glob("metadata_*.json"))
    if not metadata_files:
        return []
    with open(metadata_files[0]) as f:
        meta = json.load(f)

    # Wrist-frame box (raw DROID -> column rearrangement)
    box_cam = box["box_camera"]
    qw, qx, qy, qz = box_cam["quat_wxyz"]
    R_droid = R.from_quat([qx, qy, qz, qw]).as_matrix()
    R_cam_pre_wrist = R_droid @ P_DROID_TO_VIS4D
    dim_pre = [
        float(box_cam["size"][0]),
        float(box_cam["size"][1]),
        float(box_cam["size"][2]),
    ]
    center_wrist = np.array(box_cam["center"], dtype=np.float64)

    # Wrist c2w for transforming to world (needed for ext views)
    T_c2w_wrist = np.array(box["wrist_T_c2w"], dtype=np.float64)
    R_c2w = T_c2w_wrist[:3, :3]
    t_c2w = T_c2w_wrist[:3, 3]
    center_world = R_c2w @ center_wrist + t_c2w
    R_w_box = R_c2w @ R_cam_pre_wrist  # box rotation in world frame

    # Train/val split (per episode, shared by all views)
    sid = shard_id_of(eid)
    is_val = (sid is not None) and (sid % 10 == 0)

    # Step2b extrinsics (only needed if ext views requested)
    ext_cams_data = None
    if any(v in _worker_views for v in ("ext1", "ext2")):
        cams_path = os.path.join(STEP2B_DIR, f"{eid}_cameras.json")
        if os.path.exists(cams_path):
            with open(cams_path) as f:
                ext_cams_data = json.load(f)

    out = []
    for view in _worker_views:
        # Build per-view: K, depth, mp4, R_cam_pre, center, gravity
        if view == "wrist":
            mp4 = ep_dir / "recordings" / "MP4" / f"{wrist_serial}-stereo.mp4"
            if not mp4.exists():
                continue
            depth, _ = read_view_depth(eid, "wrist", best_frame)
            if depth is None:
                continue
            K_view = np.array(box["intrinsic_K"], dtype=np.float64)
            R_cam_pre_view = R_cam_pre_wrist
            center_view = center_wrist
            T_w2c = np.linalg.inv(T_c2w_wrist)
            gravity_view = T_w2c[:3, :3] @ np.array([0.0, 0.0, -1.0])
        else:
            # ext1 / ext2
            serial_key = f"{view}_cam_serial"
            if serial_key not in meta:
                continue
            serial = meta[serial_key]
            if ext_cams_data is None or serial not in ext_cams_data:
                continue
            mp4 = ep_dir / "recordings" / "MP4" / f"{serial}-stereo.mp4"
            if not mp4.exists():
                continue
            depth, K_h5 = read_view_depth(eid, view, best_frame)
            if depth is None:
                continue
            T_w2c = np.array(
                ext_cams_data[serial]["optimized_extrinsics"],
                dtype=np.float64,
            )
            # Prefer measured intrinsics from step2b (matches optimization);
            # fall back to h5 intrinsic if absent.
            K_view = np.array(
                ext_cams_data[serial].get(
                    "measured_intrinsics", K_h5.tolist()
                ),
                dtype=np.float64,
            )
            R_w2c = T_w2c[:3, :3]
            t_w2c = T_w2c[:3, 3]
            R_cam_pre_view = R_w2c @ R_w_box
            center_view = R_w2c @ center_world + t_w2c
            gravity_view = R_w2c @ np.array([0.0, 0.0, -1.0])

        rec = _build_view_record(
            view=view,
            eid=eid,
            object_name=object_name,
            is_val=is_val,
            mp4_path=mp4,
            best_frame=best_frame,
            K=K_view,
            depth=depth,
            R_cam_pre_view=R_cam_pre_view,
            dim_pre=dim_pre,
            center_view=center_view,
            gravity_view=gravity_view,
        )
        if rec is not None:
            out.append(rec)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the WildDet3D / Boxer3D-style DROID pipeline outputs "
            "(step1 VLM + step2a depth + step2b extrinsics + step3b1 "
            "single-frame box) into an Omni3D-format annotation JSON."
        )
    )
    parser.add_argument(
        "--box_dir", required=True,
        help="Pipeline output: <eid>_box.json + <eid>.done (step3b1).",
    )
    parser.add_argument(
        "--depth_h5_dir", required=True,
        help="Pipeline output: per-episode <eid>.h5 with depth_mm "
        "uint16 frames (step2a).",
    )
    parser.add_argument(
        "--step2b_dir", required=True,
        help="Pipeline output: ext1/ext2 extrinsics + camera params "
        "(step2b).",
    )
    parser.add_argument(
        "--vlm_dir", required=True,
        help="Pipeline output: vlm_result_<eid>.json (step1).",
    )
    parser.add_argument(
        "--droid_raw_root", required=True,
        help="Local DROID raw 1.0.1 root (contains episode dirs).",
    )
    parser.add_argument(
        "--out_dir", default="data/droid",
        help="Output root; writes annotations/, data/DROID/, depth/ "
        "under this directory (default: data/droid).",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=0,
        help="Cap number of episodes (0 = all)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=32,
        help="Parallel processes; 0 = single-threaded",
    )
    parser.add_argument(
        "--no_image", action="store_true",
        help="Skip RGB JPEG writing (use existing files)",
    )
    parser.add_argument(
        "--no_depth", action="store_true",
        help="Skip depth PNG writing",
    )
    parser.add_argument(
        "--views", type=str, default="wrist,ext1,ext2",
        help="Comma-separated views to extract (subset of wrist,ext1,ext2)",
    )
    args = parser.parse_args()

    # Resolve and bind the module-level paths that the worker helpers
    # close over. Done here (not at import time) so that running with
    # different roots is just a CLI swap.
    global BOX_DIR, DEPTH_H5_DIR, STEP2B_DIR, VLM_DIR, DROID_RAW_ROOT
    global OUT_ROOT, OUT_ANN_DIR, OUT_IMG_DIR, OUT_DEPTH_DIR
    BOX_DIR = os.path.abspath(args.box_dir)
    DEPTH_H5_DIR = os.path.abspath(args.depth_h5_dir)
    STEP2B_DIR = os.path.abspath(args.step2b_dir)
    VLM_DIR = os.path.abspath(args.vlm_dir)
    DROID_RAW_ROOT = os.path.abspath(args.droid_raw_root)
    OUT_ROOT = os.path.abspath(args.out_dir)
    OUT_ANN_DIR = os.path.join(OUT_ROOT, "annotations")
    OUT_IMG_DIR = os.path.join(OUT_ROOT, "data", "DROID")
    OUT_DEPTH_DIR = os.path.join(OUT_ROOT, "depth")

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    valid_views = {"wrist", "ext1", "ext2"}
    if not set(views).issubset(valid_views):
        raise SystemExit(f"--views must be subset of {valid_views}, got {views}")
    print(f"Views to extract: {views}")

    os.makedirs(OUT_ANN_DIR, exist_ok=True)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_DEPTH_DIR, exist_ok=True)

    print(f"Listing box JSONs under {BOX_DIR} ...")
    eids = sorted(
        f[:-len("_box.json")]
        for f in os.listdir(BOX_DIR)
        if f.endswith("_box.json")
    )
    print(f"  found {len(eids)} candidate episodes")
    if args.max_episodes > 0:
        eids = eids[: args.max_episodes]
        print(f"  capped to {len(eids)}")

    if args.num_workers <= 0:
        _worker_init(args.no_image, args.no_depth, views)
        results = [
            process_one(eid)
            for eid in tqdm(eids, desc="Converting")
        ]
    else:
        with mp.Pool(
            args.num_workers,
            initializer=_worker_init,
            initargs=(args.no_image, args.no_depth, views),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_one, eids, chunksize=8),
                total=len(eids), desc="Converting",
            ))

    train_images, train_annos = [], []
    val_images, val_annos = [], []
    cat_counter = Counter()
    train_img_id = val_img_id = 0
    train_ann_id = val_ann_id = 0
    n_filtered_episodes = 0
    view_counter = Counter()

    for r in results:
        if not r:
            n_filtered_episodes += 1
            continue
        for img, ann, is_val in r:
            # Track which view this came from via filename suffix
            stem = Path(img["file_path"]).stem
            view = stem.rsplit("_", 1)[-1] if "_" in stem else "unknown"
            view_counter[view] += 1
            if is_val:
                img["id"] = val_img_id
                val_images.append(img)
                ann["id"] = val_ann_id
                ann["image_id"] = val_img_id
                val_annos.append(ann)
                val_img_id += 1
                val_ann_id += 1
            else:
                img["id"] = train_img_id
                train_images.append(img)
                ann["id"] = train_ann_id
                ann["image_id"] = train_img_id
                train_annos.append(ann)
                train_img_id += 1
                train_ann_id += 1
            cat_counter[ann["category_name"]] += 1

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(sorted(cat_counter.keys()))
    ]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    for ann in train_annos + val_annos:
        ann["category_id"] = cat_name_to_id[ann["category_name"]]

    for split, imgs, anns in [
        ("train", train_images, train_annos),
        ("val", val_images, val_annos),
    ]:
        out = {
            "images": imgs,
            "annotations": anns,
            "categories": categories,
        }
        out_path = os.path.join(OUT_ANN_DIR, f"DROID_{split}.json")
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(
            f"  {split}: {len(imgs)} images, {len(anns)} anns -> {out_path}"
        )

    print(f"\nEpisodes filtered out (0 views): "
          f"{n_filtered_episodes} / {len(eids)}")
    print("Per-view sample counts:")
    for v in views:
        print(f"  {v}: {view_counter.get(v, 0)}")
    print(f"Categories: {len(categories)}")
    print("Top 20:")
    for k, v in cat_counter.most_common(20):
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
