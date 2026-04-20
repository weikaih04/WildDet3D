"""Safe 3D IoU computation for training.

Uses shapely for BEV polygon intersection (CPU, never crashes).
Works with vis4d decoded box format: (center_3d(3), dims(3), quat(4)).
Supports full rotation (not yaw-only).

Usage:
    from opendet3d.op.box.iou_3d_safe import batch_box3d_iou
    iou = batch_box3d_iou(pred_decoded, gt_decoded)  # (M,)
"""

import numpy as np
from shapely.geometry import Polygon


def _quat_to_rotmat(quat):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix (numpy)."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _box_to_corners(center, dims, quat):
    """Convert a single decoded box to 8 corners.

    Args:
        center: (3,) center_3d in camera frame
        dims: (3,) dimensions (w, h, l) in vis4d/Omni3D convention
        quat: (4,) quaternion (w, x, y, z)

    Returns:
        corners: (8, 3) numpy array
    """
    w, h, l = dims
    R = _quat_to_rotmat(quat)

    # 8 unit corners scaled by half-dims
    dx, dy, dz = w / 2, h / 2, l / 2
    local_corners = np.array([
        [ dx,  dy,  dz],
        [ dx,  dy, -dz],
        [ dx, -dy,  dz],
        [ dx, -dy, -dz],
        [-dx,  dy,  dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [-dx, -dy, -dz],
    ])

    # Rotate and translate
    corners = (R @ local_corners.T).T + center
    return corners


def _bev_intersection_area(corners1, corners2):
    """Compute BEV (XZ plane) intersection area using shapely.

    OPENCV frame: Y is down, Z is forward.
    BEV = top-down view = XZ plane.
    """
    # Project all 8 corners to XZ, take convex hull
    xz1 = corners1[:, [0, 2]]  # (8, 2) -> (x, z)
    xz2 = corners2[:, [0, 2]]

    poly1 = Polygon(xz1).convex_hull
    poly2 = Polygon(xz2).convex_hull

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    if not poly1.intersects(poly2):
        return 0.0
    return poly1.intersection(poly2).area


def _height_overlap(corners1, corners2):
    """Compute height (Y-axis) overlap between two boxes."""
    y1_min, y1_max = corners1[:, 1].min(), corners1[:, 1].max()
    y2_min, y2_max = corners2[:, 1].min(), corners2[:, 1].max()

    overlap_min = max(y1_min, y2_min)
    overlap_max = min(y1_max, y2_max)
    return max(0.0, overlap_max - overlap_min)


def _box_volume(corners):
    """Compute box volume from corners using convex hull area * height."""
    xz = corners[:, [0, 2]]
    poly = Polygon(xz).convex_hull
    if not poly.is_valid:
        return 0.0
    area = poly.area
    height = corners[:, 1].max() - corners[:, 1].min()
    return area * height


def box3d_iou_single(box1, box2):
    """Compute 3D IoU between two decoded boxes.

    Args:
        box1, box2: (10,) numpy array = (center_3d(3), dims(3), quat(4))
            vis4d decoded format.

    Returns:
        iou: float in [0, 1]
    """
    center1, dims1, quat1 = box1[:3], box1[3:6], box1[6:10]
    center2, dims2, quat2 = box2[:3], box2[3:6], box2[6:10]

    # Skip degenerate boxes (zero/negative dims)
    if np.any(dims1 <= 0) or np.any(dims2 <= 0):
        return 0.0

    corners1 = _box_to_corners(center1, dims1, quat1)
    corners2 = _box_to_corners(center2, dims2, quat2)

    # Check for NaN/Inf
    if not np.all(np.isfinite(corners1)) or not np.all(np.isfinite(corners2)):
        return 0.0

    bev_area = _bev_intersection_area(corners1, corners2)
    h_overlap = _height_overlap(corners1, corners2)
    inter_vol = bev_area * h_overlap

    vol1 = _box_volume(corners1)
    vol2 = _box_volume(corners2)
    union = vol1 + vol2 - inter_vol

    if union <= 0:
        return 0.0

    return float(np.clip(inter_vol / union, 0.0, 1.0))


def batch_box3d_iou(pred_boxes, gt_boxes):
    """Compute pairwise diagonal 3D IoU for matched box pairs.

    Args:
        pred_boxes: torch.Tensor (M, 10) decoded pred boxes
        gt_boxes: torch.Tensor (M, 10) decoded GT boxes
            Both in vis4d format: (center_3d(3), dims(3), quat(4))

    Returns:
        ious: torch.Tensor (M,) IoU values in [0, 1]
    """
    import torch
    M = pred_boxes.shape[0]
    ious = torch.zeros(M, device=pred_boxes.device)

    pred_np = pred_boxes.detach().cpu().float().numpy()
    gt_np = gt_boxes.detach().cpu().float().numpy()

    for i in range(M):
        if not np.all(np.isfinite(pred_np[i])) or not np.all(np.isfinite(gt_np[i])):
            continue
        ious[i] = box3d_iou_single(pred_np[i], gt_np[i])

    return ious
