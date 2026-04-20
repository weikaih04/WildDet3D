"""Preprocessing utilities for WildDet3D inference.

Handles image resizing, normalization, center padding, and intrinsics
adjustment to prepare raw inputs for the WildDet3D model.
"""

from typing import Optional

import numpy as np

from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.resize import (
    ResizeDepthMaps,
    ResizeImages,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor

from wilddet3d.data.transforms.pad import (
    CenterPadDepthMaps,
    CenterPadImages,
    CenterPadIntrinsics,
)
from wilddet3d.data.transforms.resize import GenResizeParameters

# WildDet3D expects 1008x1008 images
IMAGE_SIZE = (1008, 1008)


def preprocess(
    image: np.ndarray,
    intrinsics: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
) -> dict:
    """Preprocess image (and optional depth) for WildDet3D.

    Args:
        image: RGB image as numpy array (H, W, 3).
        intrinsics: Camera intrinsics (3, 3), or None to use
            default/predicted.
        depth: Optional depth map (H, W) float32 in meters, at the
            same resolution as `image`. When provided it goes through
            the same resize + center-pad pipeline as the image (using
            the same transforms eval uses), and the result is returned
            under `data["depth_gt"]` (shape `(1, 1, 1008, 1008)`) for
            `model(..., depth_gt=data["depth_gt"].cuda())`. Omit this
            arg to let the model use its monocular LingBot-Depth
            prediction instead.

    Returns:
        Dict with preprocessed tensors and metadata. Always contains
        `images`, `intrinsics`, `original_hw`, `input_hw`, `padding`,
        `original_intrinsics`; contains `depth_gt` iff `depth` was
        provided.
    """
    images = image.astype(np.float32)[None, ...]
    H, W = images.shape[1], images.shape[2]

    # If no intrinsics provided, create a placeholder.
    # When use_predicted_intrinsics=True in the model, the geometry backend's
    # K_pred will be used for 3D box decoding instead of this placeholder.
    # The placeholder is still needed so the data pipeline doesn't crash.
    if intrinsics is None:
        focal = max(H, W)
        intrinsics = np.array(
            [
                [focal, 0, W / 2],
                [0, focal, H / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    data_dict = {
        "images": images,
        "original_images": images.copy(),
        "input_hw": (H, W),
        "original_hw": (H, W),
        "intrinsics": intrinsics.astype(np.float32),
        "original_intrinsics": intrinsics.astype(np.float32).copy(),
    }

    transforms = [
        GenResizeParameters(shape=IMAGE_SIZE),
        ResizeImages(),
        ResizeIntrinsics(),
    ]
    if depth is not None:
        assert depth.shape[:2] == (H, W), (
            f"depth shape {depth.shape[:2]} must match image {H, W}"
        )
        data_dict["depth_maps"] = depth.astype(np.float32)
        transforms.append(ResizeDepthMaps(interpolation="nearest"))
    transforms += [
        NormalizeImages(),
        CenterPadImages(stride=1, shape=IMAGE_SIZE, update_input_hw=True),
        CenterPadIntrinsics(),
    ]
    if depth is not None:
        transforms.append(CenterPadDepthMaps())

    preprocess_transforms = compose(transforms=transforms)

    data = preprocess_transforms([data_dict])[0]
    to_tensor = ToTensor()
    data = to_tensor([data])[0]

    if depth is not None:
        # Expose under the name the model kwarg uses (depth_gt) and add
        # the (B=1, 1, H, W) shape the model expects.
        dm = data["depth_maps"]
        # After ToTensor the shape is (1, H, W) per sample; add a
        # channel dim to get (1, 1, H, W).
        if dm.dim() == 3:
            dm = dm.unsqueeze(1)
        elif dm.dim() == 2:
            dm = dm.unsqueeze(0).unsqueeze(0)
        data["depth_gt"] = dm

    return data
