"""Waymo Open Dataset for 3D object detection.

Outdoor driving scenes with sparse LiDAR depth maps (uint16, depth_m * 256).
Categories: vehicle, pedestrian, cyclist, sign.
"""

from __future__ import annotations

import json
import os

from vis4d.common.typing import ArgsType, DictStrAny

from wilddet3d.data.datasets.coco3d import COCO3DDataset


def get_waymo_det_map(
    dataset_name: str,
    data_root: str = "data/waymo",
) -> dict[str, int]:
    """Build det_map from Waymo annotation JSON categories.

    Waymo has 4 categories (vehicle, pedestrian, cyclist, sign).
    Since our model is open-vocabulary (text-prompted), we build
    det_map dynamically from the annotation JSON.

    Args:
        dataset_name: e.g. "Waymo_train" or "Waymo_val"
        data_root: Root directory for Waymo data.
    """
    cache_path = os.path.join(
        data_root, "annotations", f"{dataset_name}_class_map.json"
    )
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    json_path = os.path.join(
        data_root, "annotations", f"{dataset_name}.json"
    )
    with open(json_path) as f:
        data = json.load(f)
    class_map = {cat["name"]: cat["id"] for cat in data["categories"]}
    with open(cache_path, "w") as f:
        json.dump(class_map, f)
    return class_map


def get_waymo_class_map(
    dataset_name: str,
    data_root: str = "data/waymo",
) -> dict[str, int]:
    """Build class_map from Waymo annotation JSON categories.

    Args:
        dataset_name: e.g. "Waymo_train" or "Waymo_val"
        data_root: Root directory for Waymo data.
    """
    return get_waymo_det_map(dataset_name, data_root)


class WaymoDataset(COCO3DDataset):
    """Waymo Open Dataset.

    Outdoor driving scenes with sparse LiDAR depth maps.
    """

    def __init__(
        self,
        max_depth: float = 80.0,
        depth_scale: float = 256.0,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__(
            max_depth=max_depth,
            depth_scale=depth_scale,
            **kwargs,
        )

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Get the depth filename for a given image.

        Maps image path to depth path:
          waymo/images/validation/xxx.jpg
          -> waymo/depth/validation/xxx.png
        """
        return img["file_path"].replace(
            "images", "depth"
        ).replace(".jpg", ".png")
