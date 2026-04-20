"""3EED dataset for 3D object detection.

Multi-platform outdoor scenes (Waymo vehicle, M3ED drone, M3ED quadruped)
with sparse LiDAR depth maps (uint16, depth_m * 256).
Categories: car, pedestrian, bus, truck, othervehicle, cyclist.
"""

from __future__ import annotations

import json
import os

from vis4d.common.typing import ArgsType, DictStrAny

from wilddet3d.data.datasets.coco3d import COCO3DDataset


def get_threeeed_det_map(
    dataset_name: str,
    data_root: str = "data/3eed",
) -> dict[str, int]:
    """Build det_map from 3EED annotation JSON categories."""
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


def get_threeeed_class_map(
    dataset_name: str,
    data_root: str = "data/3eed",
) -> dict[str, int]:
    """Build class_map from 3EED annotation JSON categories."""
    return get_threeeed_det_map(dataset_name, data_root)


class ThreeEEDDataset(COCO3DDataset):
    """3EED Dataset.

    Multi-platform outdoor scenes with sparse LiDAR depth maps.
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
          3eed/3eed_dataset/{platform}/{seq}/{frame}/image.jpg
          -> 3eed/depth/{platform}/{seq}/{frame}.png
        """
        # image: 3eed/3eed_dataset/waymo/seq/frame/image.jpg
        # depth: 3eed/depth/waymo/seq/frame.png
        path = img["file_path"]
        parts = path.replace("3eed/3eed_dataset/", "3eed/depth/")
        parts = parts.replace("/image.jpg", ".png")
        return parts
