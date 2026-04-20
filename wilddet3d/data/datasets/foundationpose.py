"""FoundationPose (GSO) dataset for 3D object detection.

Synthetic dataset from FoundationPose with Google Scanned Objects (GSO).
438 categories, ~446K images with dense depth maps (uint16, depth_m * 256).
"""

from __future__ import annotations

import json
import os

from vis4d.common.typing import ArgsType, DictStrAny

from wilddet3d.data.datasets.coco3d import COCO3DDataset


def get_foundationpose_det_map(
    dataset_name: str,
    data_root: str = "data/foundationpose",
) -> dict[str, int]:
    """Build det_map from FoundationPose annotation JSON categories."""
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


def get_foundationpose_class_map(
    dataset_name: str,
    data_root: str = "data/foundationpose",
) -> dict[str, int]:
    """Build class_map from FoundationPose annotation JSON categories."""
    return get_foundationpose_det_map(dataset_name, data_root)


class FoundationPoseDataset(COCO3DDataset):
    """FoundationPose (GSO) Dataset.

    Synthetic scenes with dense depth maps (uint16, depth_m * 256).
    """

    def __init__(
        self,
        max_depth: float = 20.0,
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
          foundationpose/images_jpg/gso/{name}.jpg
          -> foundationpose/depth/gso/{name}.png
        """
        path = img["file_path"]
        path = path.replace(
            "foundationpose/images_jpg/", "foundationpose/depth/"
        )
        path = path.replace(".jpg", ".png")
        return path
