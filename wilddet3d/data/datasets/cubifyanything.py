"""CubifyAnything (CA-1M) dataset for 3D object detection."""

from __future__ import annotations

import json
import os

from vis4d.common.typing import ArgsType, DictStrAny

from wilddet3d.data.datasets.coco3d import COCO3DDataset


def get_cubifyanything_det_map(
    dataset_name: str,
    data_root: str = "data/cubifyanything",
) -> dict[str, int]:
    """Build det_map from CA-1M annotation JSON categories.

    CA-1M has ~3000 free-form categories. Since our model is
    open-vocabulary (text-prompted), we build det_map dynamically
    from the annotation JSON's categories list.

    Args:
        dataset_name: e.g. "CubifyAnything_train" or "CubifyAnything_val"
        data_root: Root directory for CubifyAnything data.
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


def get_cubifyanything_class_map(
    dataset_name: str,
    data_root: str = "data/cubifyanything",
) -> dict[str, int]:
    """Build class_map from CA-1M annotation JSON categories.

    CA-1M has ~3000 categories (not in omni3d_class_map), so
    we build class_map dynamically from the annotation JSON.
    class_map maps category_name -> category_id (same as det_map
    for CA-1M, since all categories are trainable).

    Args:
        dataset_name: e.g. "CubifyAnything_train" or "CubifyAnything_val"
        data_root: Root directory for CubifyAnything data.
    """
    return get_cubifyanything_det_map(dataset_name, data_root)


class CubifyAnything(COCO3DDataset):
    """CubifyAnything (CA-1M) Dataset.

    Indoor scenes with uint16 mm-encoded depth maps.
    """

    def __init__(
        self,
        max_depth: float = 20.0,
        depth_scale: float = 1000.0,
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
          cubifyanything/data/CubifyAnything/train/42446540/ts.jpg
          -> cubifyanything/depth_gt/train/42446540/ts.png
        """
        return img["file_path"].replace(
            "data/CubifyAnything", "depth_gt"
        ).replace(".jpg", ".png")
