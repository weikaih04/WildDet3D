"""DROID single-frame 3D box dataset.

One image per episode taken from the wrist stereo camera at the SAM3D
pipeline's `best_frame`. Depth maps are FoundationStereo (uint16 mm).
Categories are free-form `object_name` from the per-episode VLM result.
"""

from __future__ import annotations

import json
import os

from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import CommonKeys as K

from wilddet3d.data.datasets.coco3d import COCO3DDataset


def get_droid_det_map(
    dataset_name: str,
    data_root: str = "data/droid",
) -> dict[str, int]:
    """Build det_map from DROID annotation JSON categories."""
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


def get_droid_class_map(
    dataset_name: str,
    data_root: str = "data/droid",
) -> dict[str, int]:
    return get_droid_det_map(dataset_name, data_root)


class DROIDDataset(COCO3DDataset):
    """DROID Dataset.

    Wrist-camera RGB at best_frame + FoundationStereo depth (uint16 mm).
    """

    def __init__(
        self,
        max_depth: float = 20.0,
        depth_scale: float = 1000.0,
        per_image_categories: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            depth_scale=depth_scale,
            **kwargs,
        )
        # Required for GDino/3D-MOOD eval to avoid BERT truncation
        # with 558 DROID categories (~3000 tokens > 256 BERT cap).
        self.per_image_categories = per_image_categories

    def __getitem__(self, idx: int):
        data_dict = super().__getitem__(idx)
        if self.per_image_categories:
            class_ids_in_img = data_dict[K.boxes2d_classes]
            if len(class_ids_in_img) > 0:
                unique_global_ids = sorted(set(class_ids_in_img.tolist()))
                data_dict[K.boxes2d_names] = [
                    self.categories[gid] for gid in unique_global_ids
                ]
            else:
                data_dict[K.boxes2d_names] = []
        return data_dict

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """data/DROID/<eid>.jpg -> depth/<eid>.png"""
        return img["file_path"].replace(
            "data/DROID", "depth"
        ).replace(".jpg", ".png")
