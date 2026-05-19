"""DROID dataset config for training and evaluation."""

from __future__ import annotations

import os
from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from wilddet3d.data.datasets.droid import (
    DROIDDataset,
    get_droid_class_map,
    get_droid_det_map,
)

from .transform import get_test_transforms_cfg, get_train_transforms_cfg


def get_droid_dataset_cfg(
    data_root: str,
    datasets: Sequence[str],
    data_backend: None | ConfigDict = None,
    remove_empty: bool = False,
    with_depth: bool = False,
    cache_as_binary: bool = False,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 100,
    truncation_thres: float = 0.33333333,
    visibility_thres: float = 0.33333333,
    min_height_thres: float = 0.0625,
    max_height_thres: float = 1.50,
) -> list[ConfigDict]:
    """Get dataset configs for DROID."""
    if use_mini_dataset:
        cached_dir = os.path.join(
            data_root, f"cache_mini{mini_dataset_size}"
        )
    else:
        cached_dir = os.path.join(data_root, "cache")

    dataset_cfg_list = []
    for dataset in datasets:
        det_map = get_droid_det_map(
            dataset_name=dataset, data_root=data_root
        )
        class_map = get_droid_class_map(
            dataset_name=dataset, data_root=data_root
        )

        dataset_cfg = class_config(
            DROIDDataset,
            class_map=class_map,
            data_backend=data_backend,
            data_root=data_root,
            dataset_name=dataset,
            det_map=det_map,
            with_depth=with_depth,
            remove_empty=remove_empty,
            data_prefix="data",
            cache_as_binary=cache_as_binary,
            cached_file_path=os.path.join(
                cached_dir, f"{dataset}.pkl"
            ),
            truncation_thres=truncation_thres,
            visibility_thres=visibility_thres,
            min_height_thres=min_height_thres,
            max_height_thres=max_height_thres,
        )

        dataset_cfg_list.append(dataset_cfg)

    return dataset_cfg_list


def get_droid_train_cfg(
    data_root: str = "data/droid",
    train_datasets: Sequence[str] = ("DROID_train",),
    data_backend: None | ConfigDict = None,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 100,
    truncation_thres: float = 0.50,
    visibility_thres: float = 0.1,
    min_height_thres: float = 0.0,
    max_height_thres: float = 1.50,
) -> ConfigDict:
    """Get train config for DROID."""
    train_dataset_cfg = get_droid_dataset_cfg(
        data_root=data_root,
        datasets=train_datasets,
        data_backend=data_backend,
        remove_empty=True,
        with_depth=True,
        cache_as_binary=cache_as_binary,
        use_mini_dataset=use_mini_dataset,
        mini_dataset_size=mini_dataset_size,
        truncation_thres=truncation_thres,
        visibility_thres=visibility_thres,
        min_height_thres=min_height_thres,
        max_height_thres=max_height_thres,
    )

    train_preprocess_cfg = get_train_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )


def get_droid_test_cfg(
    data_root: str = "data/droid",
    test_datasets: Sequence[str] = ("DROID_val",),
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
) -> ConfigDict:
    """Get test config for DROID."""
    test_dataset_cfg = get_droid_dataset_cfg(
        data_root=data_root,
        datasets=test_datasets,
        data_backend=data_backend,
        with_depth=with_depth,
        cache_as_binary=cache_as_binary,
    )

    test_preprocess_cfg = get_test_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=test_dataset_cfg,
        preprocess_fn=test_preprocess_cfg,
    )
