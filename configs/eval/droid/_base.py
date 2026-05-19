"""Shared helper for DROID eval configs (4 modes).

Modes (parametrized by `oracle_eval` and `with_depth`):
  text + mono      : oracle_eval=False, with_depth=False
  text + GT depth  : oracle_eval=False, with_depth=True
  oracle + mono    : oracle_eval=True,  with_depth=False
  oracle + GT depth: oracle_eval=True,  with_depth=True
"""

from __future__ import annotations

import os

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.const import AxisMode
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import VisualizerCallback
from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.zoo.base import get_default_callbacks_cfg, get_default_cfg

from configs.base.callback import get_droid_eval_callbacks
from configs.base.connector import (
    WildDet3DDetect3DEvalConnector,
    WildDet3DVisConnector,
    get_wilddet3d_data_connector_cfg,
)
from configs.base.data import get_wilddet3d_data_cfg
from configs.base.dataset.omni3d import get_omni3d_train_cfg
from configs.base.dataset.transform import get_test_transforms_cfg
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.pl import get_pl_cfg

from wilddet3d.data.datasets.droid import DROIDDataset, get_droid_class_map


def build_droid_eval_config(
    exp_name: str,
    oracle_eval: bool,
    with_depth: bool,
    canonical_rotation: bool = True,
) -> ExperimentConfig:
    """Build a DROID eval config for a single (oracle, depth) combo."""
    config = get_default_cfg(exp_name=exp_name)
    config.use_checkpoint = True

    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=1,
        samples_per_gpu=4,
        workers_per_gpu=4,
        base_lr=1e-4,
    )
    config.params = params

    data_backend = class_config(HDF5Backend)
    sam3_image_shape = (1008, 1008)
    omni3d_data_root = "data/omni3d"
    droid_data_root = "data/droid"

    # Train datapipe is required by get_wilddet3d_data_cfg signature even
    # though we only test; reuse Omni3D train cfg as a no-op placeholder.
    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=sam3_image_shape,
    )

    # Use the unified val annotation (DROID_val_unified.json) which collapses
    # color/material/compound-noun variants to their head noun (e.g.,
    # "yellow block" -> "block", "glass lid" -> "lid"). Reduces val vocab
    # from 210 -> 146 categories with cleaner per-class AP signal.
    eval_dataset_name = "DROID_val_unified"

    class_map = get_droid_class_map(
        eval_dataset_name, data_root=droid_data_root
    )

    cache_name = (
        f"{eval_dataset_name}_depth.pkl" if with_depth
        else f"{eval_dataset_name}.pkl"
    )

    droid_test_data_cfg = class_config(
        DataPipe,
        datasets=class_config(
            DROIDDataset,
            data_root=droid_data_root,
            dataset_name=eval_dataset_name,
            class_map=class_map,
            det_map=class_map,
            with_depth=with_depth,
            data_backend=data_backend,
            data_prefix="data",
            cache_as_binary=True,
            # DROID-specific: disable Omni3D-style filters that drop
            # small/edge/full-frame manipulation objects.
            min_height_thres=0.0,
            max_height_thres=10.0,
            truncation_thres=1.0,
            cached_file_path=os.path.join(
                droid_data_root, "cache",
                cache_name.replace(".pkl", "_loose.pkl"),
            ),
        ),
        preprocess_fn=get_test_transforms_cfg(
            shape=sam3_image_shape, with_depth=with_depth
        ),
    )

    config.data = get_wilddet3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=[droid_test_data_cfg],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        use_geometry_prompts=True,
        oracle_eval=oracle_eval,
    )

    config.model, box_coder = get_wilddet3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=21,
        backbone_freeze_blocks=28,
        canonical_rotation=canonical_rotation,
        oracle_eval=oracle_eval,
        use_depth_input_test=with_depth,
        unpad_test=True,
    )

    config.loss = get_wilddet3d_loss_cfg(params, box_coder)

    config.optimizers = get_wilddet3d_optim_cfg(
        params,
        freeze_backbone=params.freeze_backbone,
        freeze_all_pretrained=params.freeze_all_pretrained,
    )

    config.train_data_connector, config.test_data_connector = (
        get_wilddet3d_data_connector_cfg()
    )

    callbacks = get_default_callbacks_cfg()
    callbacks.extend(
        get_droid_eval_callbacks(
            data_root=droid_data_root,
            output_dir=config.output_dir,
            test_connector=class_config(WildDet3DDetect3DEvalConnector),
        )
    )
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBox3DVisualizer,
                axis_mode=AxisMode.OPENCV,
                width=4,
                camera_near_clip=0.01,
                plot_heading=False,
                vis_freq=50,
                plot_trajectory=False,
                canvas=class_config(PillowCanvasBackend, font_size=16),
                save_boxes3d=True,
            ),
            output_dir=config.output_dir,
            save_prefix="box3d",
            test_connector=class_config(
                WildDet3DVisConnector, score_threshold=0.0
            ),
        )
    )
    config.callbacks = callbacks

    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
