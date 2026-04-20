"""Stage 1: Omni3D Canonical Training (12 epochs).

This is the first stage of WildDet3D's 3-stage training pipeline.

- Data: Omni3D only (~100K images from 6 sub-datasets)
- Starting point: SAM3 pretrained checkpoint (sam3_detector.pt)
- Key settings: canonical_rotation=True, backbone_freeze=28, depth_freeze=21
- Collator: 5-mode (text + geometry prompts)
- Epochs: 12

Usage:
    # Single node (8 GPUs)
    vis4d fit --config configs/training/stage1_omni3d.py --gpus 8

    # Multi-node (2 nodes, 16 GPUs total)
    vis4d fit --config configs/training/stage1_omni3d.py --gpus 8 --num_nodes 2
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.zoo.base import get_default_cfg

from configs.base.callback import (
    get_callback_cfg,
    get_omni3d_evaluator_cfg,
)
from configs.base.connector import get_wilddet3d_data_connector_cfg
from configs.base.data import (
    get_wilddet3d_data_cfg,
    wilddet3d_5mode_collate_fn,
)
from configs.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.pl import get_pl_cfg


# ============================================================
# Experiment parameters
# ============================================================
EXPERIMENT_NAME = "wilddet3d_stage1_omni3d"

# Data
DATA_ROOT = "data/omni3d"
SAM3_IMAGE_SHAPE = (1008, 1008)
OMNI3D50 = True

# Hyperparameters
NUM_EPOCHS = 12
SAMPLES_PER_GPU = 4
WORKERS_PER_GPU = 8
BASE_LR = 1e-4

# Freeze settings
BACKBONE_FREEZE_BLOCKS = 28  # Freeze first 28/32 ViT blocks
LINGBOT_ENCODER_FREEZE_BLOCKS = 21  # Freeze first 21 LingBot encoder blocks


def get_config() -> ExperimentConfig:
    """Get Stage 1 training config: Omni3D canonical."""
    config = get_default_cfg(exp_name=EXPERIMENT_NAME)

    # ==================== Hyperparameters ====================
    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=NUM_EPOCHS,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        base_lr=BASE_LR,
        nms=True,
        nms_iou_threshold=0.6,
        score_threshold=0.0,
    )

    # ==================== Model ====================
    config.model, box_coder = get_wilddet3d_cfg(
        params,
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=LINGBOT_ENCODER_FREEZE_BLOCKS,
        backbone_freeze_blocks=BACKBONE_FREEZE_BLOCKS,
        canonical_rotation=True,
        use_predicted_intrinsics=True,
        eval_3d_conf_weight=0.5,
    )

    # ==================== Data ====================
    # Omni3D train datasets (6 sub-datasets)
    omni3d_train_datasets = [
        "KITTI_train",
        "KITTI_val",
        "nuScenes_train",
        "nuScenes_val",
        "SUNRGBD_train",
        "SUNRGBD_val",
        "Hypersim_train",
        "Hypersim_val",
        "ARKitScenes_train",
        "ARKitScenes_val",
        "Objectron_train",
        "Objectron_val",
    ]
    omni3d_test_datasets = [
        "KITTI_val",
        "nuScenes_val",
        "SUNRGBD_val",
        "Hypersim_val",
        "ARKitScenes_val",
        "Objectron_val",
    ]

    data_backend = class_config(HDF5Backend)

    train_cfg = get_omni3d_train_cfg(
        data_root=DATA_ROOT,
        train_datasets=omni3d_train_datasets,
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        omni3d50=OMNI3D50,
        with_depth=True,
    )
    test_cfg = get_omni3d_test_cfg(
        data_root=DATA_ROOT,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        omni3d50=OMNI3D50,
        with_depth=True,
    )

    config.data = get_wilddet3d_data_cfg(
        train_datasets=train_cfg,
        test_datasets=test_cfg,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        use_text_prompts=True,
        use_geometry_prompts=True,
    )

    # ==================== Loss ====================
    config.loss = get_wilddet3d_loss_cfg(
        params,
        box_coder=box_coder,
        use_3d_conf=True,
        use_ignore_suppress=True,
    )

    # ==================== Optimizer ====================
    config.optimizers = get_wilddet3d_optim_cfg(params)

    # ==================== Callbacks ====================
    config.callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        open_test_datasets=[],
        omni3d_evaluator=get_omni3d_evaluator_cfg(
            data_root=DATA_ROOT,
            omni3d50=OMNI3D50,
            test_datasets=omni3d_test_datasets,
        ),
    )

    # ==================== Connectors ====================
    (
        config.train_data_connector,
        config.test_data_connector,
    ) = get_wilddet3d_data_connector_cfg()

    # ==================== PL Trainer ====================
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
