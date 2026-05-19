"""Stage 4: DROID add-on finetune from Stage 3 (canonical_neg_pt) ckpt.

Adds DROID as a 2nd training dataset alongside Omni3D, finetuning from
the Stage 3 mixed-prompt checkpoint.

- Data: 2 datasets
    - Omni3D (70%, preserves Stage 3 capability)
    - DROID  (30%, ~22K samples drives epoch length)
- Starting point: Stage 3 checkpoint (--ckpt)
- Collator: `wilddet3d_5mode_neg_pt_collate_fn`. DROID samples have no
    instance masks, so the collator's `_sample_points_for_box` falls back
    to `sample_points_without_mask` (verified: never crashes; DROID
    contributes box+text only, Omni3D still contributes box+text+point).
- Sampler: `build_train_dataloader_with_ratios` with `epoch_dataset_idx=1`
    so 1 epoch = DROID seen once (~22K / 0.30 = ~73K total samples).
- Key settings: canonical_rotation=True, backbone_freeze=28,
    depth_freeze=21, eval_3d_conf_weight=0.5
- Epochs: 3 (lr schedule: step_1=0, step_2=1)
- LR: 1e-4

Usage:
    vis4d fit --config configs/training/stage4_droid_ft.py --gpus 8 \\
        --ckpt <path_to_stage3_checkpoint>
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_default_cfg,
    get_inference_dataloaders_cfg,
)

from configs.base.callback import (
    get_callback_cfg,
    get_omni3d_evaluator_cfg,
)
from configs.base.connector import get_wilddet3d_data_connector_cfg
from configs.base.data import (
    wilddet3d_5mode_neg_pt_collate_fn,
    wilddet3d_test_collate_fn,
)
from configs.base.dataset.droid import (
    get_droid_test_cfg,
    get_droid_train_cfg,
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

from wilddet3d.data.samplers import build_train_dataloader_with_ratios


# ============================================================
# Experiment parameters
# ============================================================
EXPERIMENT_NAME = "wilddet3d_stage4_droid_ft"

SAM3_IMAGE_SHAPE = (1008, 1008)

NUM_EPOCHS = 3
SAMPLES_PER_GPU = 4
WORKERS_PER_GPU = 8
BASE_LR = 1e-4
STEP_1 = 0
STEP_2 = 1

BACKBONE_FREEZE_BLOCKS = 28
LINGBOT_ENCODER_FREEZE_BLOCKS = 21

# Dataset sampling proportions (must sum to 1.0)
TARGET_PROPORTIONS = [
    0.70,    # 0: Omni3D
    0.30,    # 1: DROID
]
EPOCH_DATASET_IDX = 1  # 1 epoch = DROID seen once


def get_config() -> ExperimentConfig:
    """Stage 4: DROID + Omni3D mixed finetune from Stage 3 ckpt."""
    config = get_default_cfg(exp_name=EXPERIMENT_NAME)
    config.use_checkpoint = True

    # ==================== Hyperparameters ====================
    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=NUM_EPOCHS,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        base_lr=BASE_LR,
        step_1=STEP_1,
        step_2=STEP_2,
        nms=True,
        nms_iou_threshold=0.6,
        score_threshold=0.05,
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
    hdf5_backend = class_config(HDF5Backend)

    omni3d_data_root = "data/omni3d"
    omni3d_test_datasets = [
        "KITTI_test", "nuScenes_test", "SUNRGBD_test",
        "Hypersim_test", "ARKitScenes_test", "Objectron_test",
    ]

    omni3d_train = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=hdf5_backend,
        shape=SAM3_IMAGE_SHAPE,
    )
    omni3d_test = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=hdf5_backend,
        shape=SAM3_IMAGE_SHAPE,
        with_depth=True,
    )

    droid_train = get_droid_train_cfg(
        data_root="data/droid",
        train_datasets=("DROID_train",),
        data_backend=hdf5_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
    )
    droid_test = get_droid_test_cfg(
        data_root="data/droid",
        test_datasets=("DROID_val",),
        data_backend=hdf5_backend,
        shape=SAM3_IMAGE_SHAPE,
        with_depth=True,
    )

    combined_train = class_config(
        DataPipe,
        datasets=[
            omni3d_train,    # 0: Omni3D
            droid_train,     # 1: DROID
        ],
    )

    data = DataConfig()

    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )
    data.train_dataloader = class_config(
        build_train_dataloader_with_ratios,
        dataset=combined_train,
        target_proportions=TARGET_PROPORTIONS,
        epoch_dataset_idx=EPOCH_DATASET_IDX,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        batchprocess_fn=train_batchprocess_cfg,
        collate_fn=wilddet3d_5mode_neg_pt_collate_fn,
    )

    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )
    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=class_config(
            DataPipe, datasets=[omni3d_test, droid_test]
        ),
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=1,
        workers_per_gpu=WORKERS_PER_GPU,
        collate_fn=wilddet3d_test_collate_fn,
    )

    config.data = data

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
        open_test_datasets=["DROID_val"],
        omni3d_evaluator=get_omni3d_evaluator_cfg(
            data_root=omni3d_data_root,
            omni3d50=True,
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
