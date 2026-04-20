"""Stage 3: High-Quality Mixed Box / Point / Text Prompt Finetuning (3 epochs).

This is the third and final stage of WildDet3D's 3-stage training pipeline.
Stage 3 trains on a mix of all three prompt modes -- text, box geometry,
and point geometry -- on the high-quality human-reviewed subset. Points
are sampled from instance masks on the fly (70% box-only, 30% point-only
per step); masks are never regressed as a target.

- Data: 2 datasets only (simplified, high-quality)
    - Omni3D (90%) + ITW-human (10%)
- Starting point: Stage 2 checkpoint (--ckpt)
- Collator: 5-mode mask_pt (exclusive point/box: 70% box-only, 30%
    point-only when mask available)
- Key settings: canonical_rotation=True, backbone_freeze=28, depth_freeze=21
- Epochs: 3 (lr schedule: step_1=1, step_2=2)
- LR: 1e-4 (lower than stage 2)
- presence_loss_weight=5.0

Usage:
    vis4d fit --config configs/training/stage3_mix_box_point_text_ft.py --gpus 8 \
        --ckpt <path_to_stage2_checkpoint>
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io import FileBackend
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
    wilddet3d_5mode_mask_pt_collate_fn,
    wilddet3d_test_collate_fn,
)
from configs.base.dataset.in_the_wild import get_in_the_wild_train_cfg
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
EXPERIMENT_NAME = "wilddet3d_stage3_mix_box_point_text_ft"

SAM3_IMAGE_SHAPE = (1008, 1008)

# Hyperparameters
NUM_EPOCHS = 3
SAMPLES_PER_GPU = 4
WORKERS_PER_GPU = 4
BASE_LR = 1e-4
STEP_1 = 1
STEP_2 = 2

# Freeze settings
BACKBONE_FREEZE_BLOCKS = 28
LINGBOT_ENCODER_FREEZE_BLOCKS = 21

# Mask annotation files for point prompt training
# Update these paths to match your data layout
MASK_ROOT = "data/masks"
MASK_FILES_COCO_OBJ365 = {
    "coco/train": [
        f"{MASK_ROOT}/lvis/lvis_v1_train.json",
        f"{MASK_ROOT}/coco/annotations/instances_train2017.json",
    ],
    "coco/val": [
        f"{MASK_ROOT}/lvis/lvis_v1_val.json",
        f"{MASK_ROOT}/coco/annotations/instances_val2017.json",
    ],
    "obj365/train": f"{MASK_ROOT}/obj365/obj365_train_with_masks.json",
    "obj365/val": f"{MASK_ROOT}/obj365/obj365_val_with_masks.json",
}

# Dataset sampling proportions (2 datasets, must sum to 1.0)
TARGET_PROPORTIONS = [
    0.90,    # 0: Omni3D
    0.10,    # 1: ITW human
]
EPOCH_DATASET_IDX = 0  # 1 epoch = Omni3D sees all samples once


def get_config() -> ExperimentConfig:
    """Get Stage 3 training config: mixed box / point / text prompt finetuning on high-quality data."""
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
    omni3d_data_root = "data/omni3d"
    omni3d_test_datasets = [
        "KITTI_test", "nuScenes_test", "SUNRGBD_test",
        "Hypersim_test", "ARKitScenes_test", "Objectron_test",
    ]

    hdf5_backend = class_config(HDF5Backend)
    file_backend = class_config(FileBackend)  # ITW (no hdf5 for v3det subtree)

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

    # ITW v3 human only (COCO/LVIS/Obj365) + masks
    itw_human_train = get_in_the_wild_train_cfg(
        data_root="data/in_the_wild",
        train_dataset="InTheWild_v3_train_human",
        data_backend=file_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
        depth_confidence_threshold=128,
        mask_annotation_files=MASK_FILES_COCO_OBJ365,
    )

    # 2 datasets: Omni3D 90% + ITW human 10%
    combined_train = class_config(
        DataPipe,
        datasets=[
            omni3d_train,      # 0: Omni3D
            itw_human_train,   # 1: ITW human
        ],
    )

    # Build data config
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
        collate_fn=wilddet3d_5mode_mask_pt_collate_fn,
    )

    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )
    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=class_config(DataPipe, datasets=[omni3d_test]),
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
        presence_loss_weight=5.0,
    )

    # ==================== Optimizer ====================
    config.optimizers = get_wilddet3d_optim_cfg(params)

    # ==================== Callbacks ====================
    config.callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        open_test_datasets=[],
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
