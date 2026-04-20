"""Stage 2: All-Data Dense Fine-Tuning (12 epochs).

This is the second stage of WildDet3D's 3-stage training pipeline.

- Data: 8 datasets with ratio sampling (human-only ITW + V3Det)
    - Omni3D (80%), CA-1M (3%), Waymo (2%), 3EED-det (1%),
      3EED-ref (1%), FoundationPose (6%),
      ITW-human (5%), V3Det-human (2%)
- Starting point: Stage 1 checkpoint (--ckpt)
- Collator: 5-mode (text + box geometry prompts; no mask-point branch)
- Key settings: canonical_rotation=True, backbone_freeze=28, depth_freeze=21
- Epochs: 12 (lr schedule: step_1=6, step_2=9)
- presence_loss_weight=5.0

Mask-based point-prompt training is introduced only in Stage 3.

Usage:
    vis4d fit --config configs/training/stage2_alldata.py --gpus 8 \
        --ckpt <path_to_stage1_checkpoint>
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
    wilddet3d_5mode_collate_fn,
    wilddet3d_test_collate_fn,
)
from configs.base.dataset.cubifyanything import get_ca1m_train_cfg
from configs.base.dataset.foundationpose import get_foundationpose_train_cfg
from configs.base.dataset.in_the_wild import get_in_the_wild_train_cfg
from configs.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from configs.base.dataset.threeeed import get_threeeed_train_cfg
from configs.base.dataset.waymo import get_waymo_train_cfg
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
EXPERIMENT_NAME = "wilddet3d_stage2_alldata"

SAM3_IMAGE_SHAPE = (1008, 1008)

# Hyperparameters
NUM_EPOCHS = 12
SAMPLES_PER_GPU = 4
WORKERS_PER_GPU = 4
BASE_LR = 1e-4
STEP_1 = 6
STEP_2 = 9

# Freeze settings
BACKBONE_FREEZE_BLOCKS = 28
LINGBOT_ENCODER_FREEZE_BLOCKS = 21

# Dataset sampling proportions (8 datasets, must sum to 1.0)
TARGET_PROPORTIONS = [
    0.80,    # 0: Omni3D
    0.03,    # 1: CA-1M
    0.02,    # 2: Waymo
    0.01,    # 3: 3EED det
    0.01,    # 4: 3EED ref
    0.06,    # 5: FoundationPose
    0.05,    # 6: ITW human
    0.02,    # 7: V3Det human
]
EPOCH_DATASET_IDX = 0  # 1 epoch = Omni3D sees all samples once


def get_config() -> ExperimentConfig:
    """Get Stage 2 training config: all data + mask point training."""
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

    data_backend = class_config(HDF5Backend)
    file_backend = class_config(FileBackend)  # ITW / V3Det (no hdf5)

    omni3d_train = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
    )
    omni3d_test = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        with_depth=True,
    )

    ca1m_train = get_ca1m_train_cfg(
        data_root="data/cubifyanything",
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
    )
    waymo_train = get_waymo_train_cfg(
        data_root="data/waymo",
        train_datasets=("Waymo_train",),
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
    )
    threeeed_det_train = get_threeeed_train_cfg(
        data_root="data/3eed",
        train_datasets=("3EED_det_train",),
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
    )
    threeeed_ref_train = get_threeeed_train_cfg(
        data_root="data/3eed",
        train_datasets=("3EED_ref_train",),
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
    )
    fp_train = get_foundationpose_train_cfg(
        data_root="data/foundationpose",
        train_datasets=("FoundationPose_train",),
        data_backend=data_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
    )

    # ITW v3 human only (COCO/LVIS/Obj365). FileBackend works for both the
    # per-image jpgs under images/{coco_train,obj365_train}/ and the .npz
    # depth under depth/train_human/.
    itw_human_train = get_in_the_wild_train_cfg(
        data_root="data/in_the_wild",
        train_dataset="InTheWild_v3_train_human",
        data_backend=file_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
        depth_confidence_threshold=128,
    )

    # V3Det human only. No HDF5 archive exists for V3Det images (they live
    # under images/v3det_train/{Q_id}/); use FileBackend.
    v3det_human_train = get_in_the_wild_train_cfg(
        data_root="data/in_the_wild",
        train_dataset="InTheWild_v3_v3det_human",
        data_backend=file_backend,
        shape=SAM3_IMAGE_SHAPE,
        cache_as_binary=True,
        depth_confidence_threshold=128,
    )

    # Combine 8 datasets (human-only)
    combined_train = class_config(
        DataPipe,
        datasets=[
            omni3d_train,        # 0: Omni3D        ~170K
            ca1m_train,          # 1: CA-1M         ~195K
            waymo_train,         # 2: Waymo          ~79K
            threeeed_det_train,  # 3: 3EED det       ~12K
            threeeed_ref_train,  # 4: 3EED ref       ~11K
            fp_train,            # 5: FP            ~424K
            itw_human_train,     # 6: ITW human      ~89K
            v3det_human_train,   # 7: V3Det human    ~14K
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
        collate_fn=wilddet3d_5mode_collate_fn,
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
