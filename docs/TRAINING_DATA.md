# WildDet3D Training Data Preparation

This document covers how to prepare every dataset used during training. Most datasets have license terms that forbid redistributing the raw images, so the pipeline is always the same: **users download the original dataset from the official release, then run one of our scripts in `scripts/data_prep/` to turn it into the Omni3D-style `annotations/*.json` + `{images,depth}.hdf5` layout that the training configs expect.** Our scripts are deterministic — same input → same output — so a user who follows them ends up with the exact same frames we trained on.

## Contents
- [Expected layout under `data/`](#expected-layout-under-data)
- [Omni3D (KITTI, nuScenes, SUNRGBD, ARKitScenes, Hypersim, Objectron)](#omni3d-kitti-nuscenes-sunrgbd-arkitscenes-hypersim-objectron)
- [Cubify Anything (CA1M)](#cubify-anything-ca1m)
- [Waymo Open Dataset v2](#waymo-open-dataset-v2)
- [3EED](#3eed)
- [FoundationPose (synthetic)](#foundationpose-synthetic)
- [In-the-Wild (ITW) + Stereo4D](#in-the-wild-itw-trainvaltest-and-stereo4d-bench)
- [Mask annotations (Stage 3 only)](#mask-annotations-stage-3-only)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Eval-only benchmarks (Argoverse 2, ScanNet, LabelAny3D)](#eval-only-benchmarks-argoverse-2-scannet-labelany3d)

## Expected layout under `data/`

```
data/
├── omni3d/
│   ├── annotations/                # KITTI / nuScenes / SUNRGBD / Hypersim / ARKitScenes / Objectron _{train,val,test}.json
│   └── cache_omni3d50/              # auto-built on first run
├── KITTI_object/
├── KITTI_object.hdf5
├── KITTI_object_depth.hdf5
├── nuscenes/                        # lower-case, matches Omni3D JSON file_path
├── nuscenes.hdf5
├── nuscenes_depth.hdf5
├── SUNRGBD/
├── SUNRGBD.hdf5
├── hypersim/
├── hypersim.hdf5
├── hypersim_depth.hdf5
├── ARKitScenes/
├── ARKitScenes.hdf5
├── ARKitScenes_depth.hdf5
├── objectron/
├── objectron.hdf5
├── objectron_depth.hdf5
├── cubifyanything/
│   ├── annotations/                # CubifyAnything_{train,val}.json
│   ├── data.hdf5
│   └── depth_gt.hdf5
├── waymo/
│   ├── annotations/                # Waymo_{train,val}.json
│   ├── images.hdf5
│   └── depth.hdf5
├── 3eed/
│   ├── annotations/                # 3EED_{det,ref}_{train,val}.json
│   ├── 3eed_dataset.hdf5
│   └── depth/
├── foundationpose/
│   ├── annotations/                # FoundationPose_{train,val}.json
│   ├── images_jpg.hdf5
│   └── depth.hdf5
├── in_the_wild/
│   ├── annotations/                # InTheWild_v3_{train,val,test,...}.json
│   └── images/
└── masks/                           # Stage 3 only
    ├── lvis/
    ├── coco/
    ├── obj365/
    └── v3det/

pretrained/
└── sam3/
    └── sam3_detector.pt
```

## Omni3D (KITTI, nuScenes, SUNRGBD, ARKitScenes, Hypersim, Objectron)

Omni3D is the only multi-split dataset in the pipeline: per Omni3D's protocol, `_train` **and** `_val` are both concatenated as training signal; `_test` is held out for final evaluation. Download all three splits for the 6 sub-datasets.

Follow [3D-MOOD `docs/DATA.md`](https://github.com/cvg/3D-MOOD/blob/main/docs/DATA.md) for download, Omni3D JSON generation, depth-GT generation, and HDF5 conversion — the same document also covers Argoverse 2 and ScanNet (eval-only benchmarks for us).

Quick-start for the Omni3D JSONs:
```bash
cd data
wget https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip
unzip Omni3D_json.zip
mkdir -p omni3d/annotations
cp datasets/Omni3D/*.json omni3d/annotations/
```

Place the rest of the outputs under `data/` per the layout above. `stage1_omni3d.py` uses `HDF5Backend`, so the `.hdf5` archives are required; the per-scene folders are only used during depth-GT generation.

## Cubify Anything (CA1M)

Only the `train` split is used for training (val is not wired into any stage).

```bash
# 1. Download Apple ml-cubifyanything tar files under <CA1M_ROOT>/data/
#    (https://github.com/apple/ml-cubifyanything).

# 2. Convert to Omni3D JSONs + depth (deterministic; every 10th frame by default).
python scripts/data_prep/ca1m/convert_ca1m_to_omni3d.py \
    --ca1m_root <CA1M_ROOT> \
    --output_dir data/cubifyanything \
    --split train --num_workers 8

# 3. Pack to HDF5 for fast loading.
python -m vis4d.data.io.to_hdf5 -p data/cubifyanything/data
python -m vis4d.data.io.to_hdf5 -p data/cubifyanything/depth_gt
```

See [`scripts/data_prep/ca1m/CA1M_CONVERSION_NOTES.md`](../scripts/data_prep/ca1m/CA1M_CONVERSION_NOTES.md) for more context (frame rate, split sizes, ~206K images after stride=10).

## Waymo Open Dataset v2

Only the `training` split is used (val is not wired into any stage). Waymo redistribution is restricted, so everything here starts from your own gsutil-authenticated download.

```bash
# 1. Download Waymo v2 parquet tables (requires accepting the Waymo license
#    and `gcloud auth`). OUTPUT_ROOT env var defaults to data/waymo_v2.
bash scripts/data_prep/waymo/download_waymo_v2_lidar.sh
bash scripts/data_prep/waymo/download_waymo_v2_lidar_camera_projection.sh

# 2. Extract frames + build Omni3D JSON (every 5th frame -> 2Hz from 10Hz).
python scripts/data_prep/waymo/convert_waymo_v2_to_omni3d.py \
    --split training --frame_interval 5 --num_workers 64

# 3. Generate dense depth maps from LiDAR + camera projection parquets.
python scripts/data_prep/waymo/generate_waymo_depth_maps.py \
    --split training --frame_interval 5 --num_workers 64

# 4. Pack to HDF5.
python -m vis4d.data.io.to_hdf5 -p data/waymo/images
python -m vis4d.data.io.to_hdf5 -p data/waymo/depth
```

See [`scripts/data_prep/waymo/WAYMO_CONVERSION_NOTES.md`](../scripts/data_prep/waymo/WAYMO_CONVERSION_NOTES.md).

## 3EED

Only the `train` split is used for training (the converter generates `val` as a side effect, but stage 2 never reads it).

```bash
# 1. Download the 3EED dataset zip and unzip under data/3eed/3eed_dataset/
#    (https://github.com/ZrrSkywalker/3EED for the current release).

# 2. Convert (uses the dataset's own splits/*.txt; no subsampling).
python scripts/data_prep/threeeed/convert_3eed_to_omni3d.py \
    --data_root data/3eed \
    --platforms waymo drone quad \
    --splits train

# 3. Pack to HDF5.
python -m vis4d.data.io.to_hdf5 -p data/3eed/3eed_dataset
```

## FoundationPose (synthetic)

The converter splits the raw FP dump into train/val with a fixed `random.seed(42)`; training only reads the train half.

```bash
# 1. Download FoundationPose synthetic data to <FP_ROOT>:
#    <FP_ROOT>/bbox3d/ and <FP_ROOT>/extracted/{images,camera_params}/.

# 2. Convert (produces FoundationPose_train.json + FoundationPose_val.json).
python scripts/data_prep/foundationpose/convert_foundationpose_fast.py \
    --bbox3d_dir    <FP_ROOT>/bbox3d \
    --extracted_dir <FP_ROOT>/extracted \
    --output_dir    data/foundationpose/annotations \
    --workers 32

# 3. Pack to HDF5.
python -m vis4d.data.io.to_hdf5 -p <FP_ROOT>/extracted/images -o data/foundationpose/images_jpg.hdf5
python -m vis4d.data.io.to_hdf5 -p <FP_ROOT>/extracted/depth  -o data/foundationpose/depth.hdf5
```

## In-the-Wild (ITW) train/val/test and Stereo4D-Bench

Only the images have a redistributable source (COCO / LVIS / Objects365 / V3Det). We release the 3D annotations + estimated depth on HuggingFace:

| Subset | Source |
|---|---|
| ITW train (human + synthetic) + val + test annotations, depth | [`allenai/WildDet3D-Data`](https://huggingface.co/datasets/allenai/WildDet3D-Data) |
| Stereo4D-Bench evaluation images + annotations + depth | [`allenai/WildDet3D-Stereo4D-Bench-Images`](https://huggingface.co/datasets/allenai/WildDet3D-Stereo4D-Bench-Images) |

After download, lay them out as `data/in_the_wild/{annotations,images}`. Originals for COCO / LVIS / Objects365 / V3Det images follow each upstream's own download instructions.

## Mask annotations (Stage 3 only)

Stage 3 samples positive points from instance masks (70% box-only / 30% point-only). Stages 1 and 2 use the plain text + box collator, so they don't need any of the files below.

| Source | Where to get it |
|---|---|
| LVIS (train/val) `lvis_v1_train.json`, `lvis_v1_val.json` | [LVIS](https://www.lvisdataset.org/dataset) (official release) |
| COCO (train/val) `instances_train2017.json`, `instances_val2017.json` | [COCO](https://cocodataset.org/#download) (official release) |
| Objects365 `obj365_{train,val}_with_masks.json` (SAM2-generated) | [`allenai/WildDet3D-Data`](https://huggingface.co/datasets/allenai/WildDet3D-Data) at `masks/obj365/` |
| V3Det `v3det_2023_v1_masks_all.json` (SAM2-generated) | [`allenai/WildDet3D-Data`](https://huggingface.co/datasets/allenai/WildDet3D-Data) at `masks/v3det/` |

Download the two SAM2 JSONs we generated:

```bash
pip install huggingface_hub
hf download allenai/WildDet3D-Data --repo-type dataset \
    --include "masks/obj365/*" "masks/v3det/*" \
    --local-dir data/masks
```

Place LVIS / COCO files under `data/masks/{lvis,coco/annotations}/` to match the paths in `configs/training/stage3_mix_box_point_text_ft.py`. If your layout differs, update `MASK_ROOT` / `MASK_FILES_*` in that config.

## Pretrained checkpoints

```bash
mkdir -p pretrained/sam3
# SAM3 detector backbone (manual download)
wget https://huggingface.co/facebook/sam3/resolve/main/sam3.pt -O pretrained/sam3/sam3_detector.pt

# LingBot-Depth is auto-downloaded the first time training runs
# (robbyant/lingbot-depth-postrain-dc-vitl14, via huggingface_hub).
```

## Eval-only benchmarks (Argoverse 2, ScanNet, LabelAny3D)

These datasets appear only in `configs/eval/...`, **not** in any training stage. Annotation JSONs live under `data/{argoverse,scannet,labelany3d_coco}/annotations/` and the corresponding images follow each dataset's upstream release (3D-MOOD `docs/DATA.md` covers Argoverse 2 and ScanNet; LabelAny3D images are COCO 2017 val).
