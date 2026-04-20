# Waymo v2 to Omni3D Conversion

Convert the [Waymo Open Dataset v2](https://waymo.com/open/) parquet tables into the Omni3D JSON layout expected by WildDet3D. Deterministic given `--frame_interval` (default 1; we train at 5 = 2 Hz from the 10 Hz source). Only the FRONT camera is used.

Waymo's license forbids redistributing the raw data, so users always run this pipeline from their own gsutil-authenticated download.

Only the `training` split is used for WildDet3D training (val isn't wired into any stage).

## Download

```bash
gcloud auth login

# Annotations + images + LiDAR + camera projections for the training split.
# OUTPUT_ROOT env var defaults to data/waymo_v2.
bash scripts/data_prep/waymo/download_waymo_v2_lidar.sh
bash scripts/data_prep/waymo/download_waymo_v2_lidar_camera_projection.sh
```

## Convert

```bash
# Extract FRONT-camera frames + build Omni3D JSON (2 Hz = every 5th frame of 10 Hz)
python scripts/data_prep/waymo/convert_waymo_v2_to_omni3d.py \
    --split training --frame_interval 5 --num_workers 64

# Dense depth maps from LiDAR + camera projection
python scripts/data_prep/waymo/generate_waymo_depth_maps.py \
    --split training --frame_interval 5 --num_workers 64
```

## HDF5 pack (required for training)

```bash
python -m vis4d.data.io.to_hdf5 -p data/waymo/images
python -m vis4d.data.io.to_hdf5 -p data/waymo/depth
```

## Output layout

```
data/waymo/
├── annotations/
│   └── Waymo_train.json
├── images.hdf5               # FRONT camera JPEG frames
└── depth.hdf5                # LiDAR depth (uint16 PNG, depth_m * 256)
```

Stage 2 training picks up the two HDF5 files automatically via `HDF5Backend`.
