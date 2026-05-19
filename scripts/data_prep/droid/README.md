# DROID single-frame 3D box -> Omni3D Conversion

Multi-view conversion: each accepted episode emits up to **3 samples**
(one per camera: wrist + ext1 + ext2). The same physical 3D box is
projected into each camera, gravity-normalized in that camera's frame.

## Scripts

| Script | Purpose |
|--------|---------|
| `convert_droid_to_omni3d.py` | Main conversion. Reads our DROID single-frame pipeline outputs (VLM + depth + extrinsics + box JSON) and emits Omni3D-style `DROID_train.json` / `DROID_val.json` + per-view JPEGs + uint16-mm PNG depth maps. |
| `build_unified_val.py` | Collapse free-form val categories to single-word head nouns (e.g. `yellow block` → `block`) to produce `DROID_val_unified.json` for cleaner AP. |
| `visualize_droid_vis4d.py` | Sanity-check converted samples by drawing 3D boxes via vis4d's `BoundingBox3DVisualizer`. |

All scripts take their input paths as required CLI args; no absolute
paths are baked in.

### Step-by-step workflow

```bash
# 1. Smoke test (5 episodes, single thread, all 3 views).
#    Replace the four pipeline-output paths with your local roots.
python scripts/data_prep/droid/convert_droid_to_omni3d.py \
    --box_dir       <pipeline_output>/step3b1_singleframe_box \
    --depth_h5_dir  <pipeline_output>/step2a_depth \
    --step2b_dir    <pipeline_output>/step2b_extrinsics \
    --vlm_dir       <pipeline_output>/step1_vlm_results \
    --droid_raw_root <droid_raw>/1.0.1 \
    --out_dir       data/droid \
    --max_episodes 5 --num_workers 0

# 2. Verify with vis4d's standard 3D box visualizer.
python scripts/data_prep/droid/visualize_droid_vis4d.py \
    --json_path data/droid/annotations/DROID_val.json \
    --num_samples 20

# 3. Full 3-view conversion (~10k episodes -> ~24k frames).
python scripts/data_prep/droid/convert_droid_to_omni3d.py \
    --box_dir ... --depth_h5_dir ... --step2b_dir ... --vlm_dir ... \
    --droid_raw_root ... --out_dir data/droid --num_workers 64

# Or wrist-only (~10k episodes -> ~8k frames).
python scripts/data_prep/droid/convert_droid_to_omni3d.py \
    ... --views wrist --num_workers 64

# 4. Build the unified-category val file used for eval.
python scripts/data_prep/droid/build_unified_val.py --droid_dir data/droid

# 5. Convert to HDF5 for fast distributed IO.
python -m vis4d.data.io.to_hdf5 -p data/droid/data    # ~2.7 GB
python -m vis4d.data.io.to_hdf5 -p data/droid/depth   # ~7.6 GB

# 6. Delete stale pkl cache after JSON regen.
rm -f data/droid/cache/DROID_train.pkl data/droid/cache/DROID_val.pkl
```

## Source Data

### Pipeline outputs

| Component | Path | Purpose |
|-----------|------|---------|
| step1 VLM | `video_data_experiments/droid/output/step1_vlm_results/vlm_result_<eid>.json` | `object_name`, `suitable_for_tracking`, source `file_path` for episode-dir resolution |
| step2a depth | `video_data_experiments/droid_pipeline/output/step2a_depth/<eid>_depth.h5` | FoundationStereo depth (uint16 mm) per camera (`wrist/depth`, `ext1/depth`, `ext2/depth`) + per-camera `intrinsic` (3x3) |
| step2b extrinsics | `video_data_experiments/droid_pipeline/output/step2b_extrinsics/<eid>_cameras.json` | Per ext-cam serial: `optimized_extrinsics` (4x4 w2c, Franka world) + `measured_intrinsics` (3x3) |
| step3b1 box | `video_data_experiments/droid_pipeline/output/step3b1_singleframe_box/<eid>_box.json` (+ `.done`) | 3D OBB in OpenCV wrist-cam frame at `best_frame`. Also has `box_world` and `wrist_T_c2w`. |
| Episode raw | `video_data_experiments/droid/droid_raw/1.0.1/<source>/success/<date>/<run>/recordings/MP4/<serial>-stereo.mp4` | One MP4 per camera (wrist + ext1 + ext2). Side-by-side stereo, left half = rectified left view. Serials looked up from `metadata_*.json`. |

### Counts (final, 3-view, as of 2026-05-01)

| Component | Count |
|-----------|-------|
| VLM candidates | 95,659 |
| step3b1 box.json (with `.done`) | 10,120 |
| After `suitable_for_tracking == "YES"` filter | 8,273 episodes |
| Successful frames (3 views) | **24,469** |
| Train | **22,026** |
| Val | **2,443** |
| Per-view: wrist | 8,273 |
| Per-view: ext1 | 8,098 (175 episodes lacked ext1 in step2b) |
| Per-view: ext2 | 8,098 |

Each output sample = one episode at its `best_frame`, projected to one
camera. Filename suffix `_<view>` (e.g. `shard00000_ep000_wrist.jpg`,
`..._ext1.jpg`, `..._ext2.jpg`).

### box JSON fields used

| Field | Notes |
|-------|-------|
| `episode_id` | `shard<NNNNN>_ep<NNN>`. Last digit of shard determines split. |
| `object_name` | Free-form category, lowercased + stripped at convert time. |
| `best_frame` | Integer frame index (single frame per episode for all 3 views). |
| `wrist_serial` | Wrist MP4 serial. Ext serials come from `metadata_*.json`. |
| `image_hw` | `[H, W]` for the wrist view. Ext views read shape from depth h5. |
| `intrinsic_K` | 3x3 wrist intrinsic at `best_frame` (rectified). |
| `wrist_T_c2w` | 4x4 wrist camera-to-world (Franka base). Used (i) to compute gravity in wrist cam, (ii) to transform box from wrist-cam to world for ext views. |
| `box_camera.center / size / quat_wxyz` | 3D box in OpenCV wrist-cam frame. Source of truth for wrist view; for ext views we reconstruct via world. |
| `box_world` | NOT directly used by converter — we derive world-frame box from `box_camera + wrist_T_c2w` (verified equivalent to machine precision: max diff = 2.2e-16 m). |

## Filter Rules

1. **VLM filter**: skip episode if `vlm_result.suitable_for_tracking != "YES"`.
2. **Existence**: require both `<eid>_box.json` and `<eid>.done`.
3. **No extrinsics re-check**: step3b1 already drops episodes whose step2b `final_loss >= 0.08` (README L173-176), so by the time we see a `_box.json` the camera extrinsics are valid.
4. **Per-view existence**: ext views are skipped silently if the serial is missing from step2b cameras JSON. Wrist always succeeds when episode passes.
5. **No quality threshold**: keep all `sam3_score` values; let dataloader's `is_ignore()` handle box-size / truncation filtering.

## Coordinate Systems

### Frames

- **Franka base (world)**: Z-up, gravity = `[0, 0, -1]`.
- **Wrist OpenCV camera**: X-right, Y-down, Z-forward. `wrist_T_c2w` maps cam-to-world.
- **Ext OpenCV camera**: same OpenCV convention. `step2b.optimized_extrinsics` is `world->cam` (`T_w2c_ext`).
- **vis4d canonical box**: `dimensions=[W, H, L]`; local axes X=L, Y=H, Z=W.

### DROID box-local axes

DROID `box_camera.size = [w, h, l]` and `quat_wxyz` decodes to a rotation
whose **columns are** `(w_dir, h_dir, l_dir)`, i.e. local X=W, Y=H, Z=L —
**X and Z are swapped vs vis4d**.

### Mapping (column rearrangement — applied to wrist view)

Identical to FoundationPose's `P_FP_TO_VIS4D`:

```python
P = [[0, 0,  1],
     [0, 1,  0],
     [-1, 0, 0]]                     # Ry(90), det = +1

R_droid = quat_wxyz_to_matrix(box_camera.quat_wxyz)      # cols (w, h, l)
R_cam_pre_wrist = R_droid @ P                            # cols (l, h, w)
dim_pre = [w, h, l] = box_camera.size                     # vis4d [W, H, L]
```

After `P`, dimension `[w, h, l]` directly equals vis4d `[W, H, L]` —
the column relabel and axis relabel cancel out.

### Cross-camera transform (ext1, ext2)

The same physical box is reused for ext views. Transform via world:

```python
# 1. Wrist-cam -> world (from raw box.json)
T_c2w_wrist  = box["wrist_T_c2w"]
center_world = T_c2w_wrist[:3,:3] @ center_wrist + T_c2w_wrist[:3,3]
R_w_box      = T_c2w_wrist[:3,:3] @ R_cam_pre_wrist

# 2. World -> ext-cam (from step2b)
T_w2c_ext    = step2b[ext_serial]["optimized_extrinsics"]
center_ext   = T_w2c_ext[:3,:3] @ center_world + T_w2c_ext[:3,3]
R_cam_pre_ext = T_w2c_ext[:3,:3] @ R_w_box

# 3. Gravity in ext cam (Franka world Z-up)
gravity_ext  = T_w2c_ext[:3,:3] @ [0, 0, -1]
```

Box geometry consistency (wrist `box_camera.corners` round-tripping via
`box_world` and `wrist_T_c2w`) is verified to **2.2e-16 m max diff** — i.e.
the cross-camera math is exact, no extra error from the transform.

### Gravity normalization (canonical rotation)

Done **per view** (gravity vector differs per camera). Same logic as the
other canonical datasets (CA1M / FP / 3EED / ITW v3 / Waymo).

```python
R_cam, dimensions = align_rotation_to_gravity(R_cam_pre, dim_pre, gravity_view)
```

`align_rotation_to_gravity` (verbatim copy from FP converter) does:

1. **Axis swap** — if local X or Z is closer to gravity than Y, swap into Y
   (and swap the matching dimension).
2. **Y flip** — ensure local Y points along gravity (not against).
3. **W <= L** — Ry(90) swap to enforce width <= length.

Note: after gravity norm, the same physical box can have **different**
`dimensions` arrangements across views, because gravity is in different
directions in each camera's frame. The 3D geometry is identical; only the
canonical labeling differs.

## Omni3D JSON Format

### Annotation fields

| Field | Type | How we fill it |
|-------|------|----------------|
| `center_cam` | `[x, y, z]` | View-specific (wrist / ext1 / ext2 frame) |
| `dimensions` | `[W, H, L]` | After per-view gravity norm |
| `R_cam` | 3x3 | After per-view canonical rotation |
| `bbox3D_cam` | 8x3 | `compute_box3d_corners(center, dimensions, R_cam)` (vis4d order) |
| `bbox2D_proj` | `[x1,y1,x2,y2]` | Pinhole projection of 3D corners (unclipped). `[-1,-1,-1,-1]` if behind camera. |
| `bbox2D_trunc` | `[x1,y1,x2,y2]` | `bbox2D_proj` clipped to image bounds. |
| `bbox2D_tight` | `[-1,-1,-1,-1]` | DROID has no human-annotated 2D box. |
| `behind_camera` | bool | `center_cam[2] <= 0` (more common for ext views than wrist) |
| `valid3D` | bool | `not behind_camera` |
| `truncation` | float | `max(0, 1 - area(trunc) / area(proj))`. -1 if behind. |
| `visibility` | -1 | not available |
| `lidar_pts` | -1 | not available |
| `segmentation_pts` | -1 | not available |
| `depth_error` | -1 | not available |
| `category_name` | str | `box.object_name.lower().strip()` (786 unique on full set) |
| `category_id` | int | sorted alphabetical, dynamically assigned |

`-1` sentinels make the corresponding `is_ignore()` checks pass-through.

### Output directory

```
data/droid/
├── annotations/
│   ├── DROID_train.json        (~22k images, 22k anns)
│   ├── DROID_val.json          (~2.4k images, 2.4k anns)
│   ├── DROID_train_class_map.json   (auto-generated by dataset class)
│   └── DROID_val_class_map.json
├── data/DROID/<eid>_<view>.jpg     (rectified left view from stereo MP4)
├── depth/<eid>_<view>.png          (per-view FoundationStereo depth, uint16 mm)
├── data.hdf5                        (after to_hdf5 packing, ~2.7 GB)
├── depth.hdf5                       (after to_hdf5 packing, ~7.6 GB)
└── cache/                           (pkl cache from CacheMappingMixin)
```

### Train/val split

`shard_id % 10 == 0 -> val`, else `train`. Roughly 9:1. All 3 views of one
episode go to the same split (no leakage between views of the same episode).

## Depth Map Format

- **Source**: `step2a_depth/<eid>_depth.h5["<view>/depth"]`, shape `(N_frames, H, W)`, uint16 mm.
- **Output**: `depth/<eid>_<view>.png`, uint16 mm, same resolution as the RGB.
- **Loader**: `depth = decode(png).astype(float32) / 1000.0`, then `depth[depth > max_depth] = 0`.
- **Dataset class**: `DROIDDataset(max_depth=20.0, depth_scale=1000.0)`.

`max_depth=20.0` matches CA-1M (other indoor dataset). Box center stats
(500 wrist samples): p99 = 1.12m, p100 = 5.59m, so all real boxes pass
the `is_ignore` filter. The 20m threshold is for **depth map clipping**
(`depth[depth > max] = 0`) — ext views see room-scale depths up to ~14m,
and 20m preserves that real signal while clipping FoundationStereo
outliers on far walls / sky / reflective surfaces.

## Categories (as of 2026-05-01)

- **786 unique** free-form `object_name` from 8,273 episodes (after VLM filter).
- Long tail: 497 categories with ≤ 2 samples (63%).
- Top: cup (1063×3 ≈ 3.2k), marker (873×3 ≈ 2.6k), bottle (374×3 ≈ 1.1k), bowl, spoon, lid, pen, can, pot, drawer, ...
- Many fine-grained variants per concept: 26 cup-variants, 28 block-variants, 27 bottle-variants.
- Handled CA-1M-style: dynamic `det_map` built from annotation JSON's `categories` list.

## COCO3D Dataloader Filtering (is_ignore)

Same -1 semantics as CA-1M / 3RScan:

| Check | DROID value | Effect |
|-------|------------|--------|
| `behind_camera` | False (filtered before output) | passes |
| `valid3D` | True | passes |
| `lidar_pts == 0` | -1 | passes (-1 != 0) |
| `segmentation_pts == 0` | -1 | passes |
| `depth_error > 0.5` | -1 | passes (-1 < 0.5) |
| `truncation >= thres` | computed | active |
| `visibility <= thres` | -1 | passes (-1 < 0) |
| `dimensions[i] <= 0` | n/a | n/a |

Note: ext views may have boxes with very low 2D height (object far from
camera). `min_height_thres=0.0` in train (relaxed default) keeps these as
training signal; `=0.0625` in eval (strict default) drops them.

## Gotchas and Lessons Learned

1. **Two output trees!** Pipeline outputs are split: VLM lives in `droid/output/` while step2a/step2b/step3b1 live in `droid_pipeline/output/`. The README is in `droid_pipeline/` but references `output/step1_vlm_results/` — the actual VLM dir is one level up.
2. **Stereo MP4 left half only.** `<serial>-stereo.mp4` is `2W x H` side-by-side. Use `frame[:, :W//2]` for the rectified left view that pairs with `intrinsic` and `depth`.
3. **DROID box-local X<->Z swap.** Same as FoundationPose. Apply `P = [[0,0,1],[0,1,0],[-1,0,0]]` (Ry(90), det=+1) to the wrist-frame rotation.
4. **Gravity from per-frame wrist pose for wrist; from static optimized_extrinsics for ext.** Wrist camera is moving — must use `wrist_T_c2w` from box.json (which is `wrist_T_c2w` AT `best_frame`, not the static metadata snapshot). Ext cameras are static, gravity is fixed per episode.
5. **Franka world is Z-up.** `gravity_world = [0, 0, -1]`. Different from OpenGL/Isaac (where FoundationPose's gravity extraction needs Y/Z flip).
6. **Resolution may differ across cameras and episodes.** Wrist often 720x1280; ext1/ext2 could be 720x1280 or 376x672 (WVGA). Read shape from depth h5 per view, never hardcode.
7. **`object_name` lowercased.** "Cup" vs "cup" vs "  cup " are merged at conversion time.
8. **`box_world` not directly used.** We derive world-frame box from `box_camera + wrist_T_c2w` (machine-precision equivalent). The two paths agree to 2.2e-16 m.
9. **Per-view gravity norm = different `dimensions` per view.** Same physical box can be canonicalized differently in each camera frame — this is correct and expected.
10. **Filename suffix _<view>.** `<eid>_wrist.jpg`, `<eid>_ext1.jpg`, `<eid>_ext2.jpg`. Dataset class's `get_depth_filenames` `.replace()` chain preserves the suffix.
11. **Ext serials looked up from `metadata_*.json`.** Box JSON only has `wrist_serial`; ext serials come from per-episode metadata via `<ext_view>_cam_serial` keys.
12. **Step2b JSON's top-level keys mix metadata and serials.** `optimization_summary`, `episode_id`, `uuid`, `scene_path` are metadata; the rest are camera serials. Look up by serial directly, don't iterate top-level keys blindly.
13. **Visualization corner-ordering bug** (`visualize_droid_3cam.py` early version): if you draw raw DROID `box_camera.corners` with vis4d EDGES, lines cross. Either use DROID-ordered EDGES on raw corners OR (preferred) use vis4d-ordered corners reconstructed from `(center, dimensions, R_cam)` of the converted Omni3D JSON.
14. **Delete pkl cache after JSON change.** `data/droid/cache/DROID_*.pkl` is auto-generated by `CacheMappingMixin` — must delete when annotations change.

## Training integration

| File | Purpose |
|------|---------|
| `opendet3d/data/datasets/droid.py` | `DROIDDataset(COCO3DDataset)`, `get_droid_det_map`, `get_droid_class_map` |
| `opendet3d/zoo/gdino3d/base/dataset/droid.py` | `get_droid_train_cfg / test_cfg / dataset_cfg` (CA-1M-pattern) |
| `opendet3d/zoo/sam3_3d/sam3_3d_lingbot_depth_freeze21_omni3d_ca1m_waymo_3eed_fp_itw_v3_canonical_droid.py` | Full canonical training recipe with DROID at 2.5% sampling |
| `mmdetection_exp/training/sam3_3d/train_lingbot_alldata_canonical_droid_4node_0430.yaml` | Beaker job: 4 nodes 32 GPUs, finetune from alldata canonical 12e ckpt, 8 epochs, lr=1e-4 (config default, no CLI override) |

## TODO

- [x] Smoke-test on 5-10 episodes, inspect viz
- [x] Full 3-view conversion (24,469 frames)
- [x] HDF5 packing (data.hdf5 2.7GB, depth.hdf5 7.6GB)
- [x] Wire DROID into training config and beaker yaml
- [ ] Submit beaker training job
- [ ] Eval on InTheWild after training; compare vs alldata-canonical baseline (no DROID)
- [ ] If positive: ablate sampling ratio (2.5% → 5% → 10%)
