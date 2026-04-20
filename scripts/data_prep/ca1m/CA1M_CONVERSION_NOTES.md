# CA-1M (Cubify Anything) to Omni3D Conversion

Convert Apple's [CA-1M](https://github.com/apple/ml-cubifyanything) tar archives into the Omni3D JSON layout expected by WildDet3D. The conversion is deterministic — given the same `--frame_interval`, it produces the same images, depth, and annotations.

## Source data

Download the **train** split (3,118 tars, ~1.7 TB) from [apple/ml-cubifyanything](https://github.com/apple/ml-cubifyanything) into a root of your choice (we'll call it `<CA1M_ROOT>`). Each tar is one video (~430 frames at full frame rate, ~366 MB). Only the train split is used for WildDet3D training.

## Convert

```bash
# Quick sanity check (3 tars, ~1 min). Swap --split val if you only want to verify setup.
python scripts/data_prep/ca1m/convert_ca1m_to_omni3d.py \
    --ca1m_root <CA1M_ROOT> \
    --output_dir data/cubifyanything \
    --split train --max_tars 3

# Full train (3118 tars, --num_workers 8 recommended)
python scripts/data_prep/ca1m/convert_ca1m_to_omni3d.py \
    --ca1m_root <CA1M_ROOT> \
    --output_dir data/cubifyanything \
    --split train --num_workers 8
```

Defaults: `--frame_interval 10` (every 10th frame). Increase workers if your storage can keep up.

## HDF5 pack (required for training)

```bash
python -m vis4d.data.io.to_hdf5 -p data/cubifyanything/data
python -m vis4d.data.io.to_hdf5 -p data/cubifyanything/depth_gt
```

## Output layout

```
data/cubifyanything/
├── annotations/
│   ├── CubifyAnything_train.json       # structural labels (wall/floor/...) filtered
│   ├── CubifyAnything_val.json
│   ├── CubifyAnything_full_train.json  # all labels kept
│   └── CubifyAnything_full_val.json
├── data.hdf5                            # RGB (JPEG, 1024x768)
└── depth_gt.hdf5                        # GT laser scanner depth (uint16 mm, 512x384)
```

Depth is stored as uint16 in millimetres (`depth_meters * 1000`). 0 means "no data" (reflective surfaces, thin structures, registration failures).

Stage 2 training config picks up the two HDF5 files automatically via `HDF5Backend`.
