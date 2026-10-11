---
name: choose-tiling-strategy
description: 'Decide whether to tile (patch mode) or train on whole images (direct mode) in segmentation_tools_pt, and pick patch_size / stride / crop_size. Use when a user asks "do I need tiling?", "should I use patches?", "my images are huge / defects are tiny", or is configuring data.patch_mode / patch.* / train.crop_size. Emits the data + patch block for build.yaml / train.yaml.'
argument-hint: '[optional] raw image size and defect size'
user-invocable: true
disable-model-invocation: false
---
# Choose Tiling Strategy (patch vs direct)

Decide how images enter the model: **patch mode** (pre-extract a reusable cache of small tiles) or
**direct mode** (load whole images, crop on the fly). Then size the tiles.

## When to TILE (patch mode — `data.patch_mode: true`, the default)

- Raw images are **large** (X-ray panels, full wafer maps) and don't fit in GPU memory at native size.
- Defects are **small / sparse** — tiling + foreground filtering concentrates learning on real signal
  and enables hard-example oversampling (`patch.min_fg_pixels`, `patch.max_empty_frac`).
- You will run **many experiments** on the same data and want a build-once, reuse-many patch cache.
- You are on a **shared cluster** — a local patch cache copied to node `/tmp` keeps GPU utilisation
  above the watchdog threshold when submitting jobs with `segpt-submit`.

## When NOT to tile (direct mode — `data.patch_mode: false`)

- Images are **already small** (roughly ≤ the crop size) after any coarse cropping.
- You want **fast prototyping** without a separate build step.
- Foreground is **large and dense** (a chip fills most of the frame) — a random crop already sees it.

## Decision matrix

| Situation | Mode | Why |
|---|---|---|
| 4k×4k panel, tiny cracks | patch | Memory + sparse-foreground oversampling |
| 256×256 crops already extracted upstream | direct | Nothing to gain from a cache |
| Millions of patches, HPC | patch (pre-stage on scratch) | Per-job `/tmp` copy too slow — see hpc/README |
| Quick "does this even train?" check | direct | Skip the build step |

## Default naming convention

For `discover_samples()` to pair images with masks correctly, the default layout requires **identical file names** in `raw_dir/images/` and `raw_dir/masks/`:

```text
raw_dir/images/sample_001.png
raw_dir/masks/sample_001.png
```

Only use a different mask name (e.g. `sample_001_mask.png`) if you set `data.mask_suffix` to match. Do not invent `mask_` prefixes or `_ch0` channel tags unless you have configured `data.data_mode` (`two_channel` / `three_channel`) or `mask_suffix`; otherwise pairing will fail silently or produce empty datasets.

## Keys this skill sets

In `configs/data_patch.yaml` (shared by build + train):

- `data.patch_mode` — `true` (tile) or `false` (direct).
- `patch.patch_size` — size of the tile **saved on disk** (e.g. `632`). Reusable across runs.
- `patch.stride` — step between tile centres (e.g. `316` = 50 % overlap).
- `patch.min_fg_pixels` / `patch.max_empty_frac` — foreground filtering (drop near-empty tiles).
- `patch.val_fraction` — train/val split fraction.

In `configs/train.yaml`:

- `train.crop_size` — the **model-view** crop taken from each patch at train time.

### Hard rule

`train.crop_size` **must be ≤** `patch.patch_size` in patch mode. `scripts/train.py` fails early if
violated. Common expert setup: extract `patch_size: 512` tiles, train on `crop_size: 256` crops for
cheap train-time augmentation variety.

## What this skill emits

- The `data` + `patch` block for `build.yaml` (and thus `train.yaml` via the shared default), plus:
  - `segpt-build configs/build.yaml` to create the cache, then
  - `segpt-train configs/train.yaml` (patch mode) **or** `segpt-train configs/train.yaml data.patch_mode=false`.

## Library anchors

- `src/segmentation_tools_pt/patches.py` — `PatchConfig`, `grid_centers`, `extract_patches_from_sample`.
- `src/segmentation_tools_pt/dataset_build.py` — `build_patch_dataset` (discover → fit rects → patch → split → stats).
- `src/segmentation_tools_pt/dataset.py` — `SegmentationDataset` (patch) vs `DirectDataset` (direct).
- `src/segmentation_tools_pt/inference.py` — `predict_tiled` / `tile_starts` for **inference-time** tiling of full-resolution images (the only tiled-inference interface; the old `tiling*`/`stitching` cluster was retired in issue #27).
- Source notebooks: `nbs/03_patches.ipynb`, `nbs/04_dataset_build.ipynb`, `nbs/06_dataset.ipynb`.
