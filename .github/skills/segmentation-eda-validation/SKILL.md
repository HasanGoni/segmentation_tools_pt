---
name: segmentation-eda-validation
description: 'Validate a segmentation dataset before training in segmentation_tools_pt: check image/mask alignment, inspect sample batches, and interpret patch-cache statistics. Use when a user asks "does my data look right?", "verify image/mask pairs", "inspect the patch cache", or before starting segpt-train. Provides notebook snippets using sanity_check, show_batch, and stats.json.'
argument-hint: '[optional] dataset path or patch_dir'
user-invocable: true
disable-model-invocation: false
---
# Segmentation EDA & Validation

Validate data **before** burning GPU hours. This skill focuses on the implemented helpers in
`visualize.py` and the artifacts produced by `segpt-build`.

## Checks to run

1. **Platform / environment sanity** — `core.platform_info()` confirms OS, Python, OpenCV, NumPy,
   PyTorch, CUDA. Catches silent cross-platform dtype/behaviour differences.
2. **Sample discovery** — `dataset_build.discover_samples()` returns the matched image/mask pairs.
   Verify the count and expected file naming.
3. **Alignment check** — `visualize.sanity_check(dataset, n=8)` plots `(image, mask)` pairs. Look for:
   - mis-registered masks
   - wrong `mask_suffix`
   - channel order surprises
4. **Model prediction sanity (after a quick train or random init)** — `visualize.show_batch(model,
   dataset, device, n=8)` overlays predictions on random validation samples.
5. **Patch-cache stats** — `stats.json` (written by `build_patch_dataset`) contains channel mean/std
   used by normalization. Inspect it to confirm the value range matches your sensor/bit depth.

## What this skill emits

- A short notebook snippet sequence for EDA.
- Guidance on whether to fix data or proceed to `segpt-train`.

## Library anchors

- `src/segmentation_tools_pt/core.py` — `platform_info`, `set_seed`.
- `src/segmentation_tools_pt/dataset_build.py` — `discover_samples`, `build_patch_dataset`.
- `src/segmentation_tools_pt/visualize.py` — `sanity_check`, `show_batch`.
- `src/segmentation_tools_pt/dataset.py` — `SegmentationDataset`, `DirectDataset`.
- Source notebooks: `nbs/00_core.ipynb`, `nbs/04_dataset_build.ipynb`, `nbs/06_dataset.ipynb`,
  `nbs/12_visualize.ipynb`.
