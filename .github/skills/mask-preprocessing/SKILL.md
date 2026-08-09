---
name: mask-preprocessing
description: 'Clean up ground-truth masks before patch extraction in segmentation_tools_pt — binarize and snap noisy blobs to rotated rectangles (fit_rects). Use when a user asks "clean my masks", "my annotations are noisy", "fit bounding boxes to blobs", or is setting data.fit_rects. Notes which morphology steps are NOT implemented yet.'
argument-hint: '[optional] mask format and defect geometry'
user-invocable: true
disable-model-invocation: false
---
# Mask Preprocessing (classical CV panel)

Prepare masks before the build step. Some steps exist; morphological cleanup does not yet.

## Implemented

- **Binarization** — `masks.binarize()` converts `{0,255}` or `{0,1}` PNG to float32 `[0,1]`.
- **Rotated-rect fitting** — `masks.fit_rotated_rects_mask()` replaces each connected blob with its
  minimum bounding rotated rectangle. Enable in the build via `data.fit_rects: true`. Use when manual
  annotations are ragged but the true defect is a clean rectangle-like shape.

```yaml
data:
  fit_rects: true   # snap each mask blob to its min-area rotated rectangle
```

## ⚠️ Not implemented yet (placeholder)

General morphological cleanup — opening/closing, small-object removal, hole filling, dilation to a
target margin — is **not** in the library. If the user needs it, say "not implemented yet" and note
it would be added in `nbs/02_masks.ipynb` (e.g. `clean_mask(open_radius, close_radius, min_area)`
built on OpenCV/`scipy.ndimage`), then exported and optionally wired into `data.*`.

## Library anchors

- `src/segmentation_tools_pt/masks.py` — `binarize`, `fit_rotated_rects_mask`.
- Source notebook: `nbs/02_masks.ipynb`.
