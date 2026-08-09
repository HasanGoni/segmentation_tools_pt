---
name: classical-segmentation-baseline
description: 'PLACEHOLDER skill — classical (non-deep-learning) segmentation baselines (Otsu/adaptive thresholding, watershed, edge-based, morphology) are NOT implemented in segmentation_tools_pt yet. Use when a user asks for a "quick classical baseline", "threshold/Otsu/watershed segmentation", or "no-training segmentation". Explains what is missing and where it would be added.'
argument-hint: '[optional] the classical method requested'
user-invocable: true
disable-model-invocation: false
---
# Classical Segmentation Baseline — NOT IMPLEMENTED YET (placeholder)

The classical-CV panel flags this clearly: **`segmentation_tools_pt` is a deep-learning library.**
Training-free classical segmentation baselines are **not implemented**. Do not pretend they exist.

## What is NOT available (would need to be built)

- Global/adaptive thresholding (Otsu, Sauvola), watershed, region growing, active contours.
- Edge-based segmentation (Canny + close), connected-component + morphology pipelines.
- A `classical_segment(image, method=...)` API.

## What DOES exist nearby (so you can still help)

- Feature maps: tophat + sobel via `data.data_mode: single_to_3` (see feature-channels skill).
- Mask geometry: `masks.fit_rotated_rects_mask`, `masks.binarize` (see mask-preprocessing skill).

## Recommendation to the user

- If they want a labeled baseline fast, use these as inputs to the DL pipeline (planner skill).
- If a true classical baseline is genuinely needed, it would be added as a new nbdev module
  (e.g. `nbs/16_classical.ipynb` → `classical.py`) exporting OpenCV/`scikit-image`-based functions,
  then optionally surfaced via a `segpt-classical` CLI. Offer to scaffold that as a separate task —
  but only if the user explicitly asks; do not silently add it.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
