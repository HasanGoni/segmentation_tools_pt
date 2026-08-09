---
name: feature-channels
description: 'Add hand-crafted feature channels (tophat, sobel) to grayscale inputs in segmentation_tools_pt via data_mode single_to_3. Use when a user asks "add edge/contrast channels", "feed sobel/tophat to the model", or wants classical feature priors. Notes which extra features are NOT implemented yet.'
argument-hint: '[optional] which feature priors you want'
user-invocable: true
disable-model-invocation: false
---
# Feature Channels (classical CV panel)

Augment a grayscale input with classical feature maps so the network gets contrast/edge priors.

## Implemented

- `data.data_mode: single_to_3` — stacks the raw grayscale PNG with a **tophat** (contrast) channel
  and a **sobel** (edge) channel into a 3-channel input. Set `model.in_channels: 3`.

```yaml
data:
  data_mode: single_to_3
model:
  in_channels: 3
```

## ⚠️ Not implemented yet (placeholder)

Other feature priors — Gabor, LBP, frangi/vesselness, morphological gradient, multi-scale tophat, or
a configurable feature stack — are **not** in the library. If requested, say "not implemented yet"
and note they would be added in `nbs/01_channels.ipynb` as new `data_mode` presets registered in
`hydra_pipeline.CHANNEL_PRESETS` (with the matching `in_channels`).

## Library anchors

- `src/segmentation_tools_pt/channels.py` — channel assembly (tophat/sobel for `single_to_3`).
- `src/segmentation_tools_pt/hydra_pipeline.py` — `CHANNEL_PRESETS`.
- Source notebook: `nbs/01_channels.ipynb`.
