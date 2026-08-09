---
name: model-interpretability
description: 'PLACEHOLDER skill — model interpretability (Grad-CAM, attention maps, feature visualization) is NOT implemented in segmentation_tools_pt yet. Use when a user asks "why did the model predict this?", "Grad-CAM", "saliency map", "attention visualization", or "explain model decisions". Explains options and where they would be added.'
argument-hint: '[optional] which interpretability method you want'
user-invocable: true
disable-model-invocation: false
---
# Model Interpretability — NOT IMPLEMENTED YET

Explaining where the model looks is valuable for debugging false alarms and building trust. None of
the standard methods are implemented in `segmentation_tools_pt` today.

## Methods to add

| Method | What it shows |
|---|---|
| Grad-CAM | Class-discriminative activation maps from the last encoder layer |
| Guided backprop / Integrated gradients | Pixel-level input importance |
| Attention rollout | For transformer backbones (if added later) |
| Feature-map montage | Intermediate encoder/decoder feature channels |

## Where it would be added

- New notebook: `nbs/20_interpretability.ipynb` → `interpretability.py`.
- Functions: `grad_cam(model, image, target_layer)`, `plot_feature_maps(model, image, layer)`.
- Hook into the overlay pipeline so interpretability maps can be saved next to predictions.

## What the user can do today

- Use `segmentation-error-analysis` to find *which* patches fail.
- Use `object-level-metrics` to quantify failure modes.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
