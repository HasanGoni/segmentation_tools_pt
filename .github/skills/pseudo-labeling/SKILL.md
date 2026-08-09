---
name: pseudo-labeling
description: 'PLACEHOLDER skill — pseudo-labeling / self-training is NOT implemented in segmentation_tools_pt yet. Use when a user asks "label unlabeled images with my model", "self-training", "semi-supervised segmentation", or "use model predictions as labels". Explains the workflow and where it would be added.'
argument-hint: '[optional] how much unlabeled data you have'
user-invocable: true
disable-model-invocation: false
---
# Pseudo-Labeling / Self-Training — NOT IMPLEMENTED YET

Use a trained model to generate soft or hard labels for unlabeled images, then add the confident
ones to the training set. This is **not implemented** in `segmentation_tools_pt` today.

## Typical workflow

1. Train an initial model on labeled data.
2. Run inference on unlabeled images.
3. Keep only predictions above a confidence threshold (e.g. probability > 0.9).
4. Optionally apply strong augmentations and enforce consistency.
5. Fine-tune the model on labeled + pseudo-labeled data.

## Where it would be added

- New notebook: `nbs/17_pseudo_labeling.ipynb` → `pseudo_labeling.py`.
- Functions: `generate_pseudo_labels(model, unlabeled_loader, threshold)`,
  `filter_pseudo_labels(probs, threshold, min_area)`.
- Hook into `dataset_build.py` so pseudo-labeled images can be added to a patch build.

## What the user can do today

- Use `inference.predict()` to generate probability maps, then threshold and save masks manually.
- Use the `segmentation-error-analysis` skill to find which pseudo-labels are trustworthy.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
