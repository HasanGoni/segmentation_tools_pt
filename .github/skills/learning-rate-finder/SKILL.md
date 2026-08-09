---
name: learning-rate-finder
description: 'PLACEHOLDER skill — a learning-rate range test is NOT implemented in segmentation_tools_pt yet. Use when a user asks "find the best learning rate", "LR range test", "LR finder", or "cyclical LR". Explains the fastai-style approach and where it would be added.'
argument-hint: '[optional] current LR range'
user-invocable: true
disable-model-invocation: false
---
# Learning-Rate Finder — NOT IMPLEMENTED YET

A fastai-style LR range test would help pick `encoder_lr` and `decoder_lr` without guessing. It is
**not implemented** in `segmentation_tools_pt` today.

## What would be added

1. A function `lr_find(model, train_loader, optimizer, ...)` that:
   - Runs one pseudo-epoch.
   - Increases LR exponentially after each batch.
   - Records loss vs LR.
   - Suggests the LR just before the loss explodes (the steepest downward slope).
2. A plot helper in `visualize.py`.
3. Optional CLI integration: `segpt-train configs/train.yaml train.lr_finder=true`.

## Where it would live

- New notebook: `nbs/16_lr_finder.ipynb` → `lr_finder.py`.
- Hook into `train.py` via a special `TrainCallback` that updates LR each batch and logs loss.

## What the user can do today

- Use a coarse manual grid: try `encoder_lr` in `[1e-5, 3e-5, 1e-4, 3e-4]` and `decoder_lr` 10× higher.
- Monitor `history.json` for divergence.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
