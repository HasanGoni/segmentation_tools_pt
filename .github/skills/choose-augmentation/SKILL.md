---
name: choose-augmentation
description: 'Choose augmentation settings for segmentation_tools_pt and — critically — tell the user which augment knobs are actually wired vs. config-only placeholders. Use when a user asks "how much augmentation?", "should I flip?", "enable mosaic/mixup", or is editing the augment.* block. Emits the augment.* block for train.yaml with honest status notes.'
argument-hint: '[optional] is the imaging orientation fixed?'
user-invocable: true
disable-model-invocation: false
---
# Choose Augmentation

Set the `augment.*` block — and be honest about what the training pipeline currently applies.

## ⚠️ Implementation status (verify before promising anything)

**Actually applied today** (in `transforms.get_train_transforms`, wired from `scripts/train.py`):
- A fixed geometric + photometric stack: RandomCrop, small Affine (translate/scale/rotate),
  GridDistortion, RandomBrightnessContrast, GaussNoise, GaussianBlur.
- `boundary_blur` — config-driven via `augment.boundary_blur*` (`BlurBoundaryZone`).
- `flips` — optional h/v flips via the `augment.flips` bool.

**Config-only PLACEHOLDERS — present in `train.yaml` but NOT yet wired** (`scripts/train.py`
passes `aug=None`): `preset` (`yolo26n/s/m/l/x`) and the per-knob sample-level controls
`translate`, `scale`, `degrees`, `shear`, `perspective`, `hsv_*`,
`bgr`, `erasing`. Setting these has **no effect yet** — say so to the user. They would be wired
in `nbs/05_transforms.ipynb` (an `AugmentConfig` → transform builder).

**Batch-level augs ARE wired** (issue #17; `nbs/10_train.ipynb` → `train_one_epoch`): master
switch `augment.batch.enabled` (default **false**) + per-type probs `mosaic`, `mixup`, `cutmix`,
`copy_paste` (default **0**) + `copy_paste_mode: flip`. YOLO-style **pick-one** — at most one type
per batch; each type fires with exactly its configured marginal probability; master off or
all-zero probs = training identical to no batch aug. Emit, e.g.:

```yaml
augment:
  batch:
    enabled: true
    mosaic: 0.10     # per-batch marginal probs; keep small (0.05-0.15) for inspection data
    mixup: 0.05
    cutmix: 0.0
    copy_paste: 0.10
    copy_paste_mode: flip
```

Notes: batch augs handle binary `(B,1,H,W)`/`(B,H,W)` and multilabel `(B,K,H,W)` masks directly;
on multiclass integer masks `mixup` blends then truncates labels — prefer `cutmix`/`copy_paste`
there. All randomness is seeded (`set_seed`) so runs stay reproducible.

## Decision matrix (for what IS active)

| Situation | Recommendation |
|---|---|
| Fixed-camera / orientation-sensitive X-ray | `flips: false` (default) — flips would create impossible geometries |
| Truly symmetric layout | `flips: true` |
| Ragged mask borders / want softer edges | Enable `boundary_blur` (e.g. `0.4`) with `boundary_blur_width/sigma` |
| Clean, plentiful data | Rely on the built-in stack; keep boundary_blur low |

## What this skill emits

- The `augment:` block for `configs/train.yaml`, with the active keys set and a comment marking
  placeholder keys as "not implemented yet".

## Library anchors

- `src/segmentation_tools_pt/transforms.py` — `get_train_transforms`, `get_val_transforms`,
  `BlurBoundaryZone`, `boundary_blur_kwargs_from_cfg`.
- Source notebook: `nbs/05_transforms.ipynb` (sample-level preset system), `nbs/10_train.ipynb`
  (batch augs: `pick_batch_aug`, `apply_batch_augs`), `nbs/15_hydra_pipeline.ipynb`
  (`resolve_batch_aug_cfg`).
