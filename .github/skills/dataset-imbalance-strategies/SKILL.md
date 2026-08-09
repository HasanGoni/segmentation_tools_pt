---
name: dataset-imbalance-strategies
description: 'Handle severe class imbalance in segmentation_tools_pt through patch filtering, loss weights (alpha/beta), boundary loss, focal gamma, and hard-example emphasis. Use when a user asks "my defects are tiny", "class imbalance", "miss small defects", or "few foreground pixels". Emits loss + patch + augmentation config guidance.'
argument-hint: '[optional] foreground fraction and defect size'
user-invocable: true
disable-model-invocation: false
---
# Dataset Imbalance Strategies

Industrial segmentation is almost always imbalanced: defect pixels are rare. Use a stacked strategy
across data, loss, and monitoring.

## 1. Patch filtering (data level)

Concentrate the dataset on patches that actually contain signal:

```yaml
patch:
  min_fg_pixels: 50       # drop patches with almost no foreground
  max_empty_frac: 0.10    # drop patches where < 90 % is background
```

Lowering `max_empty_frac` / raising `min_fg_pixels` increases foreground density but risks losing
clean-negative examples the model needs to learn background texture.

## 2. Loss weights (model level)

- `loss.alpha` (false-negative weight): raise when missing defects is costly.
- `loss.beta` (false-positive weight): raise when false alarms are costly.
- `loss.gamma` (focal exponent): higher values focus the loss on hard pixels.

For very sparse defects, a common starting point is `alpha=0.7, beta=0.3, gamma=2.0`.

## 3. Boundary emphasis

If borders are the main error mode, use the phase schedule to introduce `w_boundary` only after
region terms converge (see `choose-loss-function`).

## 4. Active augmentation

- Keep orientation flips **off** if geometry is fixed.
- Use `boundary_blur` to teach the model that exact edge location is uncertain.

## 5. Object-level monitoring

Track `detection_rate` and `false_alarms` with `ObjectMetrics` — pixel IoU can hide whole missed
defects (see `object-level-metrics` and `segmentation-error-analysis`).

## What this skill emits

- Adjustments to `patch.*`, `loss.*`, `augment.*`, and `metrics.*` for imbalanced data.

## Library anchors

- `src/segmentation_tools_pt/losses.py` — `TverskyFocalLoss` (`alpha`, `beta`, `gamma`).
- `src/segmentation_tools_pt/patches.py` — `PatchConfig`, foreground filtering.
- `src/segmentation_tools_pt/metrics.py` — `ObjectMetrics`.
- Related skills: `choose-loss-function`, `choose-augmentation`, `choose-tiling-strategy`,
  `object-level-metrics`.
