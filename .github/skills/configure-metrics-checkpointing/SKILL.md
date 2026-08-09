---
name: configure-metrics-checkpointing
description: 'Choose the validation metric to optimise and the sigmoid threshold for segmentation_tools_pt, controlling which checkpoint becomes best.pth. Use when a user asks "optimise for recall vs IoU?", "what threshold?", or is editing the metrics.* block. Emits the metrics.* block for train.yaml.'
argument-hint: '[optional] is recall (miss nothing) or precision more important?'
user-invocable: true
disable-model-invocation: false
---
# Configure Metrics & Checkpointing

Pick the metric that selects `best.pth` and the decision threshold.

## Keys

- `metrics.best_metric` — one of `iou` | `dice` | `recall` | `precision`. The epoch that maximises
  this metric is saved as `best.pth`.
- `metrics.threshold` — sigmoid probability cutoff for binarising predictions (default `0.5`).

## Decision matrix

| Priority | best_metric |
|---|---|
| Balanced overlap (default) | `iou` |
| Overlap, softer than IoU | `dice` |
| Miss no defects (safety-critical) | `recall` (pair with high `loss.alpha`) |
| Minimise false alarms | `precision` |

Tuning the operating threshold **after** training (on val) is a separate step — see the run-inference
skill and `find_operating_threshold` in the inference tutorial.

## What this skill emits

- The `metrics:` block for `configs/train.yaml`; e.g.
  `segpt-train configs/train.yaml metrics.best_metric=recall metrics.threshold=0.4`.

## Library anchors

- `src/segmentation_tools_pt/metrics.py` — IoU/Dice/recall/precision, `MetricsTracker`, `ObjectMetrics`.
- Source notebook: `nbs/08_metrics.ipynb`.
