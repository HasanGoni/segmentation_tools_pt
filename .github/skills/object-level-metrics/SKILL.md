---
name: object-level-metrics
description: 'Use object-level detection metrics in segmentation_tools_pt to evaluate whether each defect/instance is found, rather than only pixel overlap. Use when a user asks "IoU looks good but I still miss whole defects", "per-object precision/recall", "detection rate", or is choosing overlap_thresh. Explains ObjectMetrics vs SegMetrics and when to optimize for detection rate.'
argument-hint: '[optional] defect size and acceptable overlap'
user-invocable: true
disable-model-invocation: false
---
# Object-Level Metrics

Pixel metrics (`iou`, `dice`) can be misleading for sparse, small defects: a prediction that barely
touches many objects can yield decent IoU while still missing entire defects. Object-level metrics
fix this by counting connected-component hits and misses.

## How it works

`metrics.object_metrics()` runs `cv2.connectedComponents` on the predicted and ground-truth masks,
then counts a ground-truth object as **detected** if at least `overlap_thresh` fraction of its area
overlaps any predicted foreground.

```python
from segmentation_tools_pt.metrics import object_metrics, ObjectMetrics

# one image
r = object_metrics(pred_mask, gt_mask, overlap_thresh=0.5)
# r = {'total_gt': 12, 'detected': 10, 'missed': 2, 'false_alarms': 1, 'detection_rate': 0.833}

# accumulate across a dataset
om = ObjectMetrics(overlap_thresh=0.5)
for p, g in zip(pred_masks, gt_masks):
    om.update(p, g)
om.compute()
```

## Choosing `overlap_thresh`

| Use case | overlap_thresh |
|---|---|
| Loose "any overlap counts" | 0.25 |
| Standard instance-detection definition | 0.5 |
| Strict, require majority overlap | 0.75 |

## When to prefer object metrics

- Sparse small defects (one missed blob is catastrophic).
- Defect counting / pass-fail inspection.
- Comparing models with similar IoU but different detection behaviour.

`find_operating_threshold` already reports `detection_rate` and `false_alarms` alongside pixel
metrics — see the `segmentation-error-analysis` skill.

## Library anchors

- `src/segmentation_tools_pt/metrics.py` — `object_metrics`, `ObjectMetrics`.
- Used by `src/segmentation_tools_pt/inference.py` — `find_operating_threshold`.
- Source notebook: `nbs/08_metrics.ipynb`.
