---
name: segmentation-error-analysis
description: 'Analyze model errors after training in segmentation_tools_pt: find the worst-loss patches, sweep the operating threshold, and compare pixel vs object-level trade-offs. Use when a user asks "where does my model fail?", "find worst predictions", "tune the decision threshold", or "inspect validation errors". Provides show_top_losses, compute_patch_losses, and find_operating_threshold snippets.'
argument-hint: '[optional] checkpoint path and dataset'
user-invocable: true
disable-model-invocation: false
---
# Segmentation Error Analysis

After training, the model's worst mistakes are more informative than its average metrics. Use these
implemented helpers to find and inspect them.

## Tools

### 0. Automatic worst-sample posters — `ErrorAnalysisCallback` (post-training)

Set `monitoring.error_analysis.enabled: true` (or pass `error_analysis=True` to
`build_default_callbacks`) and the trainer writes, at **train end**:

- `checkpoints/error_analysis/{train,val}_top{k}.jpg` — 4-column posters of the worst
  samples per split (image | GT overlay α=0.1 | pred overlay | FN green / FP red / TP
  magenta difference); multilabel gets one poster per class head (`per_class: true`)
- `checkpoints/error_analysis/top_ranked.json` — `sample_id`, `split`, `score`,
  `rank_by`, optional `class`

`rank_by: loss` = highest per-sample loss first; `rank_by: iou|dice|recall|precision`
= lowest pixel metric first. Ranking helpers are exported for notebook use:
`rank_samples` / `rank_samples_per_class` / `save_error_poster` (`visualize.py`).

### 1. Per-sample losses — `compute_patch_losses`

Returns `(idx, loss, sample_id)` sorted worst-first. Use to build a "top losses" table or feed into
`show_top_losses`.

### 2. Visual top-loss grid — `show_top_losses`

Overlays the `k` highest-loss validation patches. Common failure modes to look for:

- **Tiny defects missed** → raise `loss.alpha` (FN weight) or add more hard examples via oversampling.
- **False alarms on texture** → raise `loss.beta` (FP weight) or tighten augmentation.
- **Boundary errors** → introduce/increase `w_boundary` in later loss phases.

### 3. Operating-threshold sweep — `find_operating_threshold`

The default `metrics.threshold: 0.5` is rarely optimal. `inference.find_operating_threshold()` scans
thresholds and reports pixel recall/precision/IoU **plus** object detection rate and false alarms.

```python
from segmentation_tools_pt.inference import find_operating_threshold
result = find_operating_threshold(model, val_loader, device,
                                  target_recall=0.99, overlap_thresh=0.5)
print(result['recommended_threshold'], result['achieved_recall'])
```

## Decision matrix

| Symptom | Next action |
|---|---|
| High loss on small defects | Tversky `alpha` ↑, more foreground patches, boundary loss later |
| High loss on clean negatives | `beta` ↑, reduce augmentation noise |
| Borders dominate errors | Add `w_boundary` in later phases |
| Threshold sweep shows 0.3 beats 0.5 | Set `metrics.threshold` / inference threshold to 0.3 |

## What this skill emits

- A notebook snippet combining `show_top_losses` + `find_operating_threshold`.
- Recommendations tied back to the `choose-loss-function`, `choose-augmentation`, and
  `configure-metrics-checkpointing` skills.

## Library anchors

- `src/segmentation_tools_pt/train.py` — `ErrorAnalysisCallback`, `build_default_callbacks`.
- `src/segmentation_tools_pt/visualize.py` — `rank_samples`, `rank_samples_per_class`,
  `save_error_poster`, `compute_patch_losses`, `show_top_losses`, `show_batch`.
- `src/segmentation_tools_pt/inference.py` — `predict`, `decode`, `find_operating_threshold`.
- `src/segmentation_tools_pt/metrics.py` — `ObjectMetrics`.
- Source notebooks: `nbs/10_train.ipynb`, `nbs/11_inference.ipynb`, `nbs/12_visualize.ipynb`,
  `tutorials/03_inference_and_top_losses.ipynb`.
