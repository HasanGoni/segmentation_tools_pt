---
name: training-monitoring-callbacks
description: 'Configure live training monitoring for segmentation_tools_pt (activation-stats histograms, validation prediction overlays) and custom TrainCallback hooks like early stopping. Use when a user asks "how do I watch training?", "dying features?", "save prediction overlays", "add early stopping", or is editing the monitoring.* block. Emits the monitoring.* block for train.yaml.'
argument-hint: '[optional] what do you want to monitor'
user-invocable: true
disable-model-invocation: false
---
# Training Monitoring & Callbacks

Two built-in callbacks are enabled by default via the `monitoring:` section, and you can add custom
`TrainCallback` hooks from a notebook.

## Built-in callbacks (`monitoring.*` in `configs/train.yaml`)

| Callback | Output | Purpose |
|---|---|---|
| `ActivationStatsCallback` | `checkpoints/activation_stats/epochNNN/...png` | fastai-style activation histograms; bright band at bin 0 = dying features |
| `PredOverlayCallback` | `checkpoints/predictions/epochNNN/val_batchN_{labels,pred,compare}.jpg` | YOLO-style GT vs prediction mosaics each epoch |
| `ErrorAnalysisCallback` (opt-in) | `checkpoints/error_analysis/{split}_top{k}[_class].jpg` + `top_ranked.json` | post-training worst-sample triage: 4-column posters (image \| GT α=0.1 \| pred \| FN/FP diff), one per split — per class head for multilabel |
| `EncoderDriftCheck` (opt-in) | stdout verdict at train end (no files) | answers "did the encoder weights change?": bit-identical digest check; per-layer cosine report on drift; raises when `expect_frozen=true` |

```yaml
monitoring:
  activation_stats: {enabled: true, every: 1, n_bins: 40, hist_range: [0, 10], max_layers: 6}
  pred_overlays:    {enabled: true, every: 1, max_val_batches: 3, jpeg_quality: 90}
  error_analysis:   {enabled: false, top_k: 12, rank_by: loss, splits: [train, val], alpha: 0.1, jpeg_quality: 90, per_class: true}
  encoder_drift_check: {enabled: false, expect_frozen: null}
```

Disable either with `monitoring.activation_stats.enabled=false` / `monitoring.pred_overlays.enabled=false`.
`error_analysis` defaults to **off** (existing runs unchanged); set `enabled: true` to get
worst-sample posters at train end — `rank_by: loss` ranks highest loss first, a metric name
(`iou`/`dice`/`recall`/`precision`) ranks lowest first.

`encoder_drift_check` defaults to **off** (existing runs unchanged). Set `enabled: true` to
verify the encoder weights after training: `expect_frozen: null` derives from `train.freeze_epochs`
(raises when `-1`, report-only otherwise); `true` always raises on drift (use with freeze-forever
DINOv3 runs — a frozen encoder must be bit-identical since AdamW never updates params without
grads); `false` is report-only (per-layer cosine similarity, worst layers first).

## Custom callbacks (from a notebook, not the CLI)

Subclass `TrainCallback`, override only the hooks you need, and pass `callbacks=[...]` to `train()`.
Set `state.stop_training = True` for early stopping. Hooks fire at train/epoch/batch/checkpoint
granularity; lower `TrainCallback.order` runs first.

```python
from segmentation_tools_pt.train import TrainCallback, train

class EarlyStopping(TrainCallback):
    def __init__(self, patience=10):
        self.patience, self.wait, self.best = patience, 0, 0.0
    def on_epoch_end(self, state):
        s = state.val_metrics[state.best_metric]
        if s > self.best: self.best, self.wait = s, 0
        else:
            self.wait += 1
            if self.wait >= self.patience: state.stop_training = True

history = train(..., callbacks=[EarlyStopping(patience=10)])
```

## What this skill emits

- The `monitoring:` block for `configs/train.yaml`, and (for custom hooks) a notebook snippet — the
  CLI wires the monitoring callbacks (including `ErrorAnalysisCallback` when
  `monitoring.error_analysis.enabled=true`) via `build_default_callbacks()`.

## Library anchors

- `src/segmentation_tools_pt/train.py` — `TrainState`, `TrainCallback`, `ActivationStatsCallback`,
  `PredOverlayCallback`, `ErrorAnalysisCallback`, `EncoderDriftCheck`, `build_default_callbacks`.
- Source notebooks: `nbs/10_train.ipynb`, `nbs/12_visualize.ipynb`.
