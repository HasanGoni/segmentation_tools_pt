---
name: multi-class-segmentation
description: 'Configure segmentation_tools_pt for multi-class segmentation (num_classes > 1, softmax). Use when a user asks "more than two classes", "multi-class labels", "softmax instead of sigmoid", or needs to set train.num_classes > 1. Emits data, model, loss, and metrics differences from binary mode.'
argument-hint: '[optional] number of classes and label format'
user-invocable: true
disable-model-invocation: false
---
# Multi-Class Segmentation

Configure the library when masks contain integer labels for more than one foreground class
(`train.num_classes > 1`). The library supports this throughout, but a few defaults change.

## Key differences from binary

| Aspect | Binary (`num_classes: 1`) | Multi-class (`num_classes: N`) |
|---|---|---|
| Model head | 1 output channel + sigmoid | `N` output channels + softmax |
| Loss | `BCEWithLogitsLoss` / Tversky-focal binary | `CrossEntropyLoss` / multi-class Tversky |
| Targets | `(B, H, W)` {0,1} or `(B,1,H,W)` | `(B, H, W)` integer labels `0..N-1` |
| Prediction decode | `(prob > threshold).long()` | `argmax` over channels |
| Metrics background | class 1 only | classes 0..N-1; mean ignores background 0 |

## Config changes

```yaml
train:
  num_classes: 3        # e.g. background=0, defect_A=1, defect_B=2
model:
  in_channels: 3        # must still match data_mode
loss:
  phases:
    - {until_epoch: -1, w_tversky: 0.5, w_dice: 0.5, w_boundary: 0.0}
metrics:
  threshold: 0.5        # only used for binary; multi-class uses argmax
```

## Mask requirements

- Masks must be single-channel integer PNGs with values `0..N-1`.
- Class 0 is treated as background and excluded from the mean metric aggregation.

## What this skill emits

- The exact `train.num_classes`, `model.in_channels`, and `loss`/`metrics` settings for multi-class.
- A reminder to verify label encoding before patch extraction.

## Library anchors

- `src/segmentation_tools_pt/core.py` — `SegConfig.is_binary`.
- `src/segmentation_tools_pt/losses.py` — `probs_and_targets` handles softmax path.
- `src/segmentation_tools_pt/metrics.py` — `SegMetrics` multi-class aggregation.
- `src/segmentation_tools_pt/inference.py` — `decode` uses `argmax` when `C > 1`.
- Source notebooks: `nbs/00_core.ipynb`, `nbs/07_losses.ipynb`, `nbs/08_metrics.ipynb`.
