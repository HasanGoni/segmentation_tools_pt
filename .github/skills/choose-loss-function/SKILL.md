---
name: choose-loss-function
description: 'Recommend a segmentation loss combination and phase schedule for segmentation_tools_pt based on the problem (class imbalance, foreground size, boundary importance). Use when a user asks "which loss should I use?", "my defects are tiny/sparse", "the borders are wrong", or wants a schedule like "combine Tversky+Dice early, then add boundary loss later". Emits the loss.* block for train.yaml.'
argument-hint: '[optional] defect sparsity and whether boundaries matter'
user-invocable: true
disable-model-invocation: false
---
# Choose Loss Function (+ phase schedule)

Pick and weight the loss terms in `losses.SegLoss`, then schedule how the weights change across
epochs via `loss.phases`. Works for binary (`num_classes: 1`, sigmoid) and multi-class (softmax).

## The four components (all combined inside `SegLoss`)

| Term | Config weight | Use when |
|---|---|---|
| `TverskyFocalLoss` (`alpha`,`beta`,`gamma`) | `w_tversky` | Sparse/imbalanced foreground; tune recall vs precision |
| `DiceLoss` | `w_dice` | General region overlap; stable default partner to Tversky |
| `BoundaryLoss` | `w_boundary` | Edges/borders matter; add **late** to sharpen |
| `BCELoss` | `w_bce` | Large, dense foreground (chip fills frame) — imbalance correction unnecessary |

`alpha` = false-negative weight (**raise to miss fewer defects → higher recall**),
`beta` = false-positive weight (raise to reduce false alarms), `gamma` = focal exponent.

## Decision matrix

| Problem | Recommendation |
|---|---|
| Tiny, sparse defects | High `w_tversky`, `alpha≈0.7 beta≈0.3`; add `w_boundary` in later phases |
| Large region / chip fills frame | `w_bce` + `w_dice`; drop or lower Tversky |
| Ragged/incorrect borders | Introduce `w_boundary` after the region terms have converged |
| Balanced, simple | `w_dice` alone or Dice+BCE |

## Phase schedule (the "combine early, add boundary later" pattern)

Coarse region learning first, boundary sharpening once regions are roughly right. Each phase applies
until `until_epoch` (`-1` = to the end):

```yaml
loss:
  alpha: 0.7
  beta: 0.3
  gamma: 1.33
  phases:
    - {until_epoch: 10, w_tversky: 0.6, w_dice: 0.4, w_boundary: 0.0}
    - {until_epoch: 35, w_tversky: 0.5, w_dice: 0.3, w_boundary: 0.2}
    - {until_epoch: -1, w_tversky: 0.4, w_dice: 0.3, w_boundary: 0.3}
```

Add `w_bce` to any phase dict when using BCE; phases without a key default that weight to 0.

### Hierarchy term (multilabel, e.g. `lifted ⊂ chip`)

Not a loss *choice* but an add-on: set `train.hierarchy.<sub>_subset_of_<super>: true` plus `train.hierarchy.weight: 0.5`
(or `w_hierarchy` inside a phase dict). Default `0` = off. It only works with `train.task: multilabel` and fails early
otherwise. To check it helps, A/B `train.hierarchy.weight=0.0` vs `0.5` on the same split/seed and compare the
violation rate (`metrics.subset_violation_rate`) on raw predictions; see `tutorials/06_hierarchy_loss.ipynb`.

## What this skill emits

- The `loss:` block for `configs/train.yaml`, matched to the interview.
- Optional CLI override form, e.g. `segpt-train configs/train.yaml loss.alpha=0.75`.

## Library anchors

- `src/segmentation_tools_pt/losses.py` — `SegLoss`, `TverskyFocalLoss`, `DiceLoss`,
  `BoundaryLoss`, `BCELoss`, `resolve_loss_weights` (5-tuple; `w_bce` / `w_hierarchy` optional per phase, default 0),
  `validate_hierarchy_loss`.
- Source notebook: `nbs/07_losses.ipynb`.
