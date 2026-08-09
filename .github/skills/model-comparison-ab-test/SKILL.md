---
name: model-comparison-ab-test
description: 'PLACEHOLDER skill — statistical comparison of two segmentation checkpoints (A/B test, paired bootstrap, McNemar) is NOT implemented in segmentation_tools_pt yet. Use when a user asks "is model A better than model B?", "statistical significance", "compare two checkpoints", or "A/B test segmentation". Explains approaches and where they would be added.'
argument-hint: '[optional] two checkpoint paths'
user-invocable: true
disable-model-invocation: false
---
# Model Comparison / A/B Testing — NOT IMPLEMENTED YET

Declaring "model A is better than model B" requires more than a single metric difference. Proper
statistical comparison is **not implemented** in `segmentation_tools_pt` today.

## Methods to add

| Method | Use case |
|---|---|
| Paired bootstrap | Confidence interval on metric difference over the validation set |
| McNemar's test | Whether one model's errors are a strict subset of the other's |
| Object-level sign test | Per-image detection-rate comparison |

## Where it would be added

- New notebook: `nbs/22_model_comparison.ipynb` → `model_comparison.py`.
- Functions: `bootstrap_metric_diff(preds_a, preds_b, gt, metric_fn, n_bootstrap=1000)`,
  `mcnemar_test(errors_a, errors_b)`.
- Output: p-value + confidence interval on metric difference.

## What the user can do today

- Run both models on the same validation set and compare IoU/Dice/detection_rate.
- Use `object-level-metrics` to compare per-object failure counts.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
