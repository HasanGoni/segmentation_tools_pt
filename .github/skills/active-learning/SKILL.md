---
name: active-learning
description: 'PLACEHOLDER skill — active learning for segmentation is NOT implemented in segmentation_tools_pt yet. Use when a user asks "which images should I label next?", "uncertainty sampling", "active learning", or "reduce annotation cost". Explains uncertainty-based selection and where it would be added.'
argument-hint: '[optional] annotation budget'
user-invocable: true
disable-model-invocation: false
---
# Active Learning — NOT IMPLEMENTED YET

Active learning selects the most informative unlabeled samples for human annotation, reducing
labeling cost. It is **not implemented** in `segmentation_tools_pt` today.

## Common strategies

| Strategy | What it selects |
|---|---|
| Uncertainty sampling | Samples with highest entropy / lowest max-probability |
| Query-by-committee | Samples where multiple models disagree most |
| Core-set | Diverse samples that best cover the feature space |

## Where it would be added

- New notebook: `nbs/18_active_learning.ipynb` → `active_learning.py`.
- Functions: `uncertainty_score(probs)`, `entropy_map(probs)`, `select_next_batch(...)`.
- Integration: after each training round, score the unlabeled pool and return a list of image IDs to
  annotate.

## What the user can do today

- Use `segmentation-error-analysis` to find the current worst-loss samples and label similar
  unlabeled examples.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
