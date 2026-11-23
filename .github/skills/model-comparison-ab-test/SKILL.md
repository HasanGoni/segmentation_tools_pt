---
name: model-comparison-ab-test
description: 'Compare finished segmentation runs on one scale with `segpt-eval` (shared IoU + object metrics, threshold sweep, held-out threshold choice, seed arms, val-id fingerprint check). Use when a user asks "is model A better than model B?", "compare two runs/checkpoints", "which seed/encoder is better", "held-out threshold", or "A/B test segmentation". Paired bootstrap / McNemar significance tests are NOT implemented yet.'
argument-hint: '[optional] two or more run directories'
user-invocable: true
disable-model-invocation: false
---
# Model Comparison / A/B Testing

Declaring "model A is better than model B" needs both models scored by **the same code on the same
patches**, with a threshold that was not tuned on the numbers being reported, and with the seed noise
in view. `segpt-eval` (module `segmentation_tools_pt.eval`, notebook `nbs/17_eval.ipynb`) does that.

## Quick recipe

```bash
# one row per run, threshold 0.5
segpt-eval outputs/a_s42 outputs/b_s42

# seeds as arms, threshold chosen on one half of the source images and scored on the other
segpt-eval outputs/a_s42 outputs/a_s43 outputs/b_s42 outputs/b_s43 --heldout --out eval_out
```

- A run = directory with `.hydra/config.yaml` + `checkpoints/best.pth`; patch mode only (`data.patch_mode: true`).
- Directories that differ only by a trailing `_s<seed>` are one **arm**: table shows mean `+-` spread.
- `--heldout`: threshold picked on half A (best object F1 of `--objects-class`, default the last class), scored on half B, and vice versa; halves are split by `source_id` so neighbouring patches never leak.
- The command **fails (exit 1)** if the runs' val-id fingerprints or class names differ. Do not work around it: it means the runs saw different patches (different patch cache or seed for `val_fraction`).
- Outputs in `--out`: `<run>.json`, `<run>_per_patch.json`, `comparison.json`. `--report-dir DIR --ids ids.json` also builds a review report (see `build-review-report`).

## Reading the table

`F1_<class>` is object-level F1 = `2*detected / (2*detected + missed + false alarms)` on the objects class; IoU columns are
per-class mean patch IoU. A gain smaller than the seed spread (`+-`) is noise: the #106 rule asks for a gain above
`max(0.01, 2 * spread)` plus no loss on the first class's IoU (`segmentation_tools_pt.eval.decide`).

## Python

```python
from segmentation_tools_pt.eval import evaluate_run, compare_runs, format_table
res = [evaluate_run(r, heldout=True) for r in runs]
print(format_table(compare_runs(res)))
```

Pure pieces (`held_out`, `arm_summary`, `decide`, `score_dataset`) take plain dicts/arrays and are
testable without a run. Verifying a real comparison: `tutorials/10_segpt_eval.ipynb`.

## Not implemented

Paired bootstrap, McNemar's test and an object-level sign test (p-values / confidence intervals on
the metric difference) are still missing; the seed spread plus `decide` is the noise guard today.
Where they would go: a new function in `nbs/17_eval.ipynb` (`bootstrap_metric_diff`, `mcnemar_test`) over
the `<run>_per_patch.json` rows.
Related skills: `object-level-metrics`, `build-review-report`, `run-inference`.
