---
name: build-review-report
description: 'Build a single HTML review report in segmentation_tools_pt so a person can check a segmentation result by eye: ground truth vs prediction vs errors for the worst, random and best samples. Use when a user asks "show me what the model gets wrong", "make a report I can open in the browser", "let me review the predictions", "compare prediction and ground truth visually", or wants something to share without running a notebook. Provides make_dummy_samples, samples_from_dataset and build_report.'
argument-hint: '[optional] checkpoint, validation data path, class names'
user-invocable: true
disable-model-invocation: false
---
# Build a review report

A review report is **one HTML file** you open in a browser. No kernel, no notebook. For each shown
sample it overlays the ground truth (cyan outline), the prediction (orange outline) and the errors
(false positive red, false negative green), with check boxes to switch each layer on and off and a
table of Dice / IoU / precision / recall per class.

Masks are drawn as exact vector outlines (never JPEG). 16-bit images keep their raw values in a
linked full-resolution file.

## Try it in 30 seconds (no data needed)

```python
from segmentation_tools_pt.review_report import make_dummy_samples, build_report
print(build_report('demo', 'reports/demo', make_dummy_samples(12, task='multilabel', dtype='uint16')))
```

Open the printed `report.html`. Tasks: `binary`, `multiclass`, `multilabel`.

## On your own model and data

```python
import cv2, torch
from segmentation_tools_pt.review_report import samples_from_dataset, build_report

samples = samples_from_dataset(
    model, val_ds, device,
    class_names=['void'],                       # one name per class / channel
    task=None,                                  # 'binary' | 'multiclass' | 'multilabel'; None infers it
    raw_image_fn=lambda sid: cv2.imread(f'{raw_dir}/{sid}.png', cv2.IMREAD_UNCHANGED),   # keeps 16-bit values
)
path = build_report('T-123', 'reports/T-123', samples, seed=0,
                    cfg=dict(n_worst=6, n_random=6, n_best=3))
print(path)
```

Rules to get right:

- `class_names` for **multiclass** include the background at index 0 (label `i` is `class_names[i]`).
  For **multilabel** give one name per channel. For **binary** one name or none.
- Without `raw_image_fn` the page shows the model input (normalised, 8 bit). Give the hook whenever
  the images are 16 bit.
- Use one folder per ticket (`reports/<ticket_id>/`). `ticket_id` may contain letters, digits, `.`, `_`, `-`.
- The same `seed` always shows the same samples.

## How to read it

| You see | It means |
|---|---|
| **Worst** group, score near 0 | Predictions that miss or invent regions; start here. |
| Red areas | False positives: prediction where there is no ground truth. |
| Green areas | False negatives: ground truth the model missed. |
| Table row missing for a class | That class is absent in both ground truth and prediction for this sample. |
| `report.json` → `metrics` | Pixel counts pooled over **all** samples (`dice.<class>`, `iou.mean`), not only the ones shown. |

## Next step after reading

Use the findings to choose a change (see `segmentation-error-analysis` for the decision matrix),
change the config, retrain, and build the report again into the same ticket folder.

## Not here yet

Click-to-comment feedback, pass/fail gates, a written summary, the preprocessing and the
synthetic-vs-real sections are separate tickets (#51–#54, map #48).
