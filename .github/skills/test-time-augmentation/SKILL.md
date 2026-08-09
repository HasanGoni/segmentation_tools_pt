---
name: test-time-augmentation
description: 'Use test-time augmentation (TTA) in segmentation_tools_pt to improve prediction robustness by averaging 90-degree rotations of the input. Use when a user asks "TTA", "test-time augmentation", "average rotated predictions", or wants a quick boost at inference without retraining. Provides the predict(tta=True) call and caveats.'
argument-hint: '[optional] whether the task is orientation-sensitive'
user-invocable: true
disable-model-invocation: false
---
# Test-Time Augmentation (TTA)

Average predictions over input rotations at inference time for a small, training-free robustness
boost. Implemented in `inference.predict(tta=True)`.

## How it works

```python
from segmentation_tools_pt.inference import predict
prob = predict(model, image, device, tta=True, n_augs=4, activate=True)
```

- `n_augs=4` averages 0°, 90°, 180°, 270° rotations.
- Each rotated prediction is rotated back before averaging.
- Output is the same shape as without TTA: `(C, H, W)` probabilities.

## When to use

- You want a quick inference-time boost without retraining.
- The imaging is **not** strongly orientation-dependent (e.g. top-down X-ray of a symmetric region).
- Latency budget allows ~4× forward passes.

## When NOT to use

- Fixed-camera geometry where a 90° rotation is physically impossible (TTA will produce unrealistic
  inputs and can hurt). In that case, prefer stronger training augmentation or ensemble training.
- Latency-constrained production pipelines.

## What this skill emits

- The `predict(..., tta=True)` call with `decode()` to get the final mask.
- A note on orientation sensitivity.

## Library anchors

- `src/segmentation_tools_pt/inference.py` — `predict`.
- Source notebook: `nbs/11_inference.ipynb`.
