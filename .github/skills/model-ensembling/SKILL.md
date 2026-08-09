---
name: model-ensembling
description: 'Build prediction ensembles in segmentation_tools_pt by averaging multiple checkpoints or combining TTA. Use when a user asks "ensemble models", "average checkpoints", "boost inference accuracy", or "combine multiple trained models". Provides Python snippets using predict() and decode().'
argument-hint: '[optional] how many checkpoints to ensemble'
user-invocable: true
disable-model-invocation: false
---
# Model Ensembling

Improve robustness by averaging predictions from multiple models or checkpoints. No special
ensemble class is required — compose `predict()` calls and average the probability tensors.

## Single-model TTA ensemble (simplest)

Already covered by the `test-time-augmentation` skill:

```python
from segmentation_tools_pt.inference import predict
prob = predict(model, image, device, tta=True, n_augs=4)
```

## Multi-checkpoint probability average

```python
from pathlib import Path
import torch
from segmentation_tools_pt.inference import predict, decode
from segmentation_tools_pt.model import get_model

checkpoints = ['checkpoints/best.pth', 'checkpoints/periodic_ep099.pth']
model = get_model(architecture='unet', arch='mobilenet_v2', in_channels=3, num_classes=1)
model.eval()

preds = []
for ckpt in checkpoints:
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    preds.append(predict(model, image, device, tta=True))

ensemble_prob = torch.stack(preds).mean(0)
mask = decode(ensemble_prob, threshold=0.5)
```

## When to use

- You trained several seeds/epochs and want a cheap accuracy boost.
- You have classifiers with complementary failure modes.

## Caveats

- Averaging probabilities only works when all models share the same `num_classes` and `in_channels`.
- For multi-class, average softmax probabilities then `argmax`.
- Orientation-sensitive tasks: avoid TTA in the ensemble.

## What this skill emits

- The Python snippet above, adjusted for binary or multi-class.

## Library anchors

- `src/segmentation_tools_pt/inference.py` — `predict`, `decode`.
- `src/segmentation_tools_pt/model.py` — `get_model`.
- Related skill: `test-time-augmentation`.
