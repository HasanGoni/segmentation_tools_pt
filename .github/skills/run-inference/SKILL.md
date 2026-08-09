---
name: run-inference
description: 'Run sliding-window full-image inference with a trained segmentation_tools_pt checkpoint. Use when a user asks "how do I predict on a full image?", "run the model on new data", "get a probability heatmap / binary mask", or "tune the operating threshold". Provides the predict() call and threshold guidance.'
argument-hint: '[optional] checkpoint path and image size'
user-invocable: true
disable-model-invocation: false
---
# Run Inference (sliding-window, full image)

Predict on arbitrarily large images by tiling into overlapping patches and stitching with
Gaussian-weighted blending — no retraining, any image size.

## Usage

```python
from segmentation_tools_pt.model import get_model
from segmentation_tools_pt.inference import predict
import torch

model = get_model(architecture='unet', arch='mobilenet_v2', in_channels=3, num_classes=1)
model.load_state_dict(torch.load('checkpoints/best.pth', map_location='cpu')['model'])
model.eval()

prob = predict(model, image, ...)          # activate=True (default) -> sigmoid/softmax probabilities
logits = predict(model, image, activate=False)  # raw logits for ensembling / custom post-processing
mask = (prob > 0.5)                          # binarize at your operating threshold
```

## Notes

- `activate=False` returns raw logits — useful for ensembling or custom post-processing.
- Choose the threshold on validation data, not by eye — see `find_operating_threshold` and
  `show_top_losses` in `tutorials/03_inference_and_top_losses.ipynb`.
- Match `in_channels` / `data_mode` to how the model was trained (see choose-channels).

## Library anchors

- `src/segmentation_tools_pt/inference.py` — `predict` (sliding-window + blending).
- `src/segmentation_tools_pt/visualize.py` — `overlay_mask` for inspecting results.
- Source notebook: `nbs/11_inference.ipynb`.
