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

## Multilabel decode chain (threshold -> class morph -> hierarchy)

For multilabel heads, use the fixed decode order; the optional class-targeted morph
(issue #19) is config-gated (`postprocess.class_morph`, master switch default **false**)
and cleans ONE named class channel with opening + min-area — never applied to the loss
or to raw train metrics:

```python
from segmentation_tools_pt.hydra_pipeline import resolve_class_morph_cfg
from segmentation_tools_pt.inference import decode, apply_class_morph, apply_hierarchy

morph_cfg = resolve_class_morph_cfg(cfg, class_names)   # None when disabled -> strict no-op
masks = decode(prob, threshold=thresholds, task='multilabel')
if morph_cfg is not None:
    masks = apply_class_morph(masks, **morph_cfg)        # class_idx / kernel_size / min_area
masks = apply_hierarchy(masks, pairs=hierarchy_pairs)    # e.g. [(1, 0), (2, 0)] = void/lifted ⊆ chip
```

`class_name` must exist in `data.class_names` or `resolve_class_morph_cfg` raises a clear
ValueError; other channels pass through `apply_class_morph` untouched.

## Batched tile inference for very large images (`predict_tiled`, issue #20)

`predict` runs one view at a time — for full inspection images use `predict_tiled`:
tiles are forwarded in chunks and stitched with Gaussian-weighted blending, with the
same activation semantics as `predict`. Batch size is a pure memory ↔ throughput knob
(batch 1 is bit-identical to any larger batch):

```python
from segmentation_tools_pt.hydra_pipeline import resolve_inference_cfg
from segmentation_tools_pt.inference import predict_tiled

knobs = resolve_inference_cfg(cfg)   # {'tile_size': 256, 'overlap': 32,
                                     #  'inference_batch_size': 8, 'prefetch_factor': None}
prob = predict_tiled(model, image, device, **knobs)   # (C, H, W) probabilities
mask = (prob[0] >= 0.5).long()
```

Knobs live in the `inference:` config block (`tile_size`, `overlap`,
`inference_batch_size`, `prefetch_factor`); `prefetch_factor` targets loader-driven
tiled-inference consumers — `predict_tiled` batches tiles in-process and ignores it.
Images smaller than one tile are zero-padded up; clamped tile starts cover every pixel.

## Instance-model output (RF-DETR) as a semantic map

`detections_to_semantic(dets, class_ids=[0, 1], first_class_id=0)` (RF-DETR's 2-class model returned 0-based ids; check `set(dets.class_id)` and pass `first_class_id` to match) paints `.mask (N,H,W)`, `.class_id`, `.confidence`
(plain arrays or a `supervision.Detections`) into an `(H, W)` long map; `output='stack'` gives
`(K, H, W)` best scores per class; `threshold=` drops weak detections; the higher score wins overlaps.
An RF-DETR **run** (`model.source: rfdetr`, trained by `segpt-train`) loads with the library:

```python
from segmentation_tools_pt.rfdetr import load_rfdetr_predictor   # lazy `import rfdetr`; install hint if missing
pred = load_rfdetr_predictor('outputs/my_rfdetr_run')            # checkpoints/last_ema.pth, class count from .hydra/config.yaml
score_map = pred(image_hw, threshold=0.2)                        # (K, H, W) float, > 0 is the prediction; image is the raw cache patch
dets = pred.detections(image_hw, 0.01); pred.stack(dets, image_hw.shape[:2], 0.3)   # predict once, cut at many thresholds
```

Large images: `predict_tiled_rfdetr(pred, image_chw, 'cuda', threshold=0.2, **resolve_inference_cfg(cfg))` runs `predict_tiled` with RF-DETR (raw 0-255 pixels in, `(K, H, W)` scores out; `> 0` is the union over tiles). **Tile at the training patch size** (`train.crop_size`; the default when `tile_size` is left out). On a real run, tiles of the training size matched whole-patch prediction exactly (lifted IoU 0.985 vs 0.985), while tiles of half size dropped some patches from 0.98 to 0.0, so `inference.tile_size: 256` from `configs/train.yaml` is wrong for a 512-patch RF-DETR run and the function warns.

`segpt-eval RUN... --report-dir DIR --ids ids.json` builds the review report for RF-DETR runs like for U-Net runs.
RF-DETR scores are not probabilities (useful thresholds 0.1 to 0.3 in #106), so do not reuse the U-Net operating point.
See `choose-model-architecture`, `model-comparison-ab-test` and `tutorials/09_rfdetr_bridge.ipynb`.

## Library anchors

- `src/segmentation_tools_pt/inference.py` — `predict` (sliding-window + blending),
  `predict_tiled` (batched tiles) and `tile_starts`.
- `src/segmentation_tools_pt/visualize.py` — `overlay_mask` for inspecting results.
- Source notebook: `nbs/11_inference.ipynb`.
