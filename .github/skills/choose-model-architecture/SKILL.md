---
name: choose-model-architecture
description: 'Pick a segmentation head + encoder backbone for segmentation_tools_pt via get_model(), and handle pretrained / air-gapped weights. Use when a user asks "which model / backbone?", "U-Net or FPN?", "which encoder?", or "how do I load weights offline on HPC?". Emits the model.* block for train.yaml.'
argument-hint: '[optional] compute budget, data size, internet availability'
user-invocable: true
disable-model-invocation: false
---
# Choose Model Architecture

Select the SMP segmentation head (`model.architecture`) and encoder backbone (`model.arch`), plus a
weights strategy. Discover valid names with `list_smp_architectures()` / `list_smp_encoders()`.

## Head (`model.architecture`)

| Head | Use when |
|---|---|
| `unet` (default) | Strong general baseline for defect/region segmentation |
| `unetplusplus` | Finer boundaries, more compute |
| `fpn` | Multi-scale objects |
| `deeplabv3plus` | Large receptive field, dense scenes |

## Encoder (`model.arch`)

| Encoder | Use when |
|---|---|
| `mobilenet_v2` (default) | Fast, light, good on HPC throughput |
| `resnet34` | Balanced accuracy/speed |
| `efficientnet-b3` | Higher accuracy, more memory |

## Weights strategy

- Internet available: `model.pretrained: true`, `model.weights_path: null` (downloads).
- Air-gapped HPC: on a connected machine run
  `download_encoder_weights('mobilenet_v2', '/path/to/mobilenet_v2.pth')`, copy the `.pth`, then set
  `model.weights_path` to it. Shape-mismatched keys (e.g. different `in_channels`) are skipped with a
  warning; all other pretrained layers still load.

## What this skill emits

- The `model:` block for `configs/train.yaml` (`architecture`, `arch`, `in_channels`, `pretrained`,
  `weights_path`) — `in_channels` must agree with the choose-channels decision.

## Library anchors

- `src/segmentation_tools_pt/model.py` — `get_model`, `list_smp_architectures`, `list_smp_encoders`,
  `download_encoder_weights`, `SMP_ARCHITECTURES`.
- Source notebook: `nbs/09_model.ipynb`.
