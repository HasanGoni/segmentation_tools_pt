---
name: choose-model-architecture
description: 'Pick a segmentation head + encoder backbone for segmentation_tools_pt via get_model(), and handle pretrained / air-gapped weights. Use when a user asks "which model / backbone?", "U-Net or FPN?", "which encoder?", or "how do I load weights offline on HPC?". Emits the model.* block or a Hydra model= package for train.yaml.'
argument-hint: '[optional] compute budget, data size, internet availability'
user-invocable: true
disable-model-invocation: false
---
# Choose Model Architecture

Select the SMP segmentation head (`model.architecture`) and encoder backbone (`model.arch`), plus a
weights strategy. Discover curated names with `list_smp_architectures()` / `list_smp_encoders()`.

Prefer **Hydra encoder preset packages** under `configs/model/` when switching backbones — they set
`model.*` only (not freeze/LR). Shut-off back to the library default is `model=mobilenet_v2`.

## Head (`model.architecture`)

| Head | Use when |
|---|---|
| `unet` (default) | Strong general baseline for defect/region segmentation |
| `unetplusplus` | Finer boundaries, more compute |
| `fpn` | Multi-scale objects |
| `deeplabv3plus` | Large receptive field, dense scenes |

## Encoder (`model.arch`) — curated vs experimental

### Curated (first-class — no warning; advertised by `list_smp_encoders()`)

| Encoder / package | Role | Notes |
|---|---|---|
| `mobilenet_v2` (`model=mobilenet_v2`) | **Library default** | Fast, light; silent starting config in `configs/train.yaml` |
| `dinov3_vit_small` (`model=dinov3_vit_small`) | **Primary recommendation** for Chip/Void/Lifted under domain shift | Also `dinov3_vit_base`; DPT-reassemble path |
| `convnext_tiny` (`model=convnext_tiny`) | **Throughput alternative** | Alias → `tu-convnext_tiny` |
| `efficientnetv2_s` (`model=efficientnetv2_s`) | Curated mid-size CNN | Alias → `tu-tf_efficientnetv2_s` |

Raw curated `tu-*` names (`tu-convnext_tiny`, `tu-tf_efficientnetv2_s`) also build without warnings.
Other common SMP encoders (`resnet34`, `efficientnet-b3`, …) remain valid registry names.

### Experimental `tu-*`

Any other SMP `tu-*` (timm) encoder is accepted by `get_model` with a **one-time warning** — useful for
agent experiments, not smoke-guaranteed. Do **not** emit experimental `tu-*` as the default for a
product recipe; prefer a curated package.

Unknown **non-`tu-*`** names hard-error.

## Weights strategy

- Internet available: `model.pretrained: true`, `model.weights_path: null` (downloads; DINOv3 via timm).
- Air-gapped HPC: on a connected machine run
  `download_encoder_weights('mobilenet_v2', '/path/to/mobilenet_v2.pth')`, copy the `.pth`, then set
  `model.weights_path` to it. Shape-mismatched keys (e.g. different `in_channels`) are skipped with a
  warning; all other pretrained layers still load.

## What this skill emits

- Prefer a Hydra defaults line: `- model: <package>` (or CLI `model=<package>`).
- Or the `model:` block (`architecture`, `arch`, `in_channels`, `pretrained`, `weights_path`) —
  `in_channels` must agree with the choose-channels decision.
- For Chip/Void/Lifted recipes: default package `dinov3_vit_small`; throughput alt `convnext_tiny`;
  shut-off `mobilenet_v2`.

## Library anchors

- `src/segmentation_tools_pt/model.py` — `get_model`, `resolve_encoder_name`, `list_smp_architectures`,
  `list_smp_encoders`, `download_encoder_weights`, `CURATED_ENCODER_ALIASES`.
- Hydra packages: `configs/model/{mobilenet_v2,dinov3_vit_small,convnext_tiny,efficientnetv2_s}.yaml`.
- Source notebook: `nbs/09_model.ipynb`.
