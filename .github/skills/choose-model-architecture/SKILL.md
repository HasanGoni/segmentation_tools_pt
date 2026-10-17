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

## Source (`model.source`): SMP or the library's own UNet

| `model.source` | Use when |
|---|---|
| `smp` (default) | Pretrained encoder (mobilenet_v2, DINOv3, convnext…) — the right default, esp. under domain shift |
| `custom_unet` | You want a small, fully library-owned classic UNet (double-conv + GroupNorm), no SMP/timm dependency, or a baseline to compare against |

`custom_unet` is **random-init**: `pretrained` / `weights_path` are ignored (a warning is raised), and
`train()` uses `freeze_epochs=0` with `encoder_lr = decoder_lr` instead of the pretrained-encoder warm-up
(override via `model.custom_unet.freeze_epochs` / `model.custom_unet.encoder_lr`). Pretrained SMP / DINOv3
stays the stronger choice for layout/domain shift until a warm-start option exists.

```bash
segpt-train configs/train.yaml model.source=custom_unet +model.custom_unet.encoder_depth=4 \
    model.pretrained=false model.weights_path=null train.crop_size=512
```

Knobs (`model.custom_unet.*`): `encoder_depth` (explicit int, `train.crop_size >= 2**depth`), `base_channels` (32),
`max_channels` (512), `num_groups` (8). Not implemented yet and rejected with a clear error: `encoder_depth: recommend|auto`,
`block_style: residual`, `norm: batch|instance`, `bottleneck: aspp|attention` (tickets #62–#65). `in_channels` and
`train.num_classes` still come from the data config, so the choose-channels decision applies unchanged.
Hands-on: `tutorials/08_custom_unet.ipynb`.

## Head (`model.architecture`) — `source: smp` only

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
- For `custom_unet`: `model.source: custom_unet` plus `model.custom_unet.{encoder_depth,base_channels,max_channels,num_groups}`.
- Or the `model:` block (`architecture`, `arch`, `in_channels`, `pretrained`, `weights_path`) —
  `in_channels` must agree with the choose-channels decision.
- For Chip/Void/Lifted recipes: default package `dinov3_vit_small`; throughput alt `convnext_tiny`;
  shut-off `mobilenet_v2`.

## Library anchors

- `src/segmentation_tools_pt/model.py` — `get_model_from_cfg`, `CustomUNet`, `get_model`, `resolve_encoder_name`, `list_smp_architectures`,
  `list_smp_encoders`, `download_encoder_weights`, `CURATED_ENCODER_ALIASES`.
- `src/segmentation_tools_pt/hydra_pipeline.py` — `resolve_train_schedule` (returns `TrainSchedule`; neutralises freeze/LR for `custom_unet`).
- Hydra packages: `configs/model/{mobilenet_v2,dinov3_vit_small,convnext_tiny,efficientnetv2_s}.yaml`.
- Source notebook: `nbs/09_model.ipynb`; tutorial: `tutorials/08_custom_unet.ipynb`.
