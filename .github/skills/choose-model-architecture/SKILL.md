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

Terms: "downs" = how many times the image is halved (`encoder_depth`, default 4 when absent); "bottleneck" = the smallest feature map in the middle of the UNet.
Knobs (`model.custom_unet.*`): `encoder_depth` (`recommend` | `auto` | explicit int; an int needs `train.crop_size >= 2**depth`),
`bottleneck_min_size` (32), `base_channels` (32), `max_channels` (512), `norm` (`group` | `batch` | `instance`), `num_groups` (8, GroupNorm only).
`recommend`/`auto` read `train.crop_size`: a 512 crop gives 4 downs; `auto` is `floor(log2(crop/min))`, `recommend` rounds to the nearest step; both clamped to [3, 6].
`batch` norm misbehaves with tiny batches (stats come from the batch), so prefer `group` or `instance` there. `block_style` (`double` default | `residual`: Conv-Norm-ReLU-Conv-Norm plus a skip, with a 1x1 conv when channels change; same output shape, a few more parameters, works with every norm).
`bottleneck` (`none` default | `aspp` | `attention`; one value, so they are mutually exclusive): `aspp` adds parallel dilated 3x3 convs (`aspp_rates`, default `[1, 6, 12, 18]`) plus a whole-map pooling branch on the smallest feature map, for layout-heavy images where wide context matters; output shape is unchanged and it adds parameters. On a bottleneck under 16 px per side, rates not smaller than the side are dropped automatically.
`attention` (`attention_heads`, default 4, must divide the bottleneck channel count, which is `min(base_channels * 2**encoder_depth, max_channels)`; a bad value is rejected before the model is built) adds one self-attention layer with a residual add over the smallest feature map, for global context when a defect can only be judged from the whole layout. Output shape is unchanged. Memory grows with the square of the bottleneck pixel count: a 512 crop at depth 4 is 32x32 = 1024 tokens (fine), at depth 3 it is 64x64 = 4096 tokens (heavy), so keep the bottleneck small. It exports to FP32 ONNX, also with dynamic H/W. `in_channels` and
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
- For `custom_unet`: `model.source: custom_unet` plus `model.custom_unet.{encoder_depth,bottleneck_min_size,base_channels,max_channels,norm,num_groups,block_style,bottleneck,aspp_rates,attention_heads}`.
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
