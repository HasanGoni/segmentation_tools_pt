# Custom UNet + SMP factory design

**Date:** 2026-10-01 (amended: ASPP/attention in v1)  
**Wayfinder map:** [Custom UNet + SMP factory #12](https://github.com/HasanGoni/segmentation_tools_pt/issues/12)  
**Parent context:** [Multi-head robustness #1](https://github.com/HasanGoni/segmentation_tools_pt/issues/1)  
**Decisions:** [#10](https://github.com/HasanGoni/segmentation_tools_pt/issues/10), [#11](https://github.com/HasanGoni/segmentation_tools_pt/issues/11), [#13](https://github.com/HasanGoni/segmentation_tools_pt/issues/13)

## Goal

One Hydra-driven factory that builds either an **SMP** segmentation model or a library-owned **classic UNet**, so chip / multilabel training can switch via config without code forks. Defaults match the CV panel recommendations; every knob is overridable. Bottleneck **ASPP** and **attention** are first-class v1 options (default off).

## Non-goals (v1)

- LoRA on custom UNet
- Changing DINOv3 / FPN SMP behavior beyond the existing `get_model` path
- Stacking ASPP **and** attention in one model (pick one via config)
- Hand-editing `src/` — implement in `nbs/09_model.ipynb` per `docs/agents/nbdev-implementation-contract.md`

## Approach (locked)

**Approach A:** `model.source: smp | custom_unet`.

| `source` | Behavior |
|----------|----------|
| `smp` (default) | Today’s path: `architecture` + `arch` → SMP via `get_model(...)` |
| `custom_unet` | Build `CustomUNet` from `model.custom_unet.*`; ignore SMP `arch` for structure |

Both return an `nn.Module` usable by `train()`, AMP, ONNX export, and freeze helpers. Prefer a thin `get_model_from_cfg(cfg)` used by `scripts/train.py`; keep `get_model(...)` for the SMP case and notebooks.

## Config schema

```yaml
model:
  source: smp                 # smp | custom_unet

  # SMP branch (source: smp)
  architecture: unet
  arch: mobilenet_v2
  pretrained: false
  weights_path: null
  in_channels: null           # usually from data_mode / data_patch

  # Custom branch (source: custom_unet)
  custom_unet:
    block_style: double       # double | residual
    encoder_depth: recommend  # recommend | auto | <int>
    bottleneck_min_size: 32   # for auto (and recommend table target)
    base_channels: 32
    max_channels: 512
    norm: group               # group | batch | instance
    num_groups: 8             # GroupNorm only; must divide channels (adjust if needed)
    bottleneck: none          # none | aspp | attention  (default none = shut off)
    aspp_rates: [1, 6, 12, 18]
    attention_heads: 4        # used when bottleneck: attention
```

Input spatial size for `recommend` / `auto` comes from `train.crop_size` (or `patch.patch_size` if crop unset). Factory must receive that size explicitly from Hydra wiring.

## CustomUNet architecture (v1)

### Block

- **`double` (default):** Conv3×3 → Norm → ReLU → Conv3×3 → Norm → ReLU (Ronneberger).
- **`residual`:** Conv → Norm → ReLU → Conv → Norm → add skip (1×1 if channels differ) → ReLU. Prefer this over a third plain conv.

Not supported in v1: three plain stacked convs.

### Encoder depth

- **`recommend`:** choose stages so bottleneck ≈ `bottleneck_min_size` for the crop. For **512** and default min **32** → **4** downs (512→256→128→64→32). Clamp to `[3, 6]`.
- **`auto`:** `stages = floor(log2(input_size / bottleneck_min_size))`, same clamp.
- **`<int>`:** user force.

### Channels

Stage `i` channels = `min(base_channels * 2**i, max_channels)` (encoder); decoder mirrors.

### Norm

- Default **GroupNorm** (`num_groups`, with channel-divisible fallback).
- `batch` / `instance` selectable for ablation.

### Bottleneck (v1 — #13)

After the deepest encoder stage, before the decoder:

| `bottleneck` | Behavior |
|--------------|----------|
| `none` (default) | Identity — shut off |
| `aspp` | Atrous Spatial Pyramid Pooling: parallel dilated 3×3 (rates from `aspp_rates`) + image pooling branch; concat → 1×1 project to bottleneck channels. If spatial H/W &lt; 16, drop rates that would exceed the map (or use lighter default `[1, 6, 12]`). |
| `attention` | Single multi-head self-attention over spatial tokens at bottleneck (`attention_heads`), channel-preserving; prefer ONNX-friendly ops |

Mutually exclusive. Chip / layout-heavy recipes set `aspp` or `attention` in project YAML; library defaults stay `none`.

### Interface for train loop

Expose `.encoder` / `.decoder` / `.segmentation_head` (or equivalent groupings) so existing `freeze_encoder` / `get_param_groups` keep working — prefer SMP-compatible attribute names where cheap. Bottleneck module lives with encoder or as `.bottleneck` between them.

## Hydra / CLI wiring

- Extend `_model_kwargs` / `scripts/train.py` (and `nbs/15_hydra_pipeline.ipynb` if that owns shared helpers) to branch on `model.source`.
- Update `configs/train.yaml` and `configs/train_multilabel_3head.yaml` with commented `custom_unet` block; default stays `source: smp`.
- Update `choose-model-architecture` skill briefly.

## Testing (nbdev)

In `nbs/09_model.ipynb`, synthetic tests:

1. `source=smp` still builds MobileNet UNet (pretrained=False).
2. `custom_unet` double + GroupNorm, depth=4, 1×1×64×64 → output same spatial, `num_classes` channels.
3. `encoder_depth: recommend` for size 512 → 4; `auto` with min_size 32 → 4; explicit `3` → 3.
4. `residual` block forward matches shape.
5. `bottleneck=aspp` and `bottleneck=attention` preserve `(B, C_bot, H_bot, W_bot)` then full UNet output shape.
6. `get_param_groups` / `freeze_encoder` do not crash on custom model.

Then `uv run nbdev-export` and `uv run nbdev-test --path nbs/09_model.ipynb`.

## Risks

- Custom UNet is random-init unless a later ticket adds ImageNet / SSL warm-start — document that DINOv3/SMP remains stronger for layout shift until then.
- GroupNorm `num_groups` must divide channel counts at every stage; resolve by adjusting groups or base_channels in factory.
- Attention ONNX: verify fixed spatial size at export (tile/crop size); document if dynamic H×W fails.
- ASPP with large rates on tiny bottleneck maps: auto-trim rates.

## Repo artifacts (wayfinder layout)

| Artifact | Location |
|----------|----------|
| Map | GitHub [#12](https://github.com/HasanGoni/segmentation_tools_pt/issues/12) (`wayfinder:map`) |
| Decision tickets | #10, #11, #13 |
| Design spec | this file under `docs/superpowers/specs/` |
| Implementation plan | `docs/superpowers/plans/` after approval (`writing-plans`) |
| nbdev contract | `docs/agents/nbdev-implementation-contract.md` |

## Implementation order

1. Depth helpers (`resolve_encoder_depth`) + tests  
2. Norm / block / encoder / decoder + `CustomUNet` + tests  
3. Bottleneck ASPP + attention modules + tests  
4. `get_model_from_cfg` + Hydra / `scripts/train.py` wiring  
5. Config comments + skill note + CHANGELOG  
