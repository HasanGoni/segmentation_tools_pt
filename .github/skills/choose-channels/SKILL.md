---
name: choose-channels
description: 'Choose how source image channels are assembled in segmentation_tools_pt (data.data_mode) and set the matching model.in_channels. Use when a user asks "grayscale or RGB?", "I have two/three image files per sample", "how do I use a pretrained encoder with 1-channel data?", or is editing data.data_mode. Emits data.data_mode + model.in_channels.'
argument-hint: '[optional] how many source image files per sample, grayscale or color'
user-invocable: true
disable-model-invocation: false
---
# Choose Channels (data_mode)

Pick `data.data_mode`, which decides how one or more source files become the model input tensor,
then set `model.in_channels` to match. Mismatch is validated at startup and fails early.

## Options (from `hydra_pipeline.CHANNEL_PRESETS`)

| `data.data_mode` | Source files | `model.in_channels` | Use when |
|---|---|---|---|
| `single` | one grayscale PNG | 1 | Simple 1-channel model, no ImageNet pretraining |
| `single_repeat_3` | one grayscale PNG ×3 | 3 | Reuse pretrained RGB encoder (MobileNetV2/ResNet/EfficientNet) without retraining conv1 |
| `single_to_3` | one PNG + tophat + sobel | 3 | Add hand-crafted contrast/edge channels (see feature-channels skill) |
| `two_channel` | `id_ch0.png`, `id_ch1.png` | 2 | Two physical acquisitions stacked |
| `three_channel` | three files | 3 | Three physical acquisitions stacked |

### Default naming convention

Unless you explicitly set `data.mask_suffix` or a multi-channel `data.data_mode`, **image and mask files must share the exact same base name**:

```text
raw_dir/
├── images/sample_001.png
└── masks/sample_001.png
```

Do **not** add `mask_` prefixes, `_mask` suffixes, or `_ch0` channel tags unless you configure the matching `mask_suffix` / `data_mode`. The defaults expect identical stems in `raw_dir/images/` and `raw_dir/masks/`.

## Decision matrix

| Situation | data_mode |
|---|---|
| Grayscale X-ray + pretrained encoder | `single_repeat_3` |
| Grayscale, training encoder from scratch | `single` |
| Want edge/tophat priors baked in | `single_to_3` |
| Genuine multi-acquisition inputs | `two_channel` / `three_channel` |

## What this skill emits

- `data.data_mode` (in `configs/data_patch.yaml`) and `model.in_channels` (kept consistent).
- A reminder that `mask_suffix` and per-file naming must match the expected layout (see AGENTS.md §10).

## Library anchors

- `src/segmentation_tools_pt/channels.py` — channel assembly.
- `src/segmentation_tools_pt/hydra_pipeline.py` — `CHANNEL_PRESETS`, `resolve_channel_preset`,
  `validate_data_mode_channels`.
- Source notebook: `nbs/01_channels.ipynb`.
