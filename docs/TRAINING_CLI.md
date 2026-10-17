# Training CLI (`segpt-train` / `segpt-build`)

Segmentation workflows from YAML configs: build reusable patches, train end-to-end, or export ONNX.

**Two config files, one shared base:** `data` + `patch` settings live in `configs/data_patch.yaml` and are included by both `build.yaml` and `train.yaml` via Hydra `defaults: [data_patch]`. Edit paths and patch knobs once — both commands stay aligned.

---

## 1. Install

From the cloned repository root:

```bash
uv sync               # creates .venv/, installs all deps + CLI commands
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` and `uv.lock` and installs everything — including the `segpt-build` and `segpt-train` entry points. No separate `pip install -e .` step is needed.

Verify the command is available:

```bash
segpt-train --help
```

---

## 2. Minimal usage (two config files)

Copy and edit the templates:

```bash
cp configs/build_example.yaml /path/to/my_project/build.yaml
cp configs/train_example.yaml /path/to/my_project/train.yaml
# edit data.raw_dir, data.patch_dir in either file (shared via data_patch defaults)
```

Build patches only (recommended — slim config, CPU-friendly HPC defaults):

```bash
segpt-build /path/to/my_project/build.yaml
segpt-build /path/to/my_project/build.yaml data.force_rebuild=true patch.patch_size=512
```

Submit patch building to HPC:

```bash
segpt-submit /path/to/my_project/build.yaml
segpt-submit /path/to/my_project/build.yaml hpc.queue=normal patch.num_workers=16
```

Train (reuses the patch cache from `data.patch_dir`):

```bash
segpt-train /path/to/my_project/train.yaml
```

> **Note:** `segpt-build configs/train.yaml` still works for backward compatibility, but `build.yaml` is preferred — it omits model, loss, augment, and GPU settings you do not need for patch extraction.

Run the full training pipeline (builds patches if missing, then trains):

That full run does this:

1. Discover samples in `data.raw_dir`
2. Build patch dataset into `data.patch_dir` (skipped if already built)
3. Compute or load `stats.json` in the patch folder
4. Train U-Net; checkpoints go to Hydra output dir (`outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`)
5. Export `model.onnx` from `best.pth` if `export.onnx: true`

### Patch size versus crop size

`patch.patch_size` controls the size stored on disk. `train.crop_size` controls the size seen by the model.

Recommended expert workflow:
- use a larger `patch.patch_size` when defects need more surrounding context
- use a smaller `train.crop_size` when GPU memory, batch size, or augmentation density favors smaller inputs
- keep validation reproducible with the deterministic center crop used by the val pipeline

Example:

```yaml
patch:
  patch_size: 512

train:
  crop_size: 256
```

That means: save 512x512 reusable patches, sample random 256x256 crops at train time, and use a 256x256 center crop at validation time.

In patch mode, `train.crop_size` must be less than or equal to `patch.patch_size`.

---

## 3. Direct mode (no patch cache)

When your images are already small enough to fit in GPU memory, or you want a quick experiment without pre-building patches, set:

```yaml
data:
  patch_mode: false
  raw_dir: /path/to/raw   # images/ + masks/ only — patch_dir not used
```

Then run as usual:

```bash
segpt-train /path/to/train.yaml data.patch_mode=false
```

**What changes:**
- Skips `build_patch_dataset` entirely
- Loads full images from `raw_dir/images/` + `raw_dir/masks/`
- Random crop at train time (`train.crop_size`); center crop for val
- Writes `stats.json` into `raw_dir` (not `patch_dir`)
- Train/val split still uses `patch.val_fraction` and `patch.seed`

## 3b. Build-only mode (`segpt-build`)

Use `segpt-build` with `configs/build.yaml` when you want to prepare or refresh the patch cache without starting model training.

```bash
segpt-build /path/to/build.yaml
segpt-build /path/to/build.yaml data.force_rebuild=true
segpt-submit /path/to/build.yaml
```

Requirements:
- `data.patch_mode=true` (default in `data_patch.yaml`)
- `data.patch_dir` must be set

What it does:
- discovers raw samples
- builds or rebuilds the patch cache in `data.patch_dir` (parallel workers via `patch.num_workers`)
- computes or loads `stats.json`
- exits — no model, no GPU, no Hydra `outputs/` dir (`hydra.job.chdir: false`)

Config sections used by build:

| Section | Purpose |
|---------|---------|
| `data` | Raw paths, channel mode, mask naming, `force_rebuild`, `fit_rects` |
| `patch` | Patch size, stride, filters, val split, **parallel workers** |
| `train` | `num_classes` only (multi-class mask encoding) |
| `model` | `in_channels` only (validated against `data_mode`) |
| `hpc` | CPU queue defaults for `segpt-submit` |

For large industrial datasets, submit patch build once to HPC (CPU queue), populate the shared cache, then run many GPU training jobs against the same `data.patch_dir`.

**When to use patch mode (`patch_mode: true`, default):**
- Very large images (X-ray panels, wafer maps) where one image >> GPU memory
- You want a reusable patch cache shared across many training runs
- Hard-example oversampling via sparse-foreground patch weights

**When to use direct mode (`patch_mode: false`):**
- Small/medium images that fit in memory after crop
- Quick iteration — no patch build step
- Prototyping on a handful of labeled images

---

## 4. Data layout

Your **raw** folder must look like this:

### Single-channel example

```
raw/
├── images/
│   ├── sample_001.png
│   └── sample_002.png
└── masks/
    ├── sample_001.png      # same stem as image (use mask_suffix: "")
    └── sample_002.png
```

Set in config:

```yaml
data:
  data_mode: single
  mask_suffix: ""
  n_source_channels: 1   # implied by data_mode
```

### Two native channels

```
raw/
├── images/
│   ├── id_ch0.png
│   ├── id_ch1.png
└── masks/
    └── id_mask.png
```

```yaml
data:
  data_mode: two_channel
  mask_suffix: "_mask"
```

`data_mode` also supports `single_to_3` (derive 3 channels from one grayscale file) and `three_channel`.

---

## 5. Config sections (cheat sheet)

### Shared base (`configs/data_patch.yaml`)

Included by both `build.yaml` and `train.yaml`. Edit here (or override in your project copy) to keep build and train aligned.

### Build config (`configs/build.yaml`)

| Section | Purpose |
|---------|---------|
| `data` | Same as shared base + `build_only: true` |
| `patch` | Patch extraction + `num_workers` for parallel I/O |
| `train` | `num_classes`, `seed` |
| `model` | `in_channels` validation |
| `hpc` | CPU job defaults (`queue: normal`, `device: cpu`) |

### Train config (`configs/train.yaml`)

| Section | Purpose |
|---------|---------|
| `data` | Raw paths, `patch_mode`, channel mode, patch rebuild |
| `patch` | Patch size, stride, `min_fg_pixels`, val split |
| `model` | Encoder arch, pretrained / local weights |
| `train` | Epochs, LR, batch size, crop size, AMP, resume |
| `loss` | Tversky/Dice/boundary weights + **phase schedule** |
| `metrics` | Threshold, which metric saves `best.pth` |
| `augment` | Per-image + batch-level augmentations |
| `monitoring` | Per-epoch `metrics_log`, activation stats, pred overlays, opt-in `error_analysis` |
| `export` | ONNX export after training |
| `hpc` | GPU job defaults |

**Encoder freeze** (`train.freeze_epochs`): `0` = train the encoder from epoch 0; `N > 0` = decoder warm-up — encoder frozen for epochs `0..N-1`, unfrozen at epoch `N`; `-1` = encoder frozen for the whole run (only decoder + head train — good for DINOv3/ViT encoders or small datasets that overfit a tuned backbone).

### Custom UNet (`model.source`, issue #61)

`model.source: smp` (default) builds the SMP model from `model.arch`; `model.source=custom_unet` builds the library's own
UNet from `model.custom_unet.*` (`encoder_depth`, `bottleneck_min_size`, `base_channels`, `max_channels`, `norm`, `num_groups`, `block_style`, `bottleneck`, `aspp_rates`). The block style (`double` conv or `residual`) and the norm (`group`, `batch`, `instance`) are selectable; the defaults are `double` and `group`.
It starts from random weights, so training forces `freeze_epochs=0` and `encoder_lr=decoder_lr` unless you set
`model.custom_unet.freeze_epochs` / `encoder_lr`. Requires `train.crop_size >= 2**encoder_depth`.

`encoder_depth` is the number of "downs" (how many times the image is halved on the way down) and defaults to 4 when absent; it can also be
`recommend`, `auto` or an integer (#62). The "bottleneck" is the smallest feature map, in the middle of the UNet, after the last halving. `recommend` and `auto` read `train.crop_size`: `auto` is
`floor(log2(crop / bottleneck_min_size))`, `recommend` rounds the same ratio to the nearest step (a 512 crop with the default
`bottleneck_min_size: 32` gives 4 downs: 512 -> 256 -> 128 -> 64 -> 32); both are clamped to [3, 6]. An integer is used as given.
`norm` is `group` (default; the group count is lowered per layer to divide that layer's channels), `batch` or `instance`.
BatchNorm takes its statistics from the batch, so train with batch size >= 2 and check eval results separately.

```bash
segpt-train configs/train.yaml model.source=custom_unet +model.custom_unet.encoder_depth=recommend +model.custom_unet.norm=group model.pretrained=false model.weights_path=null
```

Hands-on: `tutorials/08_custom_unet.ipynb`. `block_style: residual` (#63) swaps each double conv for a residual block (Conv-Norm-ReLU-Conv-Norm, add the input, ReLU; a bias-free 1x1 conv carries the input when the channel count changes), e.g. `+model.custom_unet.block_style=residual`. Works with every `norm`; an unknown value is rejected up front.

`bottleneck: aspp` (#64) runs parallel dilated 3x3 convs (`aspp_rates`, default `[1, 6, 12, 18]`) plus a whole-map pooling branch on the bottleneck, concatenates them and projects back with a 1x1 conv, so the output shape is unchanged: `+model.custom_unet.bottleneck=aspp`. When the bottleneck is smaller than 16 px per side (`train.crop_size // 2**encoder_depth`), rates that are not smaller than the side are dropped (falling back to `[1]`). `bottleneck: none` (default) changes nothing. `attention` follows in #65.

### Loss phases

`loss.phases` is a list of `{until_epoch, w_tversky, w_dice, w_boundary}`.
`until_epoch` is exclusive (epoch 0..9 uses the first block if `until_epoch: 10`).
Use `until_epoch: -1` on the last phase for “until end of training”.

### Class-hierarchy loss (`lifted ⊂ chip`)

For `task: multilabel`, name the nesting and give it a weight; the loss then penalises a sub-class predicted
outside its super-class (`relu(p_sub - p_super)`), and `train_loss_hierarchy` appears in the epoch log and `metrics.csv`:

```yaml
train:
  task: multilabel
  hierarchy:
    lifted_subset_of_chip: true   # which classes nest (names come from data.class_names)
    weight: 0.5                   # 0 / missing = off (default); run-wide, phases may override
loss:
  phases:
    - {until_epoch: 10, w_tversky: 0.6, w_dice: 0.4, w_boundary: 0.0, w_hierarchy: 0.0}  # optional per-phase weight
    - {until_epoch: -1, w_tversky: 0.5, w_dice: 0.3, w_boundary: 0.2}                     # falls back to train.hierarchy.weight
```

Off by default: no `weight` (or `0`) gives exactly the previous loss. Training fails early if the weight is `> 0` with
a non-multilabel task or when no pair resolves (typo in a class name). This is a soft nudge; `apply_hierarchy` at
inference still hard-clips. A/B it with `segpt-train configs/train_multilabel_3head.yaml train.hierarchy.weight=0.0`
vs `=0.5` — hands-on walkthrough in `tutorials/06_hierarchy_loss.ipynb` (synthetic) and the real-data acceptance run in `tutorials/07_hierarchy_ab_real_data.ipynb`.

### Augmentations

**Per-image** (Albumentations, YOLO26-style): `translate`, `scale`, `degrees`, `shear`, `perspective`, `fliplr`, `flipud`, `hsv_*`, `bgr`, `erasing`, …

Preset shortcut: `augment.preset: yolo26m` (or `yolo26n/s/l/x`, `legacy`). Override any field; `null` keeps the preset value.

**Batch-level** (applied in the train loop, YOLO-style **pick-one** — at most one type per batch):
`augment.batch.*` — master switch `enabled` (default **false**) plus per-type probabilities
`mosaic`, `mixup`, `cutmix`, `copy_paste` (default **0**) and `copy_paste_mode`. Enabling the
master switch alone does nothing until you set a prob > 0; master off or all-zero probs reproduces
"no batch aug" training exactly. Each type fires with exactly its configured probability; types
never stack in the same batch.

For **orientation-sensitive X-ray** (fixed camera), start conservative — keep flips off and batch
augs off or tiny:

```yaml
augment:
  preset: yolo26m
  fliplr: 0.0
  flipud: 0.0
  translate: 0.10
  scale: 0.30
  batch:
    enabled: true
    mosaic: 0.10
    mixup: 0.05
    cutmix: 0.0
    copy_paste: 0.10
    copy_paste_mode: flip
```

For **high recall** inspection: `metrics.best_metric: recall` and slightly higher `mixup` / `copy_paste` (watch false alarms on real val).

> **Note:** batch augs see binary (`(B,1,H,W)` / `(B,H,W)`) and multilabel (`(B,K,H,W)`) masks
> directly. On **multiclass** integer-label masks, `mixup` blends then truncates labels back to
> integers (soft labels become hard) — prefer `cutmix` / `copy_paste` / `mosaic` there.

**See it visually:** [tutorials/04_augmentation_playground.ipynb](../tutorials/04_augmentation_playground.ipynb)
— point it at your own `IMAGE_PATHS` / `MASK_PATHS` (or run it on its synthetic demo data) to
visualize every batch-aug type forced on, realistic pick-one sampling over virtual batches, and
the sample-level Albumentations stack.

### Class-targeted morph postprocess (inference decode only, issue #19)

Multilabel sigmoid decoders can leave tiny speckle false alarms in one class channel.
`postprocess.class_morph` cleans **one named class** at decode time with classical
morphology (opening + connected-component min-area) — opt-in, master switch default
**false**, and never applied to the training loss or to raw train metrics (those must
judge the model as-is):

```yaml
postprocess:
  class_morph:
    enabled: true
    class_name: void     # resolved against data.class_names; unknown name fails clearly
    kernel_size: 3       # opening structuring element side (odd)
    min_area: 16         # drop connected components smaller than this (pixels)
```

Fixed decode order at inference/production:

```python
from segmentation_tools_pt.hydra_pipeline import resolve_class_morph_cfg
from segmentation_tools_pt.inference import decode, apply_class_morph, apply_hierarchy

morph_cfg = resolve_class_morph_cfg(cfg, class_names)   # None when disabled -> decode unchanged
masks = decode(probs, threshold=thresholds, task='multilabel')
if morph_cfg is not None:
    masks = apply_class_morph(masks, **morph_cfg)        # class_idx/kernel_size/min_area
masks = apply_hierarchy(masks, pairs=hierarchy_pairs)
```

### Production tiled-inference knobs (`inference.*`, issue #20)

Full inspection images don't fit one forward pass. `predict_tiled` tiles them, forwards
tiles in chunks of `inference_batch_size`, and stitches with Gaussian-weighted blending —
batch size trades GPU memory against throughput only (batch 1 is bit-identical to batch
999), never accuracy. Knobs live in config so production needs no code change:

```yaml
inference:
  tile_size: 256            # tile side fed to the model
  overlap: 32               # adjacent-tile overlap; Gaussian blending stitches tiles back
  inference_batch_size: 8   # tiles per forward — raise for throughput, lower if OOM
  prefetch_factor: null     # loader-driven tiled inference only; predict_tiled ignores it
```

```python
from segmentation_tools_pt.hydra_pipeline import resolve_inference_cfg
from segmentation_tools_pt.inference import predict_tiled

knobs = resolve_inference_cfg(cfg)
prob = predict_tiled(model, image, device, **knobs)   # (C, H, W) probabilities
```

The same `prefetch_factor` knob exists for training loaders (`train.prefetch_factor`,
threaded into `train()`'s patch-mode loaders; inert unless `num_workers > 0`) and for
`build_dataloaders` / `build_direct_dataloaders` — it keeps batches in flight per worker
so the GPU doesn't idle between batches on network-share patch caches.

ONNX export stays **FP32 only** (`export.onnx`, opset 18, dynamic batch axis,
runtime-verified); curated encoders (`convnext_tiny`, `efficientnetv2_s`,
`dinov3_vit_small`, …) export through the standard `get_model` path. No INT8, no TensorRT.

### Visual review report (`review_report`, issue #50)

One self-contained `report.html` for checking a model's result by eye, without starting a
kernel: ground truth vs prediction vs errors for the worst / random / best samples, with layer
toggles, a class selector and a per-class Dice/IoU table. Masks are exact SVG outlines, never
JPEG; 16-bit images keep their raw values in `assets/<id>/full.png`.

```python
from segmentation_tools_pt.review_report import samples_from_dataset, build_report
samples = samples_from_dataset(model, val_ds, device, class_names=['void'])   # one model pass
build_report('T-123', 'reports/T-123', samples, seed=0, cfg=dict(n_worst=6, n_random=6, n_best=3))
```

Open `reports/T-123/report.html` in a browser. `report.json` next to it holds the pooled metrics
(`dice.void`, `iou.mean`, …) and which samples were shown. The datasets read 8-bit grayscale, so
pass `raw_image_fn=` (re-read the original file with `cv2.IMREAD_UNCHANGED`) to keep 16-bit
values. Try it without data: `make_dummy_samples(12, task='multilabel', dtype='uint16')`.
Click-to-comment feedback, gates and the summary text arrive with #51 / #52; evaluation and
embedding work add their own sections to this same page (maps #28, #39, #48).

---

## 6. CLI overrides (Hydra)

Any config key can be overridden on the command line:

```bash
segpt-build /path/to/build.yaml data.force_rebuild=true patch.patch_size=512

segpt-train /path/to/train.yaml \
  data.raw_dir=/path/to/dataset/segmentation \
  data.patch_dir=/path/to/dataset/patches \
  train.epochs=80 \
  metrics.best_metric=recall \
  augment.batch.enabled=true augment.batch.mixup=0.10
```

Force rebuild patches during training (or use `segpt-build` instead):

```bash
segpt-train /path/to/train.yaml data.force_rebuild=true
```

### Alternative invocations

```bash
# Explicit Hydra style
segpt-build --config-path /path/to/configs --config-name build
segpt-train --config-path /path/to/configs --config-name train

# Without installing CLI (from repo root)
python scripts/build.py data.raw_dir=... data.patch_dir=...
python scripts/train.py data.raw_dir=... data.patch_dir=...
```

---

## 7. Outputs

After a run, Hydra creates something like:

```
outputs/2026-06-02/15-30-45/
├── .hydra/                  # resolved config
├── checkpoints/
│   ├── best.pth
│   ├── latest.pth
│   ├── history.json         # per-epoch train_*/val_* loss + metrics (+ per-class IoU)
│   ├── metrics.csv          # same rows as history.json, flat, open while training runs
│   ├── error_analysis/      # if monitoring.error_analysis.enabled: true
│   │   ├── train_top12.jpg  # worst samples per split (4-col: image | GT | pred | FN/FP diff)
│   │   ├── val_top12.jpg    # multilabel: one per class, val_top12_chip.jpg ...
│   │   └── top_ranked.json  # sample_id, split, score, rank_by, class
│   └── model.onnx           # if export.onnx: true
└── segpt-train.log          # if configured
```

`data.patch_dir` (your chosen path) holds:

```
patches/
├── train/images/ ...
├── train/masks/ ...
├── val/...
├── metadata.json
└── stats.json               # channel mean/std for normalization
```

Resume training: `train.resume: true` loads `latest.pth` from the new run’s `checkpoints/` folder.

---

## 8. Offline encoder weights (HPC without internet)

On a machine with internet:

```python
from segmentation_tools_pt.model import download_encoder_weights
download_encoder_weights('resnet34', '/data/weights/resnet34.pth')
```

Copy the `.pth` to HPC and set:

```yaml
model:
  arch: resnet34
  pretrained: false
  weights_path: /data/weights/resnet34.pth
```

Or switch a curated encoder package (sets `model.*` only; shut-off = MobileNet):

```bash
segpt-train configs/train.yaml model=dinov3_vit_small
segpt-train configs/train_multilabel_3head.yaml model=mobilenet_v2   # shut-off
```

Packages live under `configs/model/` (`mobilenet_v2`, `dinov3_vit_small`, `convnext_tiny`, `efficientnetv2_s`).

## 9. Use from another project folder

1. Clone the repo and run `uv sync` once (see Section 1)
2. Activate the environment: `source /path/to/segmentation_tools_pt/.venv/bin/activate`
3. Put `my_train.yaml` anywhere
4. Run `segpt-train /path/to/my_train.yaml`

No need to copy the library into your project — only the config and data paths.

---

## 10. Troubleshooting

| Issue | Check |
|-------|--------|
| `No samples with matching images+masks` | `data_mode`, `mask_suffix`, file names under `images/` and `masks/` |
| All patches `fg_pixels=0` | Mask encoding: must be `{0,255}` or `{0,1}`; see tutorial Step 1b |
| Empty cleaned mask (rotated rect) | Sync latest `masks.py` (`int32` for OpenCV contours on Linux) |
| `segpt-train` not found | Re-run `uv sync` and make sure `.venv` is activated |
| Mixup seems inactive | Batch augs are off unless `augment.batch.enabled=true` **and** the type prob > 0; on multiclass integer masks `mixup` blends then truncates labels (prefer `cutmix`/`copy_paste` there) |
| Direct mode OOM | Switch to `patch_mode: true` or reduce `train.crop_size` / `train.batch_size` |

Interactive walkthrough with synthetic data: [tutorials/01_training_walkthrough.ipynb](../tutorials/01_training_walkthrough.ipynb).

## Per-epoch train vs validation metrics

Every epoch writes **separate train and validation** numbers to `history.json`, `metrics.csv` and the console:
`train_loss`, `val_loss`, `train_iou`/`val_iou` (also `dice`, `precision`, `recall`) and, per class,
`train_iou_<class>` / `val_iou_<class>`. **Background is never listed or averaged**: in `multiclass`
label 0 is skipped; `binary` has one foreground channel (`foreground`, or the first `class_names` entry);
every `multilabel` head is a foreground class. Class names come from `class_names` (indexed by label id).

```yaml
monitoring:
  metrics_log:
    per_class: true           # per-class IoU columns (default on)
    console_per_class: false  # also print per-class val IoU on the epoch line
    csv: true                 # metrics.csv, rewritten from history so resume never duplicates rows
```

Quick look while training: `column -s, -t < checkpoints/metrics.csv | less -S`.
