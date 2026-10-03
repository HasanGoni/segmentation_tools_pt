# AGENTS.md — Repository Overview

> Read this first. It tells you everything about what this repo is, why it was built,
> how every folder and file fits together, and the rules you must follow when working in it.

---

## 1. What this repo is

**`segmentation_tools_pt`** is a PyTorch library for **industrial image segmentation**.

It was built to solve a recurring problem: every new segmentation project
(pin detection, wafer maps, PCB inspection, …) required re-writing the same boilerplate —
mask loading, patch extraction, paired augmentations, IoU metrics, training loops, ONNX export.
This library collects all of that into one clean, tested, reusable package so each new project
starts from a solid foundation, not from copy-pasted scripts.

**Domain:**
- Input images are typically grayscale or multi-channel industrial X-ray / optical inspection images
- Ground-truth masks are binary (defect / no-defect) or multi-class label maps
- Models are U-Net style encoder–decoder networks (via `segmentation-models-pytorch`)
- Inference runs on HPC GPU nodes (LSF cluster) or locally

---

## 2. Development methodology — nbdev

This project uses **[nbdev](https://nbdev.fast.ai/)**.

### The golden rule

```
Edit notebooks in nbs/  →  nbdev_export  →  src/ is auto-generated
```

**Never edit files inside `src/segmentation_tools_pt/` directly.**
The notebooks in `nbs/` are the source of truth.
The `.py` files in `src/` are generated artifacts.

### Why nbdev?

- Notebooks let you write exploratory code, test it immediately with real data, and then
  crystallize working code into a function — all in one place.
- The notebook IS the documentation. If you come back after two years, you see not just
  the function but also why it was built and how it was tested.
- Every function has a docstring, an example, and a test in the same cell sequence.

### Workflow

```bash
# After editing any notebook:
nbdev-export        # regenerate src/ from nbs/
nbdev-test          # run all in-notebook tests
nbdev-preview       # local docs preview (Quarto)
```

On NFS mounts, verify agent edits landed on disk before export — see `.cursor/skills/nbdev-notebook-verify/SKILL.md`.

---

## 3. Full repository structure

```
segmentation_tools_pt/
│
├── nbs/                        ← SOURCE OF TRUTH (edit here)
│   ├── index.ipynb             # Library overview; also generates README.md
│   ├── 00_core.ipynb           # → core.py       — version, shared utilities
│   ├── 01_channels.ipynb       # → channels.py   — multi-channel image loading
│   ├── 02_masks.ipynb          # → masks.py       — mask binarization, rotated-rect fitting
│   ├── 03_patches.ipynb        # → patches.py     — sliding-window patch extraction
│   ├── 04_dataset_build.ipynb  # → dataset_build.py — full pipeline: discover → patch → split
│   ├── 05_transforms.ipynb     # → transforms.py  — Albumentations augmentation presets (YOLO26)
│   ├── 06_dataset.ipynb        # → dataset.py     — SegmentationDataset, DirectDataset, loaders
│   ├── 07_losses.ipynb         # → losses.py      — Tversky, Dice, boundary losses + phase schedule
│   ├── 08_metrics.ipynb        # → metrics.py     — IoU, Dice, recall, precision; per-epoch tracking
│   ├── 09_model.ipynb          # → model.py       — get_model(), download_encoder_weights()
│   ├── 10_train.ipynb          # → train.py       — resumable training loop with AMP
│   ├── 11_inference.ipynb      # → inference.py   — sliding-window inference on full images
│   ├── 12_visualize.ipynb      # → visualize.py   — overlay masks, plot history
│   ├── 13_export.ipynb         # → export.py      — ONNX export and verification
│   ├── 14_cli.ipynb            # → cli.py         — segpt-build, segpt-train, segpt-submit CLI entry points
│   ├── 15_hydra_pipeline.ipynb # → hydra_pipeline.py — shared Hydra helpers for build + train scripts
│   ├── _quarto.yml             # Quarto docs config
│   └── nbdev.yml               # nbdev config
│
├── src/
│   └── segmentation_tools_pt/  ← AUTO-GENERATED (do not edit)
│       ├── __init__.py
│       ├── channels.py
│       ├── cli.py
│       ├── core.py
│       ├── dataset.py
│       ├── dataset_build.py
│       ├── export.py
│       ├── hydra_pipeline.py
│       ├── inference.py
│       ├── losses.py
│       ├── masks.py
│       ├── metrics.py
│       ├── model.py
│       ├── patches.py
│       ├── train.py
│       ├── transforms.py
│       └── visualize.py
│
├── scripts/
│   ├── build.py                # Hydra-driven patch-cache preparation (segpt-build)
│   └── train.py                # Hydra-driven end-to-end training (segpt-train)
│
├── configs/
│   ├── data_patch.yaml         # Shared data + patch base (Hydra defaults)
│   ├── build.yaml              # Slim patch-build config (segpt-build)
│   ├── build_example.yaml      # Annotated build template
│   ├── train.yaml              # Full training config (segpt-train)
│   └── train_example.yaml      # Annotated training template
│
├── docs/
│   └── TRAINING_CLI.md         # Detailed user guide for segpt-train
│
├── hpc/
│   ├── submit.sh               # bsub job script (called by segpt-submit)
│   └── README.md               # HPC submission guide + troubleshooting
│
├── tutorials/
│   ├── 01_training_walkthrough.ipynb   # Step-by-step: data → patches → train → inspect
│   ├── 02_next_steps.ipynb             # Multi-class, inference, ONNX deployment
│   └── 04_augmentation_playground.ipynb # Visualize batch augs (pick-one) + sample-level stack on your own images/masks
│
├── outputs/                    # Hydra run outputs (gitignored); see Section 8
│
├── AGENTS.md                   # ← this file
├── CHANGELOG.md                # Feature history
├── README.md                   # Auto-generated from nbs/index.ipynb
├── pyproject.toml              # Package metadata, dependencies, CLI entry points
└── uv.lock                     # Pinned dependency lockfile (uv)
```

---

## 4. Module-by-module guide

### `channels.py` — multi-channel image loading
Resolves how source channels are assembled from one or more image files on disk.
`data_mode` in the config controls the strategy:
- `single` — one grayscale PNG → 1-channel tensor
- `single_repeat_3` — one grayscale PNG repeated 3× → 3-channel tensor (use with ImageNet-pretrained encoders)
- `single_to_3` — one grayscale PNG + tophat + sobel → 3-channel tensor
- `two_channel` — two separate image files (`id_ch0.png`, `id_ch1.png`) stacked
- `three_channel` — three files stacked

### `masks.py` — mask handling
- `binarize()` — converts `{0,255}` or `{0,1}` mask PNG to float32 `[0,1]`
- `fit_rotated_rects_mask()` — replaces each connected blob in the mask with its
  minimum bounding rotated rectangle; used to snap noisy manual annotations to
  clean geometric shapes before patch extraction

### `patches.py` — sliding-window patch extraction
Extracts fixed-size patches from large images using a configurable stride.
Filters out near-empty patches (controlled by `min_fg_pixels` and `max_empty_frac`).
Returns patch metadata (position, fg pixel count) for weighted sampling later.

### `dataset_build.py` — full build pipeline
`build_patch_dataset()` orchestrates the whole data-preparation pipeline:
1. `discover_samples()` — find matching image/mask pairs in `raw_dir`
2. Apply `fit_rotated_rects_mask()` if `fit_rects: true`
3. Extract patches with `patches.py` → save to `patch_dir/train/` and `patch_dir/val/`
4. Write `metadata.json` and `stats.json` (channel mean/std for normalization)

### `transforms.py` — augmentation presets (YOLO26-aligned)
`AugmentConfig` dataclass maps to Albumentations transforms.
Six presets are defined in the YAML (`yolo26n/s/m/l/x` and `legacy`), and the
config exposes knobs for geometric (translate, scale, rotate, shear, perspective),
color (HSV, BGR swap), and erasing (CoarseDropout). **These presets are still
config-only placeholders; `scripts/train.py` passes `aug=None` to `train()`**, so
they are **not** applied. **Batch-level augmentations ARE wired** (issue #17):
`mosaic` / `mixup` / `cutmix` / `copy_paste` live under `augment.batch.*`
(master switch `enabled`, default **false**; per-type probs default **0**) and run
in `train_one_epoch` as YOLO-style **pick-one** — at most one type per batch, each
type firing with exactly its configured probability. Master off or all-zero probs
is a strict no-op (training identical to before). The active sample-level pipeline
in `transforms.get_train_transforms()` is a fixed geometric + photometric stack
(RandomCrop, Affine, GridDistortion, RandomBrightnessContrast, GaussNoise,
GaussianBlur) plus the config-driven `boundary_blur` and optional `flips` —
unchanged. The full preset system would be wired in `nbs/05_transforms.ipynb`;
batch augs live in `nbs/10_train.ipynb` (`apply_batch_augs`) with the Hydra
resolver `resolve_batch_aug_cfg` in `nbs/15_hydra_pipeline.ipynb`.

### `dataset.py` — PyTorch datasets and loaders

Two dataset classes:

| Class | When to use |
|---|---|
| `SegmentationDataset` | Patch mode — reads pre-extracted patches from `patch_dir` |
| `DirectDataset` | Direct mode — loads full images from `raw_dir`, random/center crop on-the-fly |

Helper functions: `build_datasets()`, `build_dataloaders()` (patch mode),
`build_direct_datasets()`, `build_direct_dataloaders()` (direct mode).

**Gotchas when writing your own loop/notebook cell against `SegmentationDataset` directly:**
- `__getitem__` returns a **dict** `{'image': tensor, 'mask': tensor, 'sample_id': str}`,
  not a tuple — use `sample['image']`, `sample['mask']`.
- `channel_names` must have exactly `n_source_channels` entries (e.g. `['ch0']` for
  `n_source_channels=1`). The class default (`DEFAULT_CHANNELS`, 2-channel) silently mismatches
  single-channel data unless you override it.
- Single-channel patch caches (`data.data_mode: single`) store `images/{id}.png` and
  `masks/{id}.png` under **identical** base names — no `_ch0`/`_mask` suffix. Multi-channel
  caches keep `_ch{i}` on images only; masks always stay plain.

### `losses.py` — combined segmentation loss
`SegmentationLoss` combines Tversky + Dice + boundary term with a **phase schedule**:
weights change at epoch boundaries defined in `loss.phases` in the config.
This lets you start with coarse region loss and progressively add boundary sharpening.

### `metrics.py` — per-epoch metrics
Computes pixel-level IoU, Dice, recall, and precision with a configurable sigmoid threshold.
`MetricsTracker` accumulates per-batch values and reports per-epoch means.
`best_metric` in the config selects which metric triggers saving `best.pth`.

### `model.py` — model factory
`get_model(arch, in_channels, num_classes, architecture='unet')` — builds any
`segmentation-models-pytorch` segmentation head (`unet`, `fpn`, `deeplabv3plus`, …) with any
supported encoder backbone (`mobilenet_v2`, `resnet34`, `efficientnet-b3`, …). Use
`list_smp_architectures()` and `list_smp_encoders()` to discover valid names.
`download_encoder_weights()` — downloads pretrained weights on a machine with internet
so they can be copied to air-gapped HPC nodes.

**Air-gapped HPC workflow:**
```python
# On a machine with internet:
download_encoder_weights('mobilenet_v2', '/tmp/mobilenet_v2.pth')
# Copy to HPC, then set in config:
# model.weights_path: /home/.../Models/smp_models/mobilenet_v2.pth
```

If the `.pth` file was saved with a different `in_channels`, mismatched keys are
skipped with a warning instead of raising an error — all other pretrained layers load.

### `train.py` — training loop
`train()` — resumable training loop with:
- AMP (automatic mixed precision)
- Encoder freeze via `freeze_epochs`: `0` = no freeze, `N > 0` = warm-up then unfreeze at epoch `N`, `-1` = frozen forever (decoder + head only train)
- Batch-level augmentations (mosaic/mixup/cutmix/copy_paste) in the loop — YOLO-style pick-one behind `augment.batch.enabled` (default off; per-type probs default 0; off/zero = strict no-op)
- Checkpoint: `latest.pth` every epoch, `best.pth` when `best_metric` improves
- `history.json` — full epoch-by-epoch loss + metric log
- `resume_from` — explicit checkpoint path to resume from (not just `latest.pth`)
- **Callbacks** — optional `TrainCallback` hooks at train / epoch / batch granularity (see below)

Accepts either `root` (patch mode) or pre-built `train_loader` + `val_loader` (direct mode).

#### Training callbacks

Pass `callbacks=[...]` to `train()` (or to `train_one_epoch` / `validate` when driving the loop yourself). Subclass `TrainCallback` and override only the hooks you need. All hooks receive a mutable `TrainState` with model, optimizer, current batch, logits, metrics, etc.

| Hook | When it fires |
|---|---|
| `on_train_begin` / `on_train_end` | Start / end of `train()` |
| `on_epoch_begin` / `on_epoch_end` | Start / end of each epoch |
| `on_train_batch_begin` / `on_train_batch_end` | Each training batch |
| `on_val_begin` / `on_val_end` | Start / end of validation |
| `on_val_batch_begin` / `on_val_batch_end` | Each validation batch |
| `on_checkpoint` | After `best.pth`, periodic, or `latest.pth` is written |

Set `state.stop_training = True` in a callback (e.g. early stopping) to exit after the current epoch. Lower `TrainCallback.order` runs first when multiple callbacks are registered.

```python
from segmentation_tools_pt.train import TrainCallback, train

class EarlyStopping(TrainCallback):
    def __init__(self, patience=5):
        self.patience, self.wait, self.best = patience, 0, 0.0

    def on_epoch_end(self, state):
        score = state.val_metrics[state.best_metric]
        if score > self.best:
            self.best, self.wait = score, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                state.stop_training = True

history = train(..., callbacks=[EarlyStopping(patience=10)])
```

#### Default monitoring callbacks

`segpt-train` enables the monitoring callbacks below via the `monitoring:` section in `train.yaml`. `ErrorAnalysisCallback` is opt-in (default off):

| Callback | Output | Purpose |
|---|---|---|
| `ActivationStatsCallback` | `checkpoints/activation_stats/epochNNN/layerXX_*_color_dim.png` | fastai-style activation histograms; bright band at bin 0 → dying features |
| `PredOverlayCallback` | `checkpoints/predictions/epochNNN/val_batchN_{labels,pred,compare}.jpg` | YOLO-style GT/pred overlay mosaics each epoch |
| `ErrorAnalysisCallback` (opt-in) | `checkpoints/error_analysis/{split}_top{k}[_class].jpg` + `top_ranked.json` | post-training worst-sample triage at train end: 4-column posters (image \| GT overlay α=0.1 \| pred overlay \| FN/FP difference) ranked by `rank_by: loss` (highest first) or a pixel metric (lowest first); multilabel `per_class: true` writes one poster per class head |
| `EncoderDriftCheck` (opt-in) | stdout verdict at train end (no files) | answers "did the encoder weights change?": bit-identical digest check before/after training; per-layer cosine report on drift; raises when `expect_frozen=true` (derive from `freeze_epochs: -1`) |

Disable the first two with `monitoring.activation_stats.enabled=false` or `monitoring.pred_overlays.enabled=false`. Custom callbacks can still be passed from a notebook via `train(..., callbacks=[...])`.

```yaml
monitoring:
  activation_stats:
    enabled: true
    every: 1
    n_bins: 40
    hist_range: [0, 10]
    max_layers: 6
  pred_overlays:
    enabled: true
    every: 1
    max_val_batches: 3
    jpeg_quality: 90
  error_analysis:
    enabled: false          # post-training worst-sample posters; default off
    top_k: 12
    rank_by: loss           # loss | iou | dice | recall | precision
    splits: [train, val]
    alpha: 0.1
    jpeg_quality: 90
    per_class: true
  encoder_drift_check:
    enabled: false        # verify encoder weights unchanged after training; default off
    expect_frozen: null   # null = derive from train.freeze_epochs == -1; true = raise on drift; false = report only
```

### `inference.py` — sliding-window inference
`predict_full_image()` — tiles a large image into overlapping patches, runs the model,
and stitches predictions back with Gaussian-weighted blending.
Handles arbitrary image sizes without retraining.

### `visualize.py` — visualization helpers
`overlay_mask()`, `plot_history()` — for notebooks and post-training inspection.

### `export.py` — ONNX export
`export_onnx()` — traces the model to ONNX with a configurable opset,
then verifies the output matches PyTorch within tolerance.
Called automatically after training if `export.onnx: true`.

### `cli.py` — installed CLI commands
Three entry points registered in `pyproject.toml`:

| Command | What it does |
|---|---|
| `segpt-build` | Runs `scripts/build.py` via Hydra; use `configs/build.yaml` (slim, CPU-friendly) |
| `segpt-train` | Runs `scripts/train.py` via Hydra; use `configs/train.yaml` |
| `segpt-submit` | Submits a build or training job to a remote HPC cluster via LSF `bsub`; reads `hpc:` from config |

Build and train share `data` + `patch` settings via Hydra `defaults: [data_patch]` in both YAML files.

---

## 5. The two training modes

### Patch mode (`data.patch_mode: true`, default)

```
raw_dir/
├── images/   →  extract patches  →  patch_dir/train/images/
└── masks/    →  extract patches  →  patch_dir/train/masks/
```

Use when:
- Images are large (X-ray panels, full wafer maps) and don't fit in GPU memory
- You want a reusable patch cache shared across many runs
- You need hard-example oversampling (sparse-foreground patch weights)

### Direct mode (`data.patch_mode: false`)

```
raw_dir/images/  →  load full image  →  random crop at train time
raw_dir/masks/   →  load full mask   →  center crop at val time
```

Use when:
- Images are already small enough after cropping
- Quick iteration / prototyping without a patch build step

---

## 6. End-to-end data flow (patch mode)

```
raw_dir/
├── images/sample_001.png        (grayscale or multi-channel)
└── masks/sample_001.png         (binary: {0,255} or {0,1})

        ↓  discover_samples()
        ↓  fit_rotated_rects_mask()  (optional)
        ↓  build_patch_dataset()

patch_dir/
├── train/images/ *.png
├── train/masks/  *.png
├── val/images/   *.png
├── val/masks/    *.png
├── metadata.json                (patch positions, fg counts)
└── stats.json                   (channel mean/std for normalization)

        ↓  SegmentationDataset + DataLoader
        ↓  train()

outputs/YYYY-MM-DD/HH-MM-SS/
├── checkpoints/
│   ├── best.pth
│   ├── latest.pth
│   ├── history.json
│   └── model.onnx               (if export.onnx: true)
└── .hydra/config.yaml           (exact config used)
```

---

## 7. Config files

Two Hydra configs share a common base (`configs/data_patch.yaml`):

| File | Used by | Contains |
|---|---|---|
| `data_patch.yaml` | (included via `defaults`) | `data.*`, `patch.*`, `train.num_classes`, `model.in_channels` |
| `build.yaml` | `segpt-build` | shared base + CPU `hpc:` defaults |
| `train.yaml` | `segpt-train` | shared base + model, loss, augment, metrics, export, GPU `hpc:` |

Copy `build_example.yaml` and `train_example.yaml` for a new project. Edit `raw_dir` / `patch_dir` in either file — both stay aligned when they include the same `data_patch` defaults.

Training config excerpt (`train.yaml` adds training-only sections on top of `data_patch`):

```yaml
defaults:
  - data_patch
  - _self_

model:
  architecture: unet               # smp head: unet | fpn | deeplabv3plus | ...
  arch: mobilenet_v2               # any SMP-supported encoder / backbone
  in_channels: 3                   # must match data_mode (single→1, single_repeat_3→3, etc.)
  pretrained: true
  weights_path: /path/to/mobilenet_v2.pth  # local .pth for air-gapped HPC; null = download

train:
  num_classes: 1                   # 1 = binary sigmoid; >1 = softmax
  epochs: 60
  batch_size: 16
  encoder_lr: 1.0e-4
  decoder_lr: 1.0e-3
  freeze_epochs: 5 # 0 = no freeze; N > 0 = unfreeze at epoch N; -1 = encoder frozen forever
  crop_size: 256
  use_amp: true
  resume: true
  resume_from: null                # null = latest.pth; set to specific .pth to resume from there
  device: cuda
  num_workers: 4                   # DataLoader workers; needs TMPDIR=/tmp on HPC (set in job script)

loss:                              # Tversky+Dice+boundary, phase-scheduled
  alpha: 0.7                       # FN weight — higher = fewer misses
  beta:  0.3                       # FP weight
  phases:
    - {until_epoch: 10, w_tversky: 0.6, w_dice: 0.4, w_boundary: 0.0}
    - {until_epoch: 35, w_tversky: 0.5, w_dice: 0.3, w_boundary: 0.2}
    - {until_epoch: -1, w_tversky: 0.4, w_dice: 0.3, w_boundary: 0.3}

metrics:
  threshold: 0.5
  best_metric: iou                 # iou | dice | recall | precision

augment:
  preset: yolo26m
  fliplr: 0.0                      # set 0 for orientation-sensitive X-ray
  flipud: 0.0
  batch:                           # batch-level augs: YOLO-style pick-one (at most one type per batch)
    enabled: false                 # master switch; false or all probs 0 = no batch aug at all
    mosaic: 0.0
    mixup: 0.0
    cutmix: 0.0
    copy_paste: 0.0
    copy_paste_mode: flip

monitoring:
  activation_stats:
    enabled: true
    every: 1
    n_bins: 40
    hist_range: [0, 10]
    max_layers: 6
  pred_overlays:
    enabled: true
    every: 1
    max_val_batches: 3
    jpeg_quality: 90
  error_analysis:
    enabled: false        # post-training worst-sample posters; default off (existing runs unchanged)
    top_k: 12
    rank_by: loss
    splits: [train, val]
    alpha: 0.1
    jpeg_quality: 90
    per_class: true
  encoder_drift_check:
    enabled: false        # verify encoder weights unchanged after training; default off
    expect_frozen: null   # null = derive from train.freeze_epochs == -1; true = raise on drift; false = report only

export:
  onnx: false                      # requires onnx + onnxruntime installed in the venv
  opset: 18
  verify: true

hpc:                               # used only by segpt-submit
  queue: gpu                       # gpu | gpu_compute | normal
  device: gpu
  num_gpus: 1
  gpu_mem_gb: 24
  num_cpus: 8
  mem_gb: 200
  interactive: true
  env_activate: "source .venv/bin/activate"
  repo_dir: "/path/to/segmentation_tools_pt"
```

Any key can be overridden on the command line:

```bash
segpt-train configs/train.yaml train.epochs=80 metrics.best_metric=recall
```

---

## 8. CLI commands

### `segpt-build`

```bash
segpt-build configs/build.yaml
segpt-build configs/build.yaml data.force_rebuild=true patch.patch_size=512
segpt-submit configs/build.yaml
```

### `segpt-train`

```bash
segpt-train configs/train.yaml
segpt-train configs/train.yaml train.epochs=80 data.patch_mode=false
```

Full guide: `docs/TRAINING_CLI.md`

### `segpt-submit` (HPC only)

Submits to LSF cluster via `bsub`. Reads `hpc:` section from config.

```bash
# Patch build (CPU — use build.yaml)
segpt-submit configs/build.yaml

# Training (GPU)
segpt-submit configs/train.yaml
segpt-submit configs/train.yaml hpc.queue=gpu_compute hpc.host=gpu-node-01

# Dry run
segpt-submit configs/build.yaml --dry-run
segpt-submit configs/train.yaml --dry-run
```

Example GPU queue names:

| Queue | GPUs | Notes |
|---|---|---|
| `gpu` | Multi-GPU shared nodes | Filter by `gmem` (e.g. 24G) |
| `gpu_compute` | Big dedicated nodes (40G+) | Pin to host with `hpc.host=gpu-node-01` |

---

## 9. Installation and environment

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency management.

```bash
# Clone and set up (first time)
git clone <repo>
cd segmentation_tools_pt
uv sync               # creates .venv/, installs all deps + CLI commands

# Activate the environment
source .venv/bin/activate

# After adding a new dependency to pyproject.toml
uv lock               # regenerate uv.lock (run once, commit the lockfile)
uv sync               # install the updated env
```

`uv lock` vs `uv sync`:
- `uv lock` — resolves and pins all versions; run when you **change** `pyproject.toml`
- `uv sync` — installs from the existing `uv.lock`; run when you **set up** the environment

On HPC the job script runs `uv sync --quiet` automatically before starting training.

---

## 10. Expected data layout

```
raw_dir/
├── images/
│   ├── sample_001.png      ← grayscale or multi-channel source image
│   └── sample_002.png
└── masks/
    ├── sample_001.png      ← binary mask, same stem (mask_suffix: "")
    └── sample_002.png
```

For two-channel data (`data_mode: two_channel`, `mask_suffix: "_mask"`):

```
raw_dir/
├── images/
│   ├── id_ch0.png
│   ├── id_ch1.png
└── masks/
    └── id_mask.png
```

---

## 11. Coding conventions

1. **Notebook first** — explore in a notebook cell, confirm it works, then wrap in a function.
   One function = one clear responsibility.
2. **Test in the same notebook** — every exported function has a test cell directly below it
   using real or synthetic data. Tests are cheap; they run via `nbdev_test`.
3. **Docstring format** — fastai/nbdev style:
   ```python
   def my_fn(x: int) -> str:
       "One-line summary. No 'Returns:' header needed."
       ...
   ```
4. **No comments that narrate code** — comments explain *why*, not *what*.
5. **Progressive complexity** — simple implementation first; add complexity only when needed.
6. **`__all__`** — every exported module defines `__all__` explicitly at the top.

---

## 12. Key design decisions and trade-offs

| Decision | Rationale |
|---|---|
| nbdev over plain Python | Exploration + tests + docs in one place; critical for industrial code that must be maintainable long-term |
| Hydra for config | Hierarchical overrides, run reproducibility (`.hydra/` saves exact config), composable configs |
| Phase-scheduled loss | Coarse Dice early → boundary sharpening late; works better than a static loss for irregular industrial defects |
| YOLO26 augmentation presets | Consistent baseline; easy to tighten (set flips=0 for fixed-camera X-ray) or loosen |
| Two training modes | Patch cache for large images + hard-example oversampling; direct mode for fast prototyping |
| `segpt-submit` in Python | Shell scripts break; Python reads the same YAML config the trainer uses → single source of truth |
| `uv` over conda | Faster, reproducible lockfile, no channel conflicts, works in air-gapped environments with Artifactory mirror |
| `TMPDIR=/tmp` in job script and Python | HPC `.bashrc` sets TMPDIR to NFS; NFS doesn't support POSIX shared memory → DataLoader workers crash. Setting it after `.bashrc` and again in Python ensures local disk is always used |
| Patch copy to `/tmp` at job start | NFS patch reads cause long GPU idle gaps and trigger HPC watchdog (< 5 % GPU util for 5 min). Copying once to local SSD makes reads ~10× faster and keeps GPU utilisation high |
| Large patch sets: pre-stage on scratch | For datasets with hundreds of thousands to millions of patches, a `/tmp` copy per job is impractical (too slow or disk too small). Pre-copying to a shared scratch filesystem once and setting `data.patch_dir` to that path avoids the per-job copy entirely — all runs reuse the same fast-path data |

---

## 13. Common workflows

### Before starting any change: check for existing uncommitted work

This repo lives on a shared network mount and is often worked on across multiple sessions —
by you, a teammate, or an AI agent. **Before re-reading source to verify or re-implement
something the user describes, run `git status --short` and `git diff --stat` first** (and
check the top of `CHANGELOG.md`, which is often updated as intent documentation even before a
commit). It is common for a fix to already exist uncommitted from a prior session — treat that
as the starting point instead of re-deriving the same conclusions from scratch.

### Start a new segmentation project

```bash
cp configs/build_example.yaml /path/to/my_project/build.yaml
cp configs/train_example.yaml /path/to/my_project/train.yaml
# Edit data.raw_dir, data.patch_dir, model.arch, train.num_classes
segpt-build /path/to/my_project/build.yaml
segpt-train /path/to/my_project/train.yaml
```

### Add a new library feature

```bash
# Edit the relevant notebook in nbs/
# Run all cells to test
nbdev_export          # regenerate src/
nbdev_test            # confirm nothing broke
# Commit both nbs/ and src/
```

### Submit a training run to HPC

```bash
# Edit hpc: section in train.yaml (or override on CLI)
segpt-submit configs/train.yaml --dry-run   # verify first
segpt-submit configs/train.yaml             # submit
```

### Resume an interrupted run

```yaml
train:
  resume: true        # loads latest.pth from checkpoints/ automatically
  resume_from: null   # or set to a specific path, e.g. checkpoints/best.pth
```

### Resume from a specific checkpoint

```yaml
train:
  resume: true
  resume_from: checkpoints/periodic_ep049.pth
```

### Export a trained model to ONNX manually

```python
from segmentation_tools_pt.export import export_onnx
export_onnx('checkpoints/best.pth', 'model.onnx', in_channels=1, num_classes=1)
```

### Large patch datasets (hundreds of thousands to millions of patches)

The default `hpc/submit.sh` copies `data.patch_dir` to `/tmp` on the compute node at
job start. This works well for small-to-medium patch caches (up to ~20 GB) but is
impractical for very large ones. Use one of these strategies instead:

| Situation | Strategy |
|---|---|
| Scratch filesystem available (`/scratch`, `/work`) | Copy once manually → set `data.patch_dir` to the scratch path in `train.yaml`. The job script skips the copy because the path is already local. All jobs reuse it. |
| Scratch not available, dataset just about fits in `/tmp` | Increase `hpc.walltime` to absorb copy time; use `hpc.mem_gb` large enough to hold the data |
| Too large for any local disk | Keep `data.patch_dir` on NFS, set `train.num_workers: 6` or higher to hide latency, monitor GPU util with `bpeek <JOBID>` |
| Very large + fast iteration needed | Switch to `data.patch_mode: false` (direct mode) — loads full images with on-the-fly crop, no patch cache at all |

Full guidance: `hpc/README.md` → "Large patch datasets" section.

## Agent skills

### Issue tracker

GitHub Issues are the official board (`gh` CLI); `.scratch/` is for private drafts only. See `docs/agents/issue-tracker.md`.

### Triage labels

Default roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.

### nbdev implementation

All feature work: literate `nbs/` → test steps → `nbdev-export` (never hand-edit `src/`). See `docs/agents/nbdev-implementation-contract.md`.

## Learned User Preferences

- Treat notebooks as literate programs (Knuth/nbdev): narrative + bullet plan first; then **experiment → verify** each step on synthetic data; only after every verify passes promote to `#| export` (dump-and-export is not literate). Tests sit as footnotes under the export. `/implement` (and TDD/code-review) still apply, but never skip this experiment→verify→function order — see `docs/agents/nbdev-implementation-contract.md`.
- Prefer plain language over jargon; define unfamiliar terms briefly when first used (e.g. avoid unexplained words like “canonical”).
- Prefer a wayfinder decision map before writing a design spec; ship phased tickets (umbrella/design first, then one implementable slice); finish one map through implement before starting the next; document deferred/out-of-scope items in the map or spec for later; create map/ticket/spec via the repo’s wayfinder flow rather than waiting indefinitely on grill answers.
- Prefer reviewing draft agent-config files before they are written to disk.
- Prefer GitHub Issues as the official tracker (`gh`); keep `.scratch/` for private drafts only.
- Prefer the default Matt Pocock triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
- On wayfinder grill questions (A/B/C), default to the agent’s expert recommendation for every question unless they explicitly override.
- Include post-training error analysis in wayfinder/planning (grill-with-docs or related tickets), not only training and export.
- Prefer config-only control so any segmentation task works by editing Hydra YAML without code changes (including class/head names and optional inference post-processing on/off).
- Prefer independently toggleable batch-level augmentations in config (mixup, cutmix, mosaic, copy_paste, alone or combined).
- Prefer default ImageNet encoder weights with freeze warmup, plus a config path to keep the encoder frozen for the full run.
- When changing library behavior in notebooks, also update related agent skills so non-power-users can ask the agent and get working scripts.

## Learned Workspace Facts

- DINOv3 ViT-S/16 and B/16 are available via `get_model()` as `dinov3_vit_small` / `dinov3_vit_base`; curated aliases `convnext_tiny` / `efficientnetv2_s` (and raw `tu-convnext_tiny` / `tu-tf_efficientnetv2_s`) are first-class in `get_model` / `list_smp_encoders`; Hydra packages live under `configs/model/`.
- Mask config is resolved with `resolve_mask_cfg(cfg)` → `{mask_mode, mask_channels, class_names}`; `validate_mask_mode(cfg)` validates companions and returns `(mask_mode, mask_channels, class_names)`.
- This is a general segmentation library: class/head names are config-driven; Chip/Void/LiftedChip is an example multi-head use case, not hard-coded product vocabulary. Active focus also includes MatriX↔SAKI domain shift, post-training error analysis, and a config-selectable SMP vs custom UNet factory (ASPP/attention knobs).
- GitHub remote for this repo is `HasanGoni/segmentation_tools_pt`.
- FP32 ONNX export works for `dinov3_vit_small` (and raw `tu-*` Unets) at `in_channels=1`, multilabel `num_classes=3`; `onnx`/`onnxruntime` are optional/commented in `pyproject.toml`; INT8/TensorRT not validated.
- Shared nbdev implementation contract lives at `docs/agents/nbdev-implementation-contract.md`; wayfinder map for multi-head encoder/robustness is tracked on GitHub Issues.
- Custom UNet/SMP factory design lives at `docs/superpowers/specs/2026-10-01-custom-unet-smp-factory-design.md` (`model.source: smp | custom_unet`; ASPP/attention are v1 options, default off).
- Multi-head robustness design lives at `docs/superpowers/specs/2026-10-01-multi-head-robustness-design.md`.
- DINOv3 pretrained weights load via timm's hub inside `Dinov3ReassembleEncoder.__init__` (`pretrained=True` threaded through the SMP encoder registry); `encoder_weights=None` only switches off smp's state-dict loader, which would fail because the DINOv3 checkpoint has no keys for the added reassemble layers.
- `ErrorAnalysisCallback` was decided in closed issue #9: fires on `on_train_end`, config under `monitoring.error_analysis`, default off, val top-k JPEG/JSON; implemented and closed via issue #18 (fuller poster spec folded into it: 4-column image | GT overlay α≈0.1 | pred overlay | diff overlay; train+val; rank by loss or metric; configurable top-k).
- `losses.py` exports a loss stack newer than the Tversky+Dice+boundary combiner: `TverskyFocalLoss`, `ForeGroundAwareLoss`, `hierarchy_regularizer_loss`, `SegLoss`, `resolve_loss_weights`.
- Multi-head robustness map #1 progress: #15 (DINOv3 encoders), #18 (`ErrorAnalysisCallback`), #16 (`freeze_epochs: -1` freeze-forever, commit `793dfbb`), and #19 (class-targeted morph postprocess: `inference.apply_class_morph` + `hydra_pipeline.resolve_class_morph_cfg` + `postprocess.class_morph` config, default off) are implemented and closed; active work branch is `feat/dinov3-encoder-decoder`; remaining `ready-for-agent` tickets: #20 (FP32 ONNX + batched tile inference).
