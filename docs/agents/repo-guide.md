# Repo Guide — deep detail behind AGENTS.md

This file holds the module-level gotchas, config walkthroughs, and long workflows that
`AGENTS.md` links to instead of copying (AGENTS.md is auto-loaded into every agent session
and must stay lean; this file is read on demand).

Sources of truth: for CLI behavior `docs/TRAINING_CLI.md`, for HPC `hpc/README.md`, for
config values `configs/*.yaml` — this guide explains and points, it does not override them.

---

## 1. Module-by-module guide

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
- DataLoader prefetch via `prefetch_factor` (threaded into the internal patch-mode loaders from `train.prefetch_factor`; inert unless `num_workers > 0`)
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

### `inference.py` — sliding-window inference
`predict_full_image()` — tiles a large image into overlapping patches, runs the model,
and stitches predictions back with Gaussian-weighted blending.
Handles arbitrary image sizes without retraining.
`predict_tiled()` — batched sliding-window inference: tiles forward in chunks of
`inference_batch_size` (memory ↔ throughput knob; batch 1 is bit-identical to any
larger batch), clamped tile starts cover every pixel including images smaller than
one tile, same activation semantics as `predict`. Production knobs (`tile_size`,
`overlap`, `inference_batch_size`, `prefetch_factor`) resolve from the `inference:`
config block via `hydra_pipeline.resolve_inference_cfg`.

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

## 2. Config files — annotated walkthrough

Two Hydra configs share a common base (`configs/data_patch.yaml`):

| File | Used by | Contains |
|---|---|---|
| `data_patch.yaml` | (included via `defaults`) | `data.*`, `patch.*`, `train.num_classes`, `model.in_channels` |
| `build.yaml` | `segpt-build` | shared base + CPU `hpc:` defaults |
| `train.yaml` | `segpt-train` | shared base + model, loss, augment, metrics, export, GPU `hpc:` |

Copy `build_example.yaml` and `train_example.yaml` for a new project. Edit `raw_dir` / `patch_dir` in either file — both stay aligned when they include the same `data_patch` defaults.

The annotated training-config excerpt below mirrors `configs/train.yaml` — **the YAML file is
the source of truth**; if this walkthrough and the file disagree, trust the file.

```yaml
defaults:
  - data_patch
  - _self_

model:
  architecture: unet      # smp head: unet | fpn | deeplabv3plus | ...
  arch: mobilenet_v2      # any SMP-supported encoder / backbone
  in_channels: 3          # must match data_mode (single→1, single_repeat_3→3, etc.)
  pretrained: true
  weights_path: /path/to/mobilenet_v2.pth  # local .pth for air-gapped HPC; null = download
  # source: smp | custom_unet  (library-owned random-init UNet; options in docs/TRAINING_CLI.md, "Custom UNet")

train:
  num_classes: 1          # 1 = binary sigmoid; >1 = softmax
  epochs: 60
  batch_size: 16
  encoder_lr: 1.0e-4
  decoder_lr: 1.0e-3
  freeze_epochs: 5        # 0 = no freeze; N > 0 = unfreeze at epoch N; -1 = encoder frozen forever
  crop_size: 256
  use_amp: true
  resume: true
  resume_from: null       # null = latest.pth; set to specific .pth to resume from there
  device: cuda
  num_workers: 4          # DataLoader workers; needs TMPDIR=/tmp on HPC (set in job script)
  prefetch_factor: null   # batches in flight per worker (needs num_workers > 0); null = torch default

loss:                     # Tversky+Dice+boundary, phase-scheduled
  alpha: 0.7              # FN weight — higher = fewer misses
  beta: 0.3               # FP weight
  phases:
    - {until_epoch: 10,  w_tversky: 0.6, w_dice: 0.4, w_boundary: 0.0}
    - {until_epoch: 35,  w_tversky: 0.5, w_dice: 0.3, w_boundary: 0.2}
    - {until_epoch: -1,  w_tversky: 0.4, w_dice: 0.3, w_boundary: 0.3}

metrics:
  threshold: 0.5
  best_metric: iou        # iou | dice | recall | precision

augment:
  preset: yolo26m
  fliplr: 0.0             # set 0 for orientation-sensitive X-ray
  flipud: 0.0
  batch:                  # batch-level augs: YOLO-style pick-one (at most one type per batch)
    enabled: false        # master switch; false or all probs 0 = no batch aug at all
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
    enabled: false       # post-training worst-sample posters; default off (existing runs unchanged)
    top_k: 12
    rank_by: loss        # loss | iou | dice | recall | precision
    splits: [train, val]
    alpha: 0.1
    jpeg_quality: 90
    per_class: true
  encoder_drift_check:
    enabled: false       # verify encoder weights unchanged after training; default off
    expect_frozen: null  # null = derive from train.freeze_epochs == -1; true = raise on drift; false = report only

export:
  onnx: false            # requires onnx + onnxruntime installed in the venv
  opset: 18
  verify: true

inference:               # production tiled-inference knobs (predict_tiled)
  tile_size: 256         # tile side fed to the model
  overlap: 32            # adjacent-tile overlap; Gaussian-weighted stitching
  inference_batch_size: 8  # tiles per forward — memory ↔ throughput only
  prefetch_factor: null  # loader-driven tiled inference only

hpc:                     # used only by segpt-submit
  queue: gpu             # gpu | gpu_compute | normal
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

## 3. Expected data layouts

Single/multi-file grayscale (`mask_suffix: ""`):

```
raw_dir/
├── images/
│   ├── sample_001.png          ← grayscale or multi-channel source image
│   └── sample_002.png
└── masks/
    ├── sample_001.png          ← binary mask, same stem
    └── sample_002.png
```

Two-channel data (`data_mode: two_channel`, `mask_suffix: "_mask"`):

```
raw_dir/
├── images/
│   ├── id_ch0.png
│   └── id_ch1.png
└── masks/
    └── id_mask.png
```

---

## 4. Workflows

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
nbdev_export    # regenerate src/
nbdev_test      # confirm nothing broke
# Commit both nbs/ and src/
```

### Submit a training run to HPC

```bash
# Edit hpc: section in train.yaml (or override on CLI)
segpt-submit configs/train.yaml --dry-run    # verify first
segpt-submit configs/train.yaml              # submit
```

Example GPU queue names:

| Queue | GPUs | Notes |
|---|---|---|
| `gpu` | Multi-GPU shared nodes | Filter by `gmem` (e.g. 24G) |
| `gpu_compute` | Big dedicated nodes (40G+) | Pin to host with `hpc.host=gpu-node-01` |

Full submission guide and troubleshooting: `hpc/README.md`. Detailed CLI guide: `docs/TRAINING_CLI.md`.

### Resume an interrupted run

```yaml
train:
  resume: true       # loads latest.pth from checkpoints/ automatically
  resume_from: null  # or set to a specific path, e.g. checkpoints/best.pth
```

### Resume from a specific checkpoint

```yaml
train:
  resume: true
  resume_from: checkpoints/periodic_ep049.pth
```

### Export a trained model to ONNX manually

`export_onnx` takes a loaded `nn.Module` (not a checkpoint path) — build the model with
`get_model(...)`, load the state dict from `checkpoints/best.pth`, then export:

```python
from segmentation_tools_pt.export import export_onnx
model.load_state_dict(
    torch.load('checkpoints/best.pth', map_location='cpu', weights_only=False)['model'])
export_onnx(model, 'model.onnx', in_channels=1, input_size=256)
```

(The library loads its own checkpoints with `weights_only=False` throughout — see
`train.py` / `model.py`; pass `weights_only=True` for untrusted files.)

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