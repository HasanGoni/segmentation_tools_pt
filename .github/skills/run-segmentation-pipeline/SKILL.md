---
name: run-segmentation-pipeline
description: 'Drive the whole segmentation_tools_pt pipeline from a folder of data with config files and the segpt-* commands. Use when a user says "here is the data", "train a model on this folder", "run the whole pipeline", "compare U-Net and RF-DETR", "what do I run next?", or hands over images plus masks and expects a trained, evaluated model. Covers segpt-build, segpt-train (U-Net variants or model.source=rfdetr), segpt-eval, inference (predict_tiled / predict_tiled_rfdetr), ONNX export and segpt-submit for LSF HPC, with the override syntax and the gotchas already hit.'
argument-hint: '[optional] path to the data folder and what the masks mean'
user-invocable: true
disable-model-invocation: false
---
# Run the segmentation pipeline end to end

Entry point when the user gives you data and wants a result. `segmentation-project-planner` interviews and
chooses knobs (it stops at two configs); this skill runs the chain and tells you what to do after each step.
Every knob lives in a YAML config, so work from a copy of an existing config, not from new code.

```
data folder -> segpt-build -> segpt-train -> segpt-eval -> segpt-predict -> (export-onnx | segpt-submit)
              (or all four stages from one run file: `segpt-run`, section 7)
              patch cache    checkpoints/   comparison     full-image masks
```

## 1. Ask for the minimum (skip what the user already said)

| Need | Why | Derive it by |
|---|---|---|
| Folder layout | `data.raw_dir` must be an **absolute** path with `images/` and `masks/` | `ls RAW/images | head`, `ls RAW/masks | head` |
| Mask naming | `data.mask_suffix` ("" when mask has the image stem) | compare one image and one mask name |
| Channels | `data.data_mode`: `single`, `single_to_3`, `two_channel`, `three_channel`; `model.in_channels` must match | `choose-channels` skill |
| Classes | binary, multi-class (exclusive) or **multilabel** (can overlap) | ask; overlap means multilabel |
| Class names | `data.class_names` is needed by `segpt-eval` and `segpt-predict` in **every** run (single class: `+data.class_names=[defect]`); multilabel also needs one `masks/<id>_<class>.png` per class | `ls RAW/masks` |
| Image size | `patch.patch_size` must be <= the image side (images of 128 px: patch 128); 256 for small images, 512 when defects/context are large; `train.crop_size` <= `patch.patch_size`, and set `patch.stride` = patch size for little data | `choose-tiling-strategy` skill |
| Compute | local GPU, or LSF HPC (queue, GPU memory, `hpc.repo_dir`) | ask |

## 2. Pick the config to copy (config-file-first)

| Task | Copy | Notes |
|---|---|---|
| binary / multi-class U-Net | `configs/train_example.yaml` (+ `configs/build_example.yaml`) | `train.num_classes` = K |
| multilabel U-Net | `configs/train_multilabel_3head.yaml` | has `mask_mode`, `class_names`, `task: multilabel`, `metrics.threshold` list. Override for other class sets |
| RF-DETR instance model | `configs/train_rfdetr_example.yaml` | `model.source: rfdetr`; presets `model=rfdetr_nano|small|medium` |

```bash
mkdir -p /abs/proj && cp -r configs /abs/proj/configs     # copy the whole folder: train_example.yaml includes data_patch.yaml and model/<name>.yaml
```

Edit `/abs/proj/configs/train_example.yaml` or leave it alone and put changes in overrides. Copying only the one YAML fails with "Could not find 'model/...'".
Fastest start when the user wants everything run: write a `segpt-run` file (section 7) that points `config:` at `/abs/proj/configs/train_example.yaml` and puts the overrides below in `overrides:`.

Rules for every command below:
- Absolute `data.raw_dir` and `data.patch_dir`. The train configs set `hydra.job.chdir: true`, so relative paths resolve from the run dir.
- `hydra.run.dir=/abs/runs/NAME` chooses the run directory (`.hydra/`, `checkpoints/`, `train.log`). Without it: `outputs/DATE/TIME`.
- No internet or no local weights file: the encoder presets (`configs/model/*.yaml`) carry a placeholder `weights_path: /path/to/models/....pth` and training fails with FileNotFoundError. Add `model.pretrained=false model.weights_path=null` (random-init encoder: fine for a first run, weaker than ImageNet weights) or download weights first (`docs/TRAINING_CLI.md` section 8).
- A key that is **not in the base config** needs `++` (`++data.mask_mode=per_class_files`); a plain override fails with "Could not override". Keys that exist take no prefix.
- `segpt-train configs/x.yaml key=value ...` takes the same overrides as the YAML. `--help` prints the config.

## 3. Build the patch cache (once per data + patch settings)

```bash
segpt-build /abs/proj/build.yaml data.raw_dir=/abs/RAW data.patch_dir=/abs/PATCHES patch.patch_size=512 patch.stride=512
```

Reuses an existing cache; add `data.force_rebuild=true` after changing patch settings or masks. `segpt-train` builds it too when missing, so this step is optional for small data and useful on HPC (CPU job).

## 4. Train

U-Net, binary:

```bash
segpt-train /abs/proj/train.yaml data.raw_dir=/abs/RAW data.patch_dir=/abs/PATCHES model=convnext_tiny \
  train.epochs=15 hydra.run.dir=/abs/runs/convnext_s42 train.seed=42
```

U-Net, multilabel (start from the 3-head preset; two classes shown):

```bash
segpt-train configs/train_multilabel_3head.yaml data.raw_dir=/abs/RAW data.patch_dir=/abs/PATCHES model=convnext_tiny \
  'data.class_names=[chip,lifted]' train.num_classes=2 'metrics.threshold=[0.5,0.5]' \
  patch.patch_size=512 patch.stride=512 train.crop_size=512 hydra.run.dir=/abs/runs/convnext_s42
```

Multilabel checklist (all four, or it fails or silently trains the wrong thing): `data.mask_mode=per_class_files`, `data.class_names`, `train.num_classes` = number of names, `metrics.threshold` a **list** of the same length. `train.hierarchy.lifted_subset_of_chip` only when one class is inside another; set it false otherwise.

RF-DETR (instance model, separate environment recommended):

```bash
uv sync --extra rfdetr                             # needs rfdetr>=1.12; 1.9.4 fails on 1-channel input
segpt-train configs/train_rfdetr_example.yaml data.raw_dir=/abs/RAW data.patch_dir=/abs/PATCHES \
  ++data.mask_mode=per_class_files '++data.class_names=[chip,lifted]' train.num_classes=2 \
  patch.patch_size=512 patch.stride=512 train.crop_size=512 train.epochs=15 train.batch_size=8 \
  hydra.run.dir=/abs/runs/rfdetr_small_s1
```

`model=rfdetr_small` alone is refused; `model.source=rfdetr` (set in the example) is the switch. RF-DETR writes `checkpoints/last_ema.pth` (not `best.pth`), has no resume and no ONNX export, and a 15-epoch Small run on the #106 data took 2,685 s on one GB10 GPU.

HPC (LSF): fill `hpc.repo_dir`, `hpc.queue` in the config (a placeholder `repo_dir` makes the job script exit), then
`segpt-submit /abs/proj/train.yaml --dry-run`, check the bsub command, rerun without `--dry-run`. `segpt-submit /abs/proj/build.yaml` submits the build job. Details: `submit-to-hpc` skill.

## 5. Evaluate: always with `segpt-eval`, never by eye

```bash
segpt-eval /abs/runs/convnext_s42 /abs/runs/rfdetr_small_s1 --heldout --objects-class lifted
segpt-eval /abs/runs/a_s42 /abs/runs/a_s43 /abs/runs/b_s42 /abs/runs/b_s43 --heldout   # seeds s<N> become one arm, mean +- spread
segpt-eval RUN --thresholds 0.3 0.5 0.9 --threshold 0.5            # one run, chosen thresholds; add --report-dir DIR for the review report
```

Read the table before trusting it: `n` is the number of runs (seeds) in the arm, and with only a handful of val source images `--heldout` has almost nothing to hold out (use at least ~10 source images); F1 1.000 with recall 0.000, or a chosen threshold at the end of the grid (0.9999), means the model predicts (almost) no foreground, not that it is perfect. Check with `segpt-predict` on a val image, then train longer or add data before comparing arms.

Rules: runs must share the **same val ids** (same cache and split) or the command fails; `--heldout` picks the threshold on one half of the source images and scores the other half, so a threshold is never tuned on the data it is scored on. U-Net thresholds are probabilities (strict `>`), RF-DETR thresholds are scores (inclusive `>=`, useful range 0.1 to 0.3); do not copy one model's threshold to the other. If a U-Net arm selects 0.9999 (the grid end), its chip IoU is not comparable; report it as grid-limited. After `git pull`, rerun `uv sync` so the `segpt-eval` entry point exists.

Decision guidance (issue #106 on the chip/lifted X-ray set): U-Net ConvNeXt-Tiny (held-out F1 0.968-0.971) beat RF-DETR Small (0.935-0.936), so the rule said no-go for building an instance pipeline. Do not tell a user RF-DETR is better without a `segpt-eval --heldout` table on their own data; run it as a challenger next to the best U-Net. Further reading: `model-comparison-ab-test`, `object-level-metrics`, `segmentation-error-analysis`, `build-review-report`.

## 6. Predict on new images

One command for U-Net and RF-DETR runs (reads the run's config, tiles at the training crop size, writes masks, overlays and `summary.json`):

```bash
segpt-predict /abs/runs/convnext_s42 /abs/new_images --out /abs/predictions            # U-Net: p > 0.5 by default
segpt-predict /abs/runs/rfdetr_small_s1 /abs/new_images --threshold 0.2                 # RF-DETR: score >= 0.2 by default
```

Thresholds are not interchangeable (probability vs detector confidence); take the one `segpt-eval --heldout` chose for that run. 8-bit images only. The Python below is for probabilities or custom post-processing.

U-Net (verified on a real run): rebuild the model from the run's own config, load `best.pth`, normalise with the cache's `stats.json`, tile at the training size.

```python
import json, torch
from omegaconf import OmegaConf
from segmentation_tools_pt.model import get_model_from_cfg
from segmentation_tools_pt.hydra_pipeline import resolve_inference_cfg
from segmentation_tools_pt.inference import predict_tiled

cfg = OmegaConf.load('/abs/runs/convnext_s42/.hydra/config.yaml')
model = get_model_from_cfg(cfg, in_channels=1, crop_size=int(cfg.train.crop_size))
model.load_state_dict(torch.load('/abs/runs/convnext_s42/checkpoints/best.pth', map_location='cpu', weights_only=True)['model_state'])
st = json.load(open(f'{cfg.data.patch_dir}/stats.json'))            # mean/std on the 0-1 scale
knobs = resolve_inference_cfg(cfg); knobs['tile_size'] = int(cfg.train.crop_size)
prob = predict_tiled(model, (img01 - st['mean'][0]) / st['std'][0], torch.device('cuda'), task=cfg.train.get('task'), **knobs)  # (K,H,W)
```

`img01` is a `(1, H, W)` float tensor in 0-1. Set `task` explicitly for multilabel. More in the `run-inference` skill. Without the cache `stats.json`, take it from the build log.

RF-DETR: `load_rfdetr_predictor(run)` then `predict_tiled_rfdetr(pred, image_chw, 'cuda', threshold=0.2, **knobs)` (raw 0-255 pixels, not normalised). **Tile at the training patch size** (`train.crop_size`; the default): half-size tiles dropped some patches from IoU 0.98 to 0.0.

## 7. One command for the whole chain: `segpt-run`

When the user agrees to run everything, write a run file (data, config copy, arms, seeds, `predict.images`) and run it; it calls the commands above, skips finished stages and resumes after a crash. See `docs/TRAINING_CLI.md` section 6c.

```bash
segpt-run /abs/run.yaml --dry-run     # check the plan first
segpt-run /abs/run.yaml               # same command again after a failure resumes at the failed stage
segpt-run /abs/run.yaml --force eval  # redo evaluation after changing eval.args
```

Put `+data.class_names=[...]` and, offline, `model.pretrained=false model.weights_path=null` in `overrides` (see section 4 rules). `--dry-run` only prints commands: it does not catch a missing config include or a placeholder weights path.

## 8. Optional

- ONNX: `export.onnx=true` in the train config, or `export-onnx` skill. U-Net only.
- Error analysis after training: `monitoring.error_analysis.enabled=true`, or the `segmentation-error-analysis` skill.

## Gotchas (all hit in practice)

1. Plain override of a key that is not in the config fails ("Could not override"): use `++key=value`.
2. Relative data paths break because the run changes into the run dir: use absolute paths.
3. Multilabel needs all four keys in section 4; a single float `metrics.threshold` with K>1 heads crashes at the first metric update (tensor size mismatch).
4. `segpt-eval` refuses runs with different val ids: build once, train every arm from the same `data.patch_dir` and `patch.*`.
5. RF-DETR: rfdetr>=1.12, `model.source=rfdetr`, `last_ema.pth`, scores not probabilities, tile at the training size.
6. `resolve_inference_cfg` also returns `prefetch_factor`; `predict_tiled` accepts and ignores it, pass the dict as is.
7. `best.pth` holds the weights under `model_state` (load with `weights_only=True`).
8. `train.crop_size` must not exceed `patch.patch_size`; for RF-DETR it is only checked against it.

## Completion criteria

- Data folder, classes, patch size and compute are fixed and written into configs the user can re-run.
- Build and train commands ran (or `segpt-submit --dry-run` was reviewed) with absolute paths and an explicit `hydra.run.dir`.
- A `segpt-eval` table exists for every run you recommend, and the claim matches the table (and its val-id fingerprint).
- Prediction was tried on at least one image with the same tile size as training.
