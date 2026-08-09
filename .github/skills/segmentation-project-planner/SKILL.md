---
name: segmentation-project-planner
description: 'Entry-point interview that plans a full segmentation run for segmentation_tools_pt. Use when a user says "I need a segmentation script/model", "start a new segmentation project", "help me set up training", or is unsure which tiling / loss / augmentation / model options to pick. Interviews the user, then routes to the choose-* sub-skills and emits matched configs/build.yaml + configs/train.yaml and the segpt-build / segpt-train / segpt-submit commands.'
argument-hint: '[optional] one-line problem description (task, image size, defect type)'
user-invocable: true
disable-model-invocation: false
---
# Segmentation Project Planner

Plan a complete segmentation run for **`segmentation_tools_pt`** by interviewing the user, then
emitting a matched pair of Hydra configs (`build.yaml` + `train.yaml`) and the CLI commands to run
them. This is the orchestrator: it does not decide every knob itself — it routes each decision to a
focused sub-skill and stitches the answers into one runnable plan.

## Answer as two expert panels

Every recommendation is cross-checked by two panels, and you state which panel drove each call:

- **🧠 Deep-learning segmentation panel** — U-Net/SMP encoder–decoder, loss scheduling, augmentation,
  patch vs direct training, metrics, HPC.
- **🔬 Classical computer-vision panel** — mask geometry/cleanup, thresholding/morphology baselines,
  hand-crafted feature channels (tophat/sobel). This library is DL-first, so the classical panel
  will often say *"not implemented yet — here is where it would live"* (see placeholder skills).

## When to use

- The user is starting a new segmentation task and wants a working config, not raw theory.
- The user is unsure whether to tile, which loss to combine, or how much augmentation to use.
- The user wants a single plan that maps their problem to `segpt-build` / `segpt-train`.

## The interview (ask in this order; skip any the user already answered)

1. **Task type** — binary defect (1 class, sigmoid) or multi-class label map (`>1`, softmax)?
   → sets `train.num_classes`.
2. **Source image size & channels** — how large are the raw images, grayscale or multi-file?
   → routes to **choose-channels** (`data.data_mode`, `model.in_channels`) and informs tiling.
3. **Defect sparsity & size** — tiny sparse defects, or large regions filling the frame?
   → drives **choose-loss-function** (Tversky vs BCE) and **choose-tiling-strategy** (oversampling).
4. **Orientation sensitivity** — is the imaging geometry fixed (e.g. fixed-camera X-ray)?
   → drives **choose-augmentation** (flips on/off).
5. **Compute target** — local GPU or a shared LSF/Slurm cluster? which queue / GPU memory?
   → sets `train.batch_size` / `train.crop_size` feasibility and the `hpc.*` submission block.
6. **Pretrained weights availability** — internet or air-gapped node?
   → routes to **choose-model-architecture** (`model.weights_path`, `download_encoder_weights`).

## Routing table (delegate each decision)

| Question resolves to | Sub-skill | Emits into |
|---|---|---|
| Tile or not; patch/crop sizes | `choose-tiling-strategy` | `data.patch_mode`, `patch.*`, `train.crop_size` |
| Which loss + phase schedule | `choose-loss-function` | `loss.*` |
| How much augmentation | `choose-augmentation` | `augment.*` |
| Channel assembly | `choose-channels` | `data.data_mode`, `model.in_channels` |
| Head + encoder + weights | `choose-model-architecture` | `model.*` |
| Metric to optimise, threshold | `configure-metrics-checkpointing` | `metrics.*` |
| Epochs / lr / freeze / resume | `configure-training-schedule` | `train.*` |
| Live monitoring | `training-monitoring-callbacks` | `monitoring.*` |
| Mask cleanup before build | `mask-preprocessing` | `data.fit_rects` (+ placeholders) |
| Full-image prediction | `run-inference` | post-training |
| ONNX deployment | `export-onnx` | `export.*` |

## What this skill emits

1. A short problem restatement + the panel calls made.
2. A copy of `configs/build_example.yaml` → `<project>/build.yaml` with `data.raw_dir`,
   `data.patch_dir`, `data.data_mode`, and `patch.*` filled from the interview.
3. A copy of `configs/train_example.yaml` → `<project>/train.yaml` with `model.*`, `loss.*`,
   `augment.*`, `metrics.*`, `train.*`, `monitoring.*`, `hpc.*` filled from the sub-skills.
4. The runnable commands:
   - `segpt-build <project>/build.yaml`
   - `segpt-train <project>/train.yaml`
   - `segpt-submit <project>/train.yaml --dry-run` (then without `--dry-run`) for HPC.
5. Any decision that hit a **placeholder** feature is called out explicitly with "not implemented
   yet" and the notebook where it would be added.

## Grounding (do not invent keys)

- Shared base config: `configs/data_patch.yaml` (included by both build + train via
  `defaults: [data_patch]`). Never duplicate `data.*`/`patch.*` across the two files.
- CLI entry points: `segpt-build`, `segpt-train`, `segpt-submit` (`src/segmentation_tools_pt/cli.py`).
- Any override works on the CLI, e.g. `segpt-train configs/train.yaml train.epochs=80 metrics.best_metric=recall`.

## Completion criteria

- `train.num_classes`, `data.data_mode`/`model.in_channels`, tiling, loss, and augmentation are all
  resolved (each by its sub-skill) and consistent (`train.crop_size <= patch.patch_size` in patch mode;
  `model.in_channels` matches `data.data_mode`).
- The user has two runnable configs and the exact CLI commands.
- Every unimplemented request is flagged, not silently dropped.
