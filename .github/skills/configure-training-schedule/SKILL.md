---
name: configure-training-schedule
description: 'Set the training loop hyperparameters for segmentation_tools_pt: epochs, batch size, encoder/decoder LR, freeze epochs, AMP, checkpoint cadence, and resume. Use when a user asks "how many epochs?", "learning rate?", "resume from a checkpoint", or is editing the train.* block. Emits the train.* block for train.yaml.'
argument-hint: '[optional] dataset size and compute budget'
user-invocable: true
disable-model-invocation: false
---
# Configure Training Schedule

Set the `train.*` loop knobs. Sensible defaults exist in `configs/train.yaml`; adjust to data size
and compute.

## Keys

| Key | Meaning / guidance |
|---|---|
| `train.epochs` | Total epochs (default 100). Align with `loss.phases` boundaries |
| `train.batch_size` | Fit to GPU memory; lower if OOM |
| `train.encoder_lr` / `train.decoder_lr` | Encoder lower (`1e-4`) than decoder (`1e-3`) — pretrained backbone |
| `train.freeze_epochs` | `0` = no freeze (default `5` = warm-up); `N > 0` unfreezes at epoch `N`; `-1` = encoder frozen forever (decoder + head only) |
| `train.crop_size` | Model-view crop; `<= patch.patch_size` in patch mode (see choose-tiling-strategy) |
| `train.use_amp` | Mixed precision; keep `true` on GPU |
| `train.save_every` | Periodic checkpoint cadence (`periodic_epNNN.pth`) |
| `train.num_workers` | DataLoader workers; needs `TMPDIR=/tmp` on HPC |
| `train.prefetch_factor` | Batches in flight per DataLoader worker; needs `num_workers > 0` (inert otherwise); `null` = torch default (2/worker) |
| `train.device` | `cuda` / `cpu` |

## Resume

- `train.resume: true` + `train.resume_from: null` → resume from `latest.pth`.
- `train.resume_from: checkpoints/periodic_ep049.pth` → resume from a specific checkpoint
  (raises `FileNotFoundError` immediately if the path is missing).

## What this skill emits

- The `train:` block for `configs/train.yaml`; CLI overrides such as
  `segpt-train configs/train.yaml train.epochs=80 train.batch_size=8`.

## Library anchors

- `src/segmentation_tools_pt/train.py` — `train`, `train_one_epoch`, `validate`.
- Source notebook: `nbs/10_train.ipynb`. Detailed guide: `docs/TRAINING_CLI.md`.
