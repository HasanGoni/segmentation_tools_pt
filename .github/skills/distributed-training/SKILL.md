---
name: distributed-training
description: 'PLACEHOLDER skill — multi-GPU / distributed data parallel (DDP) training is NOT implemented in segmentation_tools_pt yet. Use when a user asks "multi-GPU", "DDP", "distributed training", "torchrun", or "data parallel". Explains the PyTorch DDP path and where it would be added.'
argument-hint: '[optional] number of GPUs available'
user-invocable: true
disable-model-invocation: false
---
# Distributed Training (DDP) — NOT IMPLEMENTED YET

Training on multiple GPUs with PyTorch DistributedDataParallel (DDP) is **not implemented** in
`segmentation_tools_pt` today. `segpt-train` and `train()` are single-GPU only.

## What would be added

1. DDP wrapper around the model inside `train()` when `world_size > 1`.
2. `DistributedSampler` for the training DataLoader.
3. `torchrun` / `mp.spawn` launch support in `scripts/train.py`.
4. Config key: `train.distributed: true` and `train.world_size`.
5. Aggregate metrics across ranks before logging/saving.

## Where it would live

- New notebook: `nbs/21_distributed.ipynb` → `distributed.py`.
- Modify `src/segmentation_tools_pt/train.py` to initialize process group, wrap model, and use
  `DistributedSampler`.
- Update `hpc/submit.sh` to launch with `torchrun --nproc_per_node=...`.

## What the user can do today

- Run separate single-GPU experiments with different seeds/hyperparameters and ensemble the results
  (see `model-ensembling`).
- Use the biggest single GPU available via `hpc.queue=gpu_compute` and `hpc.gpu_mem_gb`.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
