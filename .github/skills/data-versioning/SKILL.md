---
name: data-versioning
description: 'PLACEHOLDER skill — dataset versioning (DVC, deterministic splits, reproducible patch caches) is NOT implemented in segmentation_tools_pt yet. Use when a user asks "version my dataset", "reproducible splits", "DVC", or "track data changes". Explains conventions and where tooling would be added.'
argument-hint: '[optional] dataset size and collaboration needs'
user-invocable: true
disable-model-invocation: false
---
# Data Versioning — NOT IMPLEMENTED YET

Reproducible dataset versioning is **not implemented** in `segmentation_tools_pt`. Today, the best
you can do is deterministic splits via `patch.seed` and careful path management.

## What would be added

1. **Deterministic hashing** — hash `raw_dir` contents and store in `metadata.json`.
2. **Split versioning** — version `metadata.json` and `stats.json` so the same train/val patches can
   be reproduced.
3. **DVC integration** — `dvc add raw_dir`, `dvc add patch_dir`, store hashes in Git.
4. **Config checksum** — include a hash of `data.*` + `patch.*` in the patch cache to detect drift.

## Recommended conventions today

- Use `patch.seed` and `train.seed` for reproducible splits.
- Name patch directories with the patch parameters:
  `mydata_patches_ps632_st316_seed42`.
- Store the exact `train.yaml` used in `outputs/.../.hydra/config.yaml`.

## Where it would live

- New notebook: `nbs/24_data_versioning.ipynb` → `data_versioning.py`.
- Hook into `dataset_build.build_patch_dataset()` to write hashes.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
