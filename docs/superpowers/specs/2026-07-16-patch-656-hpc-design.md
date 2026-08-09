# Patch extraction script for example curated 8-bit grayscale dataset

Date: 2026-07-16
Author: AI assistant via brainstorming skill
Status: Approved for implementation

## Problem

User has grayscale industrial images at:

```text
/path/to/dataset/train/
├── images/
└── masks/
```

They want to cut them into 656×656 patches with ~50% overlap and submit the patch build job to the HPC cluster.

## Constraints

- Only a Python script (no Hydra config, no notebook).
- Runs on the HPC (Linux paths, LSF/bsub scheduler).
- Grayscale single-channel input (`data_mode: single`).
- Mask and image share the same filename stem (`mask_suffix: ""`).
- Binary segmentation (`num_classes: 1`).

## Design

### Script

`scripts/patch_656_example.py`

- Imports from the existing `segmentation_tools_pt` library:
  - `PatchConfig` from `segmentation_tools_pt.patches`
  - `build_patch_dataset` from `segmentation_tools_pt.dataset_build`
- Hardcodes the raw/output paths and the patch parameters listed below.
- Provides a normal run mode and an `--submit` mode that queues the script on HPC via `bsub`.

### Parameters

| Parameter | Value | Reason |
|---|---|---|
| `raw_dir` | `/path/to/dataset/train/` | User-provided source location |
| `patch_dir` | `/path/to/dataset/train_patches_656/` | Cache next to the raw data |
| `patch_size` | 656 | User request |
| `stride` | 328 | 50% overlap between neighbours |
| `data_mode` | `single` | Grayscale PNG → 1 channel |
| `mask_suffix` | `""` | Same stem as image |
| `num_classes` | 1 | Binary defect mask |
| `fit_rects` | True | Snap noisy annotations to clean rotated rectangles before patching |
| `min_fg_pixels` | 50 | Patches with fewer fg pixels are considered empty |
| `max_empty_frac` | 0.10 | Keep ~10% of empty patches for hard negatives |
| `n_jobs` | 8 | Parallel sample workers |
| `val_fraction` | 0.20 | 20% of source images held out for validation |

### HPC submission (`--submit`)

When `--submit` is passed, the script:

1. Builds an LSF `bsub` command with CPU-appropriate resources:
   - queue: `batch`
   - CPUs: 16
   - memory: 64 GB
   - walltime: 6:00
   - job name: `segpt_patch_656_example`
2. The bsub job script:
   - Sources `~/.bashrc`
   - Activates the repo `.venv`
   - Sets `TMPDIR=/tmp`
   - Runs `cd <repo_dir>` and `python scripts/patch_656_example.py`
3. Prints the submitted job ID and stdout/stderr log paths.

### Outputs

```text
train_patches_656/
├── train/
│   ├── images/   {sample_id}_y{cy}_x{cx}_ch0.png
│   └── masks/    {sample_id}_y{cy}_x{cx}_mask.png
├── val/
│   ├── images/
│   └── masks/
├── metadata.json
└── stats.json
```

### Error handling

- Raise a clear `FileNotFoundError` if `raw_dir` does not exist.
- Print a summary after the build: number of train/val patches, number of foreground patches, and the output directory.
- If `bsub` is not available and `--submit` is used, print a helpful error and the equivalent command to run manually.

## Alternatives considered

1. **Reuse `segpt-build` with a custom YAML config** — Rejected because the user explicitly asked for a Python script, not a Hydra-driven CLI.
2. **Embed the patch logic directly without importing the library** — Rejected because it duplicates tested code in `patches.py` and `dataset_build.py`.

## Implementation steps

1. Create `scripts/patch_656_example.py` with the parameters and `--submit` logic.
2. Run a local smoke test on a tiny synthetic directory if possible, or verify the script parses correctly.
3. Add a short usage note to the script docstring.
4. (Optional) Run or submit on HPC per user request.
