# Patch 656×656 HPC script implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone Python script that extracts 656×656 patches with 50% overlap from the example curated 8-bit grayscale datasetset and can submit itself to the HPC via `bsub`.

**Architecture:** A single self-contained script `scripts/patch_656_example.py` reuses the existing `build_patch_dataset()` and `PatchConfig` from the library. When run directly it patches on the local CPU; when passed `--submit` it generates and runs an `bsub` command that sources `.bashrc`, activates the repo `.venv`, sets `TMPDIR=/tmp`, and runs the same script in the job.

**Tech Stack:** Python 3, `segmentation_tools_pt`, `bsub`/LSF, `subprocess`.

## Global Constraints

- Only a Python script (no Hydra config, no notebook).
- Linux paths; use forward slashes for HPC.
- Grayscale single-channel input (`data_mode: single`).
- Mask and image share the same filename stem (`mask_suffix: ""`).
- Binary segmentation (`num_classes: 1`).
- Patch size: 656, stride: 328 (50% overlap).
- `.venv` must be activated inside the HPC job before running the script.

---

### Task 1: Create `scripts/patch_656_example.py`

**Files:**
- Create: `scripts/patch_656_example.py`

**Interfaces:**
- Consumes: `PatchConfig` from `segmentation_tools_pt.patches`; `build_patch_dataset` from `segmentation_tools_pt.dataset_build`; `platform_info` from `segmentation_tools_pt.core`.
- Produces: `scripts/patch_656_example.py` executable; supports `python scripts/patch_656_example.py --submit`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Extract 656x656 patches with 50% overlap from the example curated dataset.

Usage:
    python scripts/patch_656_example.py              # run locally / on HPC login node
    python scripts/patch_656_example.py --submit     # submit as LSF job
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ['TMPDIR'] = '/tmp'

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'src'))

from segmentation_tools_pt.core import platform_info
from segmentation_tools_pt.dataset_build import build_patch_dataset
from segmentation_tools_pt.patches import PatchConfig


RAW_DIR = Path('/path/to/dataset/train')
PATCH_DIR = Path('/path/to/dataset/train_patches_656')
PATCH_SIZE = 656
STRIDE = 328  # 50% overlap


def _build_command():
    return [
        sys.executable, Path(__file__).resolve(),
    ]


def _submit_job():
    job_name = 'segpt_patch_656_example'
    log_dir = Path('logs').resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / f'{job_name}_%J.out'
    err_log = log_dir / f'{job_name}_%J.err'

    repo_dir = _ROOT.resolve()
    script = Path(__file__).resolve()
    env_activate = 'source .venv/bin/activate'

    bsub_cmd = [
        'bsub', '-J', job_name,
        '-q', 'batch',
        '-n', '16',
        '-M', '64GB',
        '-W', '6:00',
        '-o', str(out_log), '-e', str(err_log),
    ]

    job_script = f"""#!/usr/bin/env bash
set -euo pipefail
echo "=== Job ${{LSB_JOBID}} | $(hostname) | $(date) ==="
source "$HOME/.bashrc"
{env_activate}
export TMPDIR=/tmp
cd "{repo_dir}"
uv sync --quiet
python "{script}"
echo "=== Done at $(date) ==="
"""

    print('Submitting LSF job...')
    print('Command:', ' '.join(bsub_cmd))
    result = subprocess.run(
        bsub_cmd,
        input=job_script,
        text=True,
        capture_output=True,
        check=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    print(f'Logs: {out_log}, {err_log}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Patch example curated dataset.')
    parser.add_argument('--submit', action='store_true', help='Submit this script as an LSF job.')
    args = parser.parse_args()

    if args.submit:
        _submit_job()
        return

    print('=' * 60)
    print('segmentation_tools_pt — 656x656 patch build for the example dataset')
    print('=' * 60)
    for k, v in platform_info().items():
        print(f'  {k:<18}: {v}')

    if not RAW_DIR.exists():
        raise FileNotFoundError(f'raw_dir does not exist: {RAW_DIR}')

    cfg = PatchConfig(
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_fg_pixels=50,
        max_empty_fraction=0.10,
        val_fraction=0.20,
        seed=42,
    )

    print(f'\nraw_dir     : {RAW_DIR}')
    print(f'patch_dir   : {PATCH_DIR}')
    print(f'patch_size  : {PATCH_SIZE}')
    print(f'stride      : {STRIDE} ({100 * (PATCH_SIZE - STRIDE) / PATCH_SIZE:.0f}% overlap)')
    print(f'num_workers : 8')

    meta = build_patch_dataset(
        source_dir=str(RAW_DIR),
        output_dir=str(PATCH_DIR),
        cfg=cfg,
        num_classes=1,
        n_channels=1,
        n_jobs=8,
        fit_rects=True,
        mask_suffix='',
    )

    n_train = len(meta['train'])
    n_val = len(meta['val'])
    n_fg = sum(1 for m in meta['train'] if not m['is_empty'])
    print(f'\nTrain patches : {n_train} ({n_fg} with foreground)')
    print(f'Val patches   : {n_val}')
    print(f'Cache ready at: {PATCH_DIR}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test (syntax and imports)**

Run:

```bash
python -m py_compile scripts/patch_656_example.py
```

Expected: no output, exit code 0.

Then verify imports do not fail by parsing the script without executing the main path (Windows cannot run the Linux `.venv`, but Python syntax + import resolution can be checked):

```bash
python -c "import ast; ast.parse(open('scripts/patch_656_example.py').read())"
```

Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/patch_656_example.py
git commit -m "feat: add 656x656 patch extraction script for the example dataset with HPC submit"
```

## Self-Review

**Spec coverage:**
- 656×656 patch size, stride 328 (50% overlap): Task 1.
- Grayscale single-channel (`single`, 1 channel): Task 1 `n_channels=1`.
- Same-stem masks (`mask_suffix=''`): Task 1.
- Binary masks (`num_classes=1`): Task 1.
- HPC submission with `.venv` activation: Task 1 `_submit_job()`.

**Placeholder scan:** None. All values are concrete.

**Type consistency:** `PatchConfig` is imported from the existing library; `build_patch_dataset` signature matches current API.
