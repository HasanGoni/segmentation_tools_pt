#!/usr/bin/env python3
"""Extract 656x656 patches with 50% overlap from the example curated dataset.

Run this from the HPC login node inside the repo after activating the environment:

    source .venv/bin/activate
    python scripts/patch_656_example.py

To submit as an LSF job instead:

    python scripts/patch_656_example.py --submit
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


def _submit_job() -> None:
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
    try:
        result = subprocess.run(
            bsub_cmd,
            input=job_script,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            'bsub not found. Are you logged into the HPC login node?'
        ) from exc
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
