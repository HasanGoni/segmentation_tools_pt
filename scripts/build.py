#!/usr/bin/env python3
"""Patch-cache preparation driven by Hydra build config.

Usage:
    python scripts/build.py
    python scripts/build.py data.force_rebuild=true patch.patch_size=512
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ['TMPDIR'] = '/tmp'

import hydra
from omegaconf import DictConfig

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'src'))

from segmentation_tools_pt.core import platform_info, set_seed
from segmentation_tools_pt.dataset_build import discover_samples
from segmentation_tools_pt.hydra_pipeline import (
    resolve_deterministic,
    resolve_run_seed,
    run_patch_pipeline,
    validate_data_mode_channels,
)


@hydra.main(config_path='../configs', config_name='build', version_base='1.3')
def main(cfg: DictConfig) -> None:
    print('=' * 60)
    print('segmentation_tools_pt — patch build')
    print('=' * 60)
    for k, v in platform_info().items():
        print(f'  {k:<18}: {v}')

    n_source, in_channels, channel_names = validate_data_mode_channels(cfg)
    run_seed = resolve_run_seed(cfg)
    deterministic = resolve_deterministic(cfg)

    raw_dir = Path(cfg.data.raw_dir)
    patch_dir = Path(cfg.data.patch_dir)
    mask_suffix = cfg.data.mask_suffix if cfg.data.mask_suffix is not None else ''

    print(f'\ndata_mode         : {cfg.data.data_mode}')
    print(f'raw_dir           : {raw_dir}')
    print(f'patch_dir         : {patch_dir}')
    print(f'n_source_channels : {n_source}')
    print(f'channel_names     : {channel_names}')
    print(f'in_channels       : {in_channels}')
    print(f'mask_suffix       : {mask_suffix!r}')
    print(f'parallel_workers  : {cfg.patch.num_workers}')
    print(f'run_seed          : {run_seed}')
    print(f'deterministic     : {deterministic}')

    if not bool(cfg.data.get('patch_mode', True)):
        raise ValueError('Patch build requires data.patch_mode=true.')

    if not str(cfg.data.patch_dir).strip():
        raise ValueError('data.patch_dir is required for patch build.')

    set_seed(run_seed, deterministic=deterministic)

    if not raw_dir.exists():
        raise FileNotFoundError(f'data.raw_dir does not exist: {raw_dir}')

    sample_ids = discover_samples(
        str(raw_dir),
        n_channels=n_source,
        exts=cfg.data.exts,
        mask_suffix=mask_suffix,
    )
    print(f'\nDiscovered {len(sample_ids)} samples in {raw_dir}')

    mean, std, _ = run_patch_pipeline(
        cfg, raw_dir, patch_dir, n_source, channel_names, mask_suffix,
    )

    print(f'mean: {[round(x, 4) for x in mean]}')
    print(f'std : {[round(x, 4) for x in std]}')
    print(
        f'\nPatch build complete. '
        f'patch_size={cfg.patch.patch_size}, stride={cfg.patch.stride}, '
        f'workers={cfg.patch.num_workers}.'
    )
    print(f'Cache ready at {patch_dir}')


if __name__ == '__main__':
    main()
