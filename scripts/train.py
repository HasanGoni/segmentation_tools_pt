#!/usr/bin/env python3
"""End-to-end segmentation training driven by Hydra config.

Usage:
    python scripts/train.py data.raw_dir=/path/to/raw data.patch_dir=/path/to/patches
    python scripts/train.py data.data_mode=two_channel model.arch=efficientnet-b3 train.epochs=80
"""
from __future__ import annotations

import os
import sys
from inspect import signature
from pathlib import Path

os.environ['TMPDIR'] = '/tmp'

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'src'))

from segmentation_tools_pt.core import platform_info, set_seed
from segmentation_tools_pt.dataset_build import discover_samples
from segmentation_tools_pt.export import export_onnx
from segmentation_tools_pt.hydra_pipeline import (
    resolve_batch_aug_cfg,
    resolve_boundary_blur_cfg,
    resolve_deterministic,
    resolve_hierarchy_pairs,
    resolve_mask_cfg,
    resolve_run_seed,
    resolve_task,
    run_direct_pipeline,
    run_patch_pipeline,
    validate_data_mode_channels,
    validate_patch_training_sizes,
)
from segmentation_tools_pt.model import get_model
from segmentation_tools_pt.train import build_default_callbacks, train


def _resolve_threshold(cfg: DictConfig):
    """`metrics.threshold` as a plain float or list (OmegaConf ListConfig -> list, for task='multilabel')."""
    th = cfg.metrics.threshold
    return OmegaConf.to_container(th, resolve=True) if OmegaConf.is_config(th) else th


def _model_kwargs(cfg: DictConfig, in_channels: int, *, pretrained: bool, weights_path=None) -> dict:
    """Hydra model section → get_model kwargs (works before/after nbdev_export)."""
    kw = dict(
        arch=cfg.model.arch,
        in_channels=in_channels,
        num_classes=cfg.train.num_classes,
        pretrained=pretrained,
    )
    if weights_path is not None:
        kw['weights_path'] = weights_path
    if 'architecture' in signature(get_model).parameters:
        kw['architecture'] = cfg.model.get('architecture', 'unet')
    return kw


def _export_best_model(
    cfg: DictConfig,
    checkpoint_dir: Path,
    in_channels: int,
    task: str | None,
    mask_cfg: dict,
) -> None:
    best_path = checkpoint_dir / 'best.pth'
    if not best_path.exists():
        print(f'No best checkpoint at {best_path}; skipping ONNX export.')
        return

    device = torch.device(
        cfg.train.device if torch.cuda.is_available() else 'cpu'
    )
    model = get_model(**_model_kwargs(cfg, in_channels, pretrained=False))
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)

    class_names = mask_cfg['class_names']
    hierarchy_pairs = resolve_hierarchy_pairs(cfg, class_names) or None
    thresholds = _resolve_threshold(cfg) if task == 'multilabel' else None

    onnx_path = checkpoint_dir / 'model.onnx'
    export_onnx(
        model,
        str(onnx_path),
        in_channels=in_channels,
        input_size=cfg.train.crop_size,
        opset=cfg.export.opset,
        verify=cfg.export.verify,
        task=task,
        class_names=class_names,
        thresholds=thresholds,
        hierarchy_pairs=hierarchy_pairs,
    )
    print(f'Exported ONNX → {onnx_path}')


@hydra.main(config_path='../configs', config_name='train', version_base='1.3')
def main(cfg: DictConfig) -> None:
    print('=' * 60)
    print('segmentation_tools_pt — Hydra training')
    print('=' * 60)
    for k, v in platform_info().items():
        print(f'  {k:<18}: {v}')

    n_source, in_channels, channel_names = validate_data_mode_channels(cfg)
    build_only = bool(cfg.data.get('build_only', False))
    run_seed = resolve_run_seed(cfg)
    deterministic = resolve_deterministic(cfg)
    task = resolve_task(cfg)
    mask_cfg = resolve_mask_cfg(cfg)

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
    print(f'build_only        : {build_only}')
    print(f'run_seed          : {run_seed}')
    print(f'deterministic     : {deterministic}')
    print(f'task              : {task or "(auto)"}')
    print(f'mask_mode         : {mask_cfg["mask_mode"]}')
    if mask_cfg['mask_mode'] != 'single':
        print(f'mask_channels     : {mask_cfg["mask_channels"]}')
        print(f'class_names       : {mask_cfg["class_names"]}')

    set_seed(run_seed, deterministic=deterministic)

    if not raw_dir.exists():
        raise FileNotFoundError(f'data.raw_dir does not exist: {raw_dir}')

    sample_ids = discover_samples(
        str(raw_dir),
        n_channels=n_source,
        exts=cfg.data.exts,
        mask_suffix=mask_suffix,
        mask_mode=mask_cfg['mask_mode'],
        class_names=mask_cfg['class_names'],
    )
    print(f'\nDiscovered {len(sample_ids)} samples in {raw_dir}')

    patch_mode = bool(cfg.data.get('patch_mode', True))
    print(f'patch_mode        : {patch_mode}')

    if build_only and not patch_mode:
        raise ValueError(
            'data.build_only=true requires data.patch_mode=true. '
            'Use segpt-build with configs/build.yaml instead.'
        )
    if patch_mode and not build_only:
        validate_patch_training_sizes(cfg)

    train_loader = val_loader = None
    data_root = None

    if patch_mode:
        if not str(cfg.data.patch_dir).strip():
            raise ValueError('data.patch_dir is required when data.patch_mode=true')
        mean, std, data_root = run_patch_pipeline(
            cfg, raw_dir, patch_dir, n_source, channel_names, mask_suffix,
        )
    else:
        print(f'\nDirect mode — training from full images in {raw_dir} (no patch cache)')
        mean, std, train_loader, val_loader = run_direct_pipeline(
            cfg, raw_dir, sample_ids, n_source, channel_names, mask_suffix,
        )

    print(f'mean: {[round(x, 4) for x in mean]}')
    print(f'std : {[round(x, 4) for x in std]}')

    if build_only:
        crop = cfg.train.get('crop_size', cfg.patch.patch_size)
        print(
            '\nPatch build complete. '
            f'Patch size={cfg.patch.patch_size}, stride={cfg.patch.stride}, '
            f'train crop={crop}. '
            'No model was created because data.build_only=true. '
            'Prefer segpt-build with configs/build.yaml for patch-only jobs.'
        )
        return

    weights_path = cfg.model.weights_path
    if weights_path is not None and str(weights_path).lower() in ('null', 'none', ''):
        weights_path = None

    print(f'Model weights path: {weights_path}')

    model = get_model(**_model_kwargs(cfg, in_channels, pretrained=cfg.model.pretrained, weights_path=weights_path))
    arch_label = (
        f"{cfg.model.get('architecture', 'unet')}+{cfg.model.arch}"
        if 'architecture' in signature(get_model).parameters
        else cfg.model.arch
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {arch_label}  params={n_params:,}  in_channels={in_channels}')

    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    loss_phases = OmegaConf.to_container(cfg.loss.phases, resolve=True)

    print(f'\nLoss schedule ({len(loss_phases)} phases):')
    for i, ph in enumerate(loss_phases):
        print(
            f'  phase {i + 1}: until_epoch={ph["until_epoch"]}  '
            f'tversky={ph["w_tversky"]} dice={ph["w_dice"]} boundary={ph["w_boundary"]}'
        )
    print(f'Metrics: threshold={cfg.metrics.threshold}  best_metric={cfg.metrics.best_metric}')
    print(
        f'Augment: preset={cfg.augment.get("preset", "legacy")}  '
        f'fliplr={cfg.augment.get("fliplr", 0.0)}  flipud={cfg.augment.get("flipud", 0.0)}  '
        f'flips={cfg.augment.get("flips", False)}  '
        f'boundary_blur={cfg.augment.get("boundary_blur", 0.0)}'
    )

    boundary_blur = resolve_boundary_blur_cfg(cfg)
    batch_augs = resolve_batch_aug_cfg(cfg)
    if batch_augs:
        active = {k: v for k, v in batch_augs.items() if v}
        print(f'Batch augs (pick-one, at most one per batch): {active}')
    else:
        print('Batch augs: off (augment.batch.enabled=false or all probs 0)')

    mon = cfg.get('monitoring', {})
    act_cfg = mon.get('activation_stats', {})
    pred_cfg = mon.get('pred_overlays', {})
    ea_cfg = mon.get('error_analysis', {})
    drift_cfg = mon.get('encoder_drift_check', {})
    ml_cfg = mon.get('metrics_log', {})
    hist_range = act_cfg.get('hist_range', [0, 10])
    callbacks = build_default_callbacks(
        checkpoint_dir=str(checkpoint_dir),
        threshold=_resolve_threshold(cfg),
        activation_stats=bool(act_cfg.get('enabled', True)),
        pred_overlays=bool(pred_cfg.get('enabled', True)),
        num_classes=cfg.train.num_classes,
        act_every=int(act_cfg.get('every', 1)),
        act_n_bins=int(act_cfg.get('n_bins', 40)),
        act_hist_range=(float(hist_range[0]), float(hist_range[1])),
        act_max_layers=int(act_cfg.get('max_layers', 6)),
        pred_every=int(pred_cfg.get('every', 1)),
        pred_max_val_batches=int(pred_cfg.get('max_val_batches', 3)),
        pred_jpeg_quality=int(pred_cfg.get('jpeg_quality', 90)),
        error_analysis=bool(ea_cfg.get('enabled', False)),
        ea_top_k=int(ea_cfg.get('top_k', 12)),
        ea_rank_by=str(ea_cfg.get('rank_by', 'loss')),
        ea_splits=tuple(ea_cfg.get('splits', ['train', 'val'])),
        ea_alpha=float(ea_cfg.get('alpha', 0.1)),
        ea_jpeg_quality=int(ea_cfg.get('jpeg_quality', 90)),
        ea_per_class=bool(ea_cfg.get('per_class', True)),
        drift_check=bool(drift_cfg.get('enabled', False)),
        drift_expect_frozen=(
            bool(drift_cfg.expect_frozen) if drift_cfg.get('expect_frozen') is not None
            else cfg.train.freeze_epochs == -1  # derive: freeze-forever runs must stay bit-identical
        ),
        task=task,
        class_names=mask_cfg['class_names'],
    )

    history = train(
        root=data_root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=cfg.train.num_classes,
        num_epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        encoder_lr=cfg.train.encoder_lr,
        decoder_lr=cfg.train.decoder_lr,
        freeze_epochs=cfg.train.freeze_epochs,
        crop_size=cfg.train.crop_size,
        n_source_channels=n_source,
        channel_names=channel_names,
        mean=mean,
        std=std,
        device_str=cfg.train.device,
        checkpoint_dir=str(checkpoint_dir),
        use_amp=cfg.train.use_amp,
        save_every=cfg.train.save_every,
        resume=cfg.train.resume,
        resume_from=cfg.train.get('resume_from') or None,
        loss_alpha=cfg.loss.alpha,
        loss_beta=cfg.loss.beta,
        loss_gamma=cfg.loss.gamma,
        loss_phases=loss_phases,
        metrics_threshold=_resolve_threshold(cfg),
        best_metric=cfg.metrics.best_metric,
        aug=None,
        boundary_blur=boundary_blur,
        batch_augs=batch_augs,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.get('prefetch_factor') or None,
        seed=run_seed,
        callbacks=callbacks or None,
        task=task,
        log_per_class=bool(ml_cfg.get('per_class', True)),
        console_per_class=bool(ml_cfg.get('console_per_class', False)),
        metrics_csv=bool(ml_cfg.get('csv', True)),
        **mask_cfg,
    )

    if history:
        last = history[-1]
        bm = cfg.metrics.best_metric
        print(
            f'\nTraining done. Last epoch — '
            f'val_{bm}={last.get(f"val_{bm}", 0):.4f}  '
            f'val_iou={last.get("val_iou", 0):.4f}  '
            f'val_dice={last.get("val_dice", 0):.4f}  '
            f'val_recall={last.get("val_recall", 0):.4f}'
        )

    if cfg.export.onnx:
        print('\nExporting best model to ONNX …')
        _export_best_model(cfg, checkpoint_dir, in_channels, task, mask_cfg)

    print('\nDone.')


if __name__ == '__main__':
    main()
