#!/usr/bin/env python3
"""Add monitoring callbacks to nbs/10_train.ipynb. Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "nbs" / "10_train.ipynb"

IMPORTS = '''#| export
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
'''

EXPORTI = '''#| exporti
from segmentation_tools_pt.channels import DEFAULT_CHANNELS
from segmentation_tools_pt.losses import SegLoss, resolve_loss_weights
from segmentation_tools_pt.metrics import SegMetrics
from segmentation_tools_pt.model import get_param_groups, freeze_encoder, unfreeze_encoder
from segmentation_tools_pt.visualize import save_color_dim_png, save_near_zero_png, save_overlay_grid_jpeg
'''

MONITORING_MD = """## Default monitoring callbacks

fastai-style activation histograms (`ActivationStatsCallback`) and YOLO-style validation overlay JPEGs (`PredOverlayCallback`). Use `build_default_callbacks()` from Hydra or pass directly to `train()`."""

MONITORING_PY = '''#| export
_HOOK_TYPES = (nn.Conv2d, nn.BatchNorm2d)

def _hookable_layers(model: nn.Module, max_layers: int) -> List[Tuple[str, nn.Module]]:
    "Up to `max_layers` hook targets, evenly spaced across Conv/BN modules."
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, _HOOK_TYPES) and n]
    if not layers:
        layers = [(n, m) for n, m in model.named_modules() if n]
    if len(layers) <= max_layers:
        return layers
    idxs = np.linspace(0, len(layers) - 1, max_layers, dtype=int)
    return [layers[i] for i in idxs]

class ActivationStatsCallback(TrainCallback):
    """Record per-layer activation histograms; save fastai colorful-dimension PNGs each epoch."""
    order = 10

    def __init__(
        self,
        checkpoint_dir: str,
        every: int = 1,
        n_bins: int = 40,
        hist_range: Tuple[float, float] = (0.0, 10.0),
        max_layers: int = 6,
    ):
        self.root = Path(checkpoint_dir) / 'activation_stats'
        self.every = every
        self.n_bins = n_bins
        self.hmin, self.hmax = hist_range
        self.max_layers = max_layers
        self._hooks: List[Any] = []
        self._layer_names: List[str] = []
        self._epoch_hists: List[List[torch.Tensor]] = []
        self._epoch_near_zero: List[List[float]] = []
        self._near_zero_history: Dict[str, List[Tuple[int, float]]] = {}
        self._pending: List[Optional[Tuple[torch.Tensor, float]]] = []

    def on_train_begin(self, state: TrainState) -> None:
        layers = _hookable_layers(state.model, self.max_layers)
        self._layer_names = [n.replace('.', '_') for n, _ in layers]
        self._epoch_hists = [[] for _ in layers]
        self._epoch_near_zero = [[] for _ in layers]
        self._near_zero_history = {name: [] for name in self._layer_names}
        self._pending = [None] * len(layers)

        def _make_hook(li: int):
            def hook(module, _inp, out):
                if not module.training:
                    return
                o = out.detach().float()
                if o.numel() == 0:
                    return
                flat = o.flatten()
                hist = torch.histc(flat, self.n_bins, self.hmin, self.hmax).cpu()
                nz = float((flat <= 0.05).float().mean().item())
                self._pending[li] = (hist, nz)
            return hook

        for i, (_, mod) in enumerate(layers):
            self._hooks.append(mod.register_forward_hook(_make_hook(i)))

    def on_train_batch_end(self, state: TrainState) -> None:
        if state.batch_idx % self.every != 0:
            return
        for i, item in enumerate(self._pending):
            if item is None:
                continue
            hist, nz = item
            self._epoch_hists[i].append(hist)
            self._epoch_near_zero[i].append(nz)
            self._pending[i] = None

    def on_epoch_end(self, state: TrainState) -> None:
        epoch_dir = self.root / f'epoch{state.epoch + 1:03d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(self._layer_names):
            hists = self._epoch_hists[i]
            if not hists:
                continue
            stack = torch.stack(hists)
            save_color_dim_png(stack.T, epoch_dir / f'layer{i:02d}_{name}_color_dim.png', title=name)
            mean_nz = float(np.mean(self._epoch_near_zero[i]))
            self._near_zero_history[name].append((state.epoch, mean_nz))
            ep, vals = zip(*self._near_zero_history[name])
            save_near_zero_png(list(ep), list(vals), epoch_dir / f'layer{i:02d}_{name}_near_zero.png', title=name)
        self._epoch_hists = [[] for _ in self._layer_names]
        self._epoch_near_zero = [[] for _ in self._layer_names]

    def on_train_end(self, state: TrainState) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

class PredOverlayCallback(TrainCallback):
    """Save YOLO-style validation overlay JPEG mosaics during training."""
    order = 20

    def __init__(
        self,
        checkpoint_dir: str,
        threshold: float = 0.5,
        every: int = 1,
        max_val_batches: int = 3,
        jpeg_quality: int = 90,
        num_classes: int = 1,
    ):
        self.root = Path(checkpoint_dir) / 'predictions'
        self.threshold = threshold
        self.every = every
        self.max_val_batches = max_val_batches
        self.jpeg_quality = jpeg_quality
        self.num_classes = num_classes

    def _decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return (torch.sigmoid(logits[:, 0]) >= self.threshold).long()
        return logits.argmax(dim=1)

    def on_val_batch_end(self, state: TrainState) -> None:
        if state.epoch % self.every != 0:
            return
        if state.batch_idx >= self.max_val_batches:
            return
        if state.batch is None or state.logits is None:
            return
        preds = self._decode_logits(state.logits)
        out_dir = self.root / f'epoch{state.epoch + 1:03d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        images, masks = state.batch['image'], state.batch['mask']
        bi = state.batch_idx
        for mode, suffix in (('labels', 'labels'), ('pred', 'pred'), ('compare', 'compare')):
            save_overlay_grid_jpeg(
                images, masks, preds,
                out_dir / f'val_batch{bi}_{suffix}.jpg',
                threshold=self.threshold,
                mode=mode,
                jpeg_quality=self.jpeg_quality,
            )

def build_default_callbacks(
    checkpoint_dir: str,
    threshold: float = 0.5,
    activation_stats: bool = True,
    pred_overlays: bool = True,
    num_classes: int = 1,
    act_every: int = 1,
    act_n_bins: int = 40,
    act_hist_range: Tuple[float, float] = (0.0, 10.0),
    act_max_layers: int = 6,
    pred_every: int = 1,
    pred_max_val_batches: int = 3,
    pred_jpeg_quality: int = 90,
) -> List[TrainCallback]:
    "Factory for on-by-default training monitoring callbacks."
    cbs: List[TrainCallback] = []
    if activation_stats:
        cbs.append(ActivationStatsCallback(
            checkpoint_dir, every=act_every, n_bins=act_n_bins,
            hist_range=act_hist_range, max_layers=act_max_layers,
        ))
    if pred_overlays:
        cbs.append(PredOverlayCallback(
            checkpoint_dir, threshold=threshold, every=pred_every,
            max_val_batches=pred_max_val_batches, jpeg_quality=pred_jpeg_quality,
            num_classes=num_classes,
        ))
    return cbs
'''

SMOKE = '''# smoke test: one epoch + validate + checkpoint round-trip + monitoring callbacks
import tempfile
from torch.utils.data import Dataset, DataLoader

class _Toy(Dataset):
    def __len__(self): return 8
    def __getitem__(self, i):
        return {'image': torch.rand(3, 16, 16), 'mask': (torch.rand(16, 16) > 0.5).long(), 'sample_id': str(i)}

class _CountBatches(TrainCallback):
    def __init__(self): self.train_batches = self.val_batches = 0
    def on_train_batch_end(self, state): self.train_batches += 1
    def on_val_batch_end(self, state): self.val_batches += 1

loader = DataLoader(_Toy(), batch_size=2)
model = nn.Conv2d(3, 1, 3, padding=1)
crit = SegLoss(w_tversky=0.5, w_dice=0.5, w_boundary=0.0)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
dev = torch.device('cpu')
st = TrainState(model=model, device=dev, num_epochs=1)
cb = _CountBatches()
tr = train_one_epoch(model, loader, crit, opt, dev, num_classes=1, callbacks=[cb], state=st)
assert 'loss' in tr and 'train_iou' in tr
assert cb.train_batches == len(loader)
va = validate(model, loader, crit, dev, num_classes=1, callbacks=[cb], state=st)
assert 'val_loss' in va and 'iou' in va
assert cb.val_batches == len(loader)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / 'ckpt.pth'
    save_checkpoint(p, 0, model, opt, sched, None, 0.0, [])
    se, best, hist = load_checkpoint(p, model, opt, sched, None, dev)
    assert se == 1 and best == 0.0
    ckpt = Path(d) / 'checkpoints'
    mon = build_default_callbacks(str(ckpt), threshold=0.5, num_classes=1)
    hist2 = train(
        root=None, model=model, train_loader=loader, val_loader=loader,
        num_epochs=1, batch_size=2, device_str='cpu', checkpoint_dir=str(ckpt),
        use_amp=False, resume=False, callbacks=mon,
    )
    assert len(hist2) == 1
    assert list((ckpt / 'activation_stats').rglob('*_color_dim.png'))
    assert list((ckpt / 'predictions').rglob('val_batch0_labels.jpg'))
'''


def _lines(s: str) -> list[str]:
    return s.splitlines(keepends=True)


def patch() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    inserted = False
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'code' and src.startswith('#| export\nimport json'):
            cell['source'] = _lines(IMPORTS)
        elif cell['cell_type'] == 'code' and '#| exporti' in src and 'DEFAULT_CHANNELS' in src:
            cell['source'] = _lines(EXPORTI)
        elif 'class _Toy(Dataset):' in src and 'smoke test' in src:
            cell['source'] = _lines(SMOKE)

    if not any('ActivationStatsCallback' in ''.join(c.get('source', [])) for c in nb['cells']):
        idx = next(
            i for i, c in enumerate(nb['cells'])
            if c['cell_type'] == 'code' and 'class TrainCallback:' in ''.join(c.get('source', []))
        )
        nb['cells'][idx + 1:idx + 1] = [
            {'cell_type': 'markdown', 'metadata': {}, 'source': _lines(MONITORING_MD)},
            {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': _lines(MONITORING_PY)},
        ]
        inserted = True

    tmp = NOTEBOOK.with_suffix('.ipynb.tmp')
    tmp.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(NOTEBOOK)
    print(f'patched {NOTEBOOK} (inserted monitoring cells: {inserted})')


if __name__ == '__main__':
    patch()
