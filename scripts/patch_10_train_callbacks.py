#!/usr/bin/env python3
"""Apply training-callback wiring to nbs/10_train.ipynb. Run from repo root."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "nbs" / "10_train.ipynb"
VERIFY = ROOT / ".cursor/skills/nbdev-notebook-verify/scripts/verify_notebook.py"

MARKERS = [
    "class TrainState",
    "class TrainCallback",
    "state: Optional[TrainState]",
    "callbacks: Optional[Sequence[TrainCallback]] = None",
    "on_train_batch_begin",
    "on_train_begin",
    "on_checkpoint",
    "if state.stop_training:",
    "_CountBatches",
    "## Training callbacks",
]

CALLBACKS_MD = """## Training callbacks

Optional hooks at train / epoch / batch granularity. Subclass `TrainCallback` and pass a list to `train()` — the default loop is unchanged when `callbacks=None`.

Set `state.stop_training = True` in a callback (e.g. early stopping) to exit the epoch loop cleanly after the current epoch.

## One training epoch

One job: run the loader once, update the model, and return averaged losses and training metrics. Mixed precision is used when a `GradScaler` is supplied."""

CALLBACKS_PY = '''#| export
def _sorted_callbacks(callbacks: Optional[Sequence[TrainCallback]]) -> List[TrainCallback]:
    if not callbacks:
        return []
    return sorted(callbacks, key=lambda cb: getattr(cb, 'order', 0))


def _call_callbacks(
    callbacks: Optional[Sequence[TrainCallback]],
    hook: str,
    state: Optional[TrainState],
    **kwargs,
) -> None:
    if not callbacks or state is None:
        return
    for cb in _sorted_callbacks(callbacks):
        getattr(cb, hook)(state, **kwargs)
'''

TRAIN_ONE_EPOCH = '''#| export
def train_one_epoch(
    model: nn.Module,                  # model returning (B, C, H, W) logits
    loader,                            # training DataLoader of {'image','mask'} batches
    criterion: SegLoss,                # loss returning a breakdown dict
    optimizer: torch.optim.Optimizer,  # optimizer
    device: torch.device,              # cpu or cuda
    scaler: Optional[object] = None,   # torch.amp.GradScaler for AMP, or None
    grad_clip: float = 1.0,            # max gradient norm
    num_classes: int = 1,              # 1 = binary, >1 = multi-class
    callbacks: Optional[Sequence[TrainCallback]] = None,
    state: Optional[TrainState] = None,
) -> Dict[str, float]:
    """Train for one epoch; return averaged losses and `train_*` metrics."""
    model.train()
    running = {'loss': 0., 'loss_tversky': 0., 'loss_dice': 0., 'loss_boundary': 0.}
    metrics = SegMetrics(num_classes=num_classes)
    for batch_idx, batch in enumerate(loader):
        if state is not None:
            state.batch_idx = batch_idx
            state.batch = batch
        _call_callbacks(callbacks, 'on_train_batch_begin', state)
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                ld = criterion(logits, targets)
            scaler.scale(ld['loss']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(images)
            ld = criterion(logits, targets)
            ld['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if state is not None:
            state.logits = logits.detach()
            state.loss_dict = {k: float(v) for k, v in ld.items()}
        _call_callbacks(callbacks, 'on_train_batch_end', state)
        for k in running:
            running[k] += float(ld[k])
        metrics.update(logits.detach(), targets)
    n = len(loader)
    out = {k: v / n for k, v in running.items()}
    out.update({f'train_{k}': v for k, v in metrics.compute().items() if k != 'per_class'})
    return out
'''

VALIDATE = '''#| export
@torch.no_grad()
def validate(
    model: nn.Module,    # model returning logits
    loader,              # validation DataLoader
    criterion: SegLoss,  # loss for tracking val loss
    device: torch.device,
    num_classes: int = 1,
    callbacks: Optional[Sequence[TrainCallback]] = None,
    state: Optional[TrainState] = None,
) -> Dict[str, float]:
    """Run validation; return overlap metrics plus `val_loss`."""
    model.eval()
    _call_callbacks(callbacks, 'on_val_begin', state)
    metrics = SegMetrics(num_classes=num_classes)
    running = 0.0
    for batch_idx, batch in enumerate(loader):
        if state is not None:
            state.batch_idx = batch_idx
            state.batch = batch
        _call_callbacks(callbacks, 'on_val_batch_begin', state)
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)
        logits = model(images)
        ld = criterion(logits, targets)
        running += float(ld['loss'])
        if state is not None:
            state.logits = logits
            state.loss_dict = {k: float(v) for k, v in ld.items()}
        _call_callbacks(callbacks, 'on_val_batch_end', state)
        metrics.update(logits, targets)
    res = {k: v for k, v in metrics.compute().items() if k != 'per_class'}
    res['val_loss'] = running / len(loader)
    if state is not None:
        state.val_metrics = res
    _call_callbacks(callbacks, 'on_val_end', state)
    return res
'''

SMOKE = '''# smoke test: one epoch + validate + checkpoint round-trip on a tiny in-memory task
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
'''


def _lines(s: str) -> list[str]:
    return s.splitlines(keepends=True)


def _patch_train_cell(src: str) -> str:
    if "def train(" not in src or "Resumable training loop" not in src:
        raise ValueError("train() cell not found")
    old = src
    old = old.replace(
        "    seed: Optional[int] = None,     # run seed for reproducible sampling and worker RNGs\n) -> list:",
        "    seed: Optional[int] = None,     # run seed for reproducible sampling and worker RNGs\n    callbacks: Optional[Sequence[TrainCallback]] = None,\n) -> list:",
    )
    loop_old = (
        "    if freeze_epochs > 0:\n"
        "        freeze_encoder(model)\n\n"
        "    for epoch in range(start_epoch, num_epochs):\n"
        "        _w = resolve_loss_weights(epoch, _phases)\n"
        "        criterion.set_weights(*_w)\n"
        "        if freeze_epochs > 0 and epoch == freeze_epochs:\n"
        "            unfreeze_encoder(model)\n"
        "        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, num_classes=num_classes)\n"
        "        va = validate(model, val_loader, criterion, device, num_classes=num_classes)"
    )
    loop_new = (
        "    if freeze_epochs > 0:\n"
        "        freeze_encoder(model)\n\n"
        "    state = TrainState(\n"
        "        model=model, device=device, num_epochs=num_epochs, num_classes=num_classes,\n"
        "        optimizer=optimizer, scheduler=scheduler, scaler=scaler, criterion=criterion,\n"
        "        train_loader=train_loader, val_loader=val_loader,\n"
        "        history=history, best_score=best_score, best_metric=best_metric,\n"
        "        checkpoint_dir=checkpoint_dir,\n"
        "    )\n"
        "    _call_callbacks(callbacks, 'on_train_begin', state)\n\n"
        "    for epoch in range(start_epoch, num_epochs):\n"
        "        state.epoch = epoch\n"
        "        state.history = history\n"
        "        state.best_score = best_score\n"
        "        _call_callbacks(callbacks, 'on_epoch_begin', state)\n"
        "        _w = resolve_loss_weights(epoch, _phases)\n"
        "        criterion.set_weights(*_w)\n"
        "        if freeze_epochs > 0 and epoch == freeze_epochs:\n"
        "            unfreeze_encoder(model)\n"
        "        tr = train_one_epoch(\n"
        "            model, train_loader, criterion, optimizer, device, scaler,\n"
        "            num_classes=num_classes, callbacks=callbacks, state=state,\n"
        "        )\n"
        "        state.train_metrics = tr\n"
        "        va = validate(\n"
        "            model, val_loader, criterion, device, num_classes=num_classes,\n"
        "            callbacks=callbacks, state=state,\n"
        "        )"
    )
    if loop_old not in old:
        if "on_train_begin" not in old:
            raise ValueError("train loop anchor not found")
    else:
        old = old.replace(loop_old, loop_new)
    if "state.is_best = is_best" not in old:
        old = old.replace(
            "        is_best = va[best_metric] > best_score\n        print(",
            "        is_best = va[best_metric] > best_score\n        state.is_best = is_best\n        print(",
        )
    tail_old = (
        "        if is_best:\n"
        "            best_score = va[best_metric]\n"
        "            save_checkpoint(str(Path(checkpoint_dir) / 'best.pth'), *args, best_score, history)\n"
        "        if (epoch + 1) % save_every == 0:\n"
        "            save_checkpoint(str(Path(checkpoint_dir) / f'periodic_ep{epoch:03d}.pth'), *args, best_score, history)\n"
        "        save_checkpoint(str(latest), *args, best_score, history)\n"
        "    return history"
    )
    tail_new = (
        "        if is_best:\n"
        "            best_score = va[best_metric]\n"
        "            state.best_score = best_score\n"
        "            best_path = str(Path(checkpoint_dir) / 'best.pth')\n"
        "            save_checkpoint(best_path, *args, best_score, history)\n"
        "            _call_callbacks(callbacks, 'on_checkpoint', state, path=best_path)\n"
        "        if (epoch + 1) % save_every == 0:\n"
        "            periodic_path = str(Path(checkpoint_dir) / f'periodic_ep{epoch:03d}.pth')\n"
        "            save_checkpoint(periodic_path, *args, best_score, history)\n"
        "            _call_callbacks(callbacks, 'on_checkpoint', state, path=periodic_path)\n"
        "        latest_path = str(latest)\n"
        "        save_checkpoint(latest_path, *args, best_score, history)\n"
        "        _call_callbacks(callbacks, 'on_checkpoint', state, path=latest_path)\n"
        "        _call_callbacks(callbacks, 'on_epoch_end', state)\n"
        "        if state.stop_training:\n"
        "            break\n"
        "    _call_callbacks(callbacks, 'on_train_end', state)\n"
        "    return history"
    )
    if tail_old in old:
        old = old.replace(tail_old, tail_new)
    elif "if state.stop_training:" not in old:
        raise ValueError("train tail anchor not found and stop_training not present")
    return old


def patch() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and src.startswith("## One training epoch"):
            cell["source"] = _lines(CALLBACKS_MD)
        elif "def _call_callbacks" in src and "def _sorted_callbacks" in src:
            cell["source"] = _lines(CALLBACKS_PY)
        elif "def train_one_epoch(" in src and "#| export" in src:
            cell["source"] = _lines(TRAIN_ONE_EPOCH)
        elif "def validate(" in src and "@torch.no_grad()" in src:
            cell["source"] = _lines(VALIDATE)
        elif "def train(" in src and "Resumable training loop" in src:
            cell["source"] = _lines(_patch_train_cell(src))
        elif "class _Toy(Dataset):" in src and "smoke test" in src:
            cell["source"] = _lines(SMOKE)

    tmp = NOTEBOOK.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(NOTEBOOK)


def verify() -> int:
    cmd = [sys.executable, str(VERIFY), str(NOTEBOOK)]
    for m in MARKERS:
        cmd.extend(["--expect", m])
    return subprocess.call(cmd)


def main() -> int:
    patch()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
