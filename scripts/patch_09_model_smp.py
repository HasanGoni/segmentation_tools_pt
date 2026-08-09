#!/usr/bin/env python3
"""Generalize get_model() for any smp architecture + encoder. Run from repo root."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "nbs" / "09_model.ipynb"
VERIFY = ROOT / ".cursor/skills/nbdev-notebook-verify/scripts/verify_notebook.py"

MARKERS = [
    "SMP_ARCHITECTURES",
    "def list_smp_architectures",
    "def list_smp_encoders",
    "architecture: str = 'unet'",
    "encoder: Optional[str] = None",
    "_load_local_encoder_weights",
    "architecture='fpn'",
    "from typing import List, Optional",
]

INTRO_MD = """# model

> Build a segmentation model (any `segmentation_models_pytorch` architecture + encoder) with configurable `in_channels` and `num_classes`, plus helpers for differential learning rates and encoder freezing.

We use `segmentation_models_pytorch`, which adapts the pretrained encoder's first convolution to any `in_channels` and lets us set `classes` to 1 (binary) or N (multi-class) directly."""

BUILD_MD = """## Build the model

`architecture` selects the smp segmentation head (`unet`, `fpn`, `deeplabv3plus`, …). `arch` / `encoder` selects any smp encoder backbone (`mobilenet_v2`, `resnet34`, `efficientnet-b3`, …).

`num_classes == 1` produces a single logit channel for sigmoid; `num_classes > 1` produces one channel per class for softmax. `activation=None` keeps the head returning raw logits, which is what our losses and metrics expect."""

GET_MODEL_PY = '''#| export
SMP_ARCHITECTURES = {
    'unet': smp.Unet,
    'unetplusplus': smp.UnetPlusPlus,
    'manet': smp.MAnet,
    'linknet': smp.Linknet,
    'fpn': smp.FPN,
    'pspnet': smp.PSPNet,
    'pan': smp.PAN,
    'deeplabv3': smp.DeepLabV3,
    'deeplabv3plus': smp.DeepLabV3Plus,
    'upernet': smp.UPerNet,
}


def list_smp_architectures() -> List[str]:
    """Segmentation head names accepted by `get_model(architecture=...)`."""
    return sorted(SMP_ARCHITECTURES.keys())


def list_smp_encoders() -> List[str]:
    """Encoder / backbone names from segmentation-models-pytorch."""
    return sorted(smp.encoders.get_encoder_names())


def _normalize_architecture(name: str) -> str:
    key = name.lower().replace('-', '').replace('_', '')
    aliases = {'unetplusplus': 'unetplusplus', 'deeplabv3plus': 'deeplabv3plus'}
    return aliases.get(key, key)


def _load_local_encoder_weights(model: nn.Module, weights_path: str, device: str) -> None:
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model_state = model.encoder.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [
        k for k in state
        if k in model_state and state[k].shape != model_state[k].shape
    ]
    if skipped:
        print(
            f'WARNING: {len(skipped)} key(s) skipped from {weights_path} due to shape mismatch '
            f'(checkpoint in_channels ≠ model in_channels): {skipped}'
        )
    res = model.encoder.load_state_dict(compatible, strict=False)
    print(f'Loaded encoder from {weights_path} ({len(compatible)}/{len(model_state)} keys matched)')
    if res.missing_keys:
        print(f'  missing keys: {len(res.missing_keys)} (e.g. {res.missing_keys[0]})')
    if res.unexpected_keys:
        print(f'  unexpected keys: {len(res.unexpected_keys)} (e.g. {res.unexpected_keys[0]})')


def get_model(
    arch: str = 'mobilenet_v2',   # encoder name — alias kept for Hydra `model.arch`
    in_channels: int = 3,
    num_classes: int = 1,         # 1 = binary, >1 = multi-class
    pretrained: bool = True,      # ImageNet weights (ignored when weights_path is set)
    weights_path: Optional[str] = None,  # local encoder .pth from download_encoder_weights
    device: str = 'cpu',
    architecture: str = 'unet',   # smp head: unet | fpn | deeplabv3plus | ...
    encoder: Optional[str] = None,  # explicit encoder name; overrides `arch` when set
) -> nn.Module:
    """Build an smp model returning raw logits `(B, num_classes, H, W)`.

    Pass `weights_path` on air-gapped HPC instead of downloading ImageNet weights.
    Use `list_smp_architectures()` and `list_smp_encoders()` to discover valid names.

    Example:
        >>> m = get_model('resnet34', architecture='fpn', in_channels=3, num_classes=1, pretrained=False)
        >>> tuple(m(torch.randn(1, 3, 64, 64)).shape)
        (1, 1, 64, 64)
    """
    encoder_name = encoder if encoder is not None else arch
    arch_key = _normalize_architecture(architecture)
    if arch_key not in SMP_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Choose one of: {', '.join(list_smp_architectures())}"
        )
    valid_encoders = smp.encoders.get_encoder_names()
    if encoder_name not in valid_encoders:
        raise ValueError(
            f"Unknown encoder {encoder_name!r}. "
            f"Use list_smp_encoders() — {len(valid_encoders)} encoders available."
        )
    use_imagenet = pretrained and weights_path is None
    model_cls = SMP_ARCHITECTURES[arch_key]
    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights='imagenet' if use_imagenet else None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    if weights_path is not None:
        _load_local_encoder_weights(model, weights_path, device)
    return model'''

TEST_PY = """# test: output channels follow num_classes for binary and multi-class
m1 = get_model('mobilenet_v2', in_channels=2, num_classes=1, pretrained=False)
assert tuple(m1(torch.randn(1, 2, 64, 64)).shape) == (1, 1, 64, 64)
m3 = get_model('mobilenet_v2', in_channels=4, num_classes=3, pretrained=False)
assert tuple(m3(torch.randn(1, 4, 64, 64)).shape) == (1, 3, 64, 64)
m_fpn = get_model('resnet18', architecture='fpn', in_channels=3, num_classes=1, pretrained=False)
assert tuple(m_fpn(torch.randn(1, 3, 64, 64)).shape) == (1, 1, 64, 64)
assert 'resnet34' in list_smp_encoders()
assert 'deeplabv3plus' in list_smp_architectures()"""


def _as_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.split("\n")]


def _cell_text(source: list[str]) -> str:
    return "".join(source)


def patch_notebook() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    patched = []

    for cell in nb["cells"]:
        src = cell.get("source", [])
        text = _cell_text(src)

        if cell["cell_type"] == "markdown" and text.startswith("# model\n"):
            cell["source"] = _as_lines(INTRO_MD)
            patched.append("intro markdown")
            continue

        if cell["cell_type"] == "markdown" and text.startswith("## Build the model"):
            cell["source"] = _as_lines(BUILD_MD)
            patched.append("build markdown")
            continue

        if cell["cell_type"] == "code" and text.startswith("#| export\nfrom typing import List"):
            cell["source"] = _as_lines(
                "#| export\nfrom typing import List, Optional\n\nimport torch\nimport torch.nn as nn\n"
            )
            patched.append("imports")
            continue

        if cell["cell_type"] == "code" and "def get_model(" in text and text.startswith("#| export"):
            cell["source"] = _as_lines(GET_MODEL_PY)
            patched.append("get_model")
            continue

        if cell["cell_type"] == "code" and text.startswith("# test: output channels follow num_classes"):
            cell["source"] = _as_lines(TEST_PY)
            patched.append("tests")
            continue

    if "get_model" not in patched:
        raise SystemExit("get_model cell not found — notebook layout changed")

    tmp = NOTEBOOK.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(NOTEBOOK)
    print("Patched:", ", ".join(patched))


def verify() -> None:
    cmd = [sys.executable, str(VERIFY), str(NOTEBOOK)]
    for marker in MARKERS:
        cmd.extend(["--expect", marker])
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    patch_notebook()
    verify()
    print("OK — all markers present in", NOTEBOOK)


if __name__ == "__main__":
    main()
