---
name: export-onnx
description: 'Export a trained segmentation_tools_pt model to ONNX and verify it matches PyTorch. Use when a user asks "export to ONNX", "deploy the model", "convert checkpoint to onnx", or is editing the export.* block. Emits the export.* block for train.yaml or a direct export_onnx() call.'
argument-hint: '[optional] checkpoint path, in_channels, num_classes'
user-invocable: true
disable-model-invocation: false
---
# Export to ONNX

Trace a trained model to ONNX at a chosen opset and verify the output matches PyTorch within
tolerance.

## Two ways to run

**Automatic after training** — set in `configs/train.yaml`:

```yaml
export:
  onnx: true      # requires onnx + onnxruntime installed in the venv
  opset: 18
  verify: true
```

**Manual** — from Python (`model` is a loaded `nn.Module`, not a checkpoint path — load the
state dict yourself first, e.g. via `get_model(...)` + `model.load_state_dict(...)`):

```python
from segmentation_tools_pt.export import export_onnx
export_onnx(model, 'model.onnx', in_channels=1, input_size=256)
```

## Notes

- `export.onnx: true` needs `onnx` and `onnxruntime` installed in the environment.
- `verify: true` compares ONNX vs PyTorch outputs and raises on mismatch.
- `in_channels` / `input_size` must match the trained model.

## Multilabel metadata (task='multilabel', e.g. chip/void/lifted)

For a multilabel model (see `multi-class-segmentation` skill's overlap callout, or
`docs/MULTILABEL_OVERLAP.md`), pass `task`, `class_names`, `thresholds`, and `hierarchy_pairs` so
the raw `logits` output is self-describing for any downstream consumer:

```python
export_onnx(
    model, 'model.onnx', in_channels=1, input_size=256,
    task='multilabel',
    class_names=['chip', 'void', 'lifted'],
    thresholds=[0.5, 0.5, 0.5],
    hierarchy_pairs=[(2, 0)],   # lifted (idx 2) subset of chip (idx 0)
)
```

These are written as JSON strings into the ONNX file's `metadata_props` (read back via
`{p.key: p.value for p in onnx.load(path).metadata_props}`), telling a consumer to apply
independent per-channel sigmoid (not softmax+argmax) and, if desired,
`apply_hierarchy(masks, hierarchy_pairs)` to hard-enforce the subset constraint post-decode.
`scripts/train.py`'s automatic export path (`export.onnx: true`) already resolves and passes
these via `resolve_hierarchy_pairs(cfg, class_names)` and the training `task`/`class_names`.

## Library anchors

- `src/segmentation_tools_pt/export.py` — `export_onnx`.
- `src/segmentation_tools_pt/hydra_pipeline.py` — `resolve_hierarchy_pairs`.
- `src/segmentation_tools_pt/inference.py` — `apply_hierarchy`.
- Source notebooks: `nbs/13_export.ipynb`, `nbs/15_hydra_pipeline.ipynb`.
