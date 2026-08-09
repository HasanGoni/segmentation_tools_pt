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

**Manual** — from Python:

```python
from segmentation_tools_pt.export import export_onnx
export_onnx('checkpoints/best.pth', 'model.onnx', in_channels=1, num_classes=1)
```

## Notes

- `export.onnx: true` needs `onnx` and `onnxruntime` installed in the environment.
- `verify: true` compares ONNX vs PyTorch outputs and raises on mismatch.
- `in_channels` / `num_classes` must match the trained model.

## Library anchors

- `src/segmentation_tools_pt/export.py` — `export_onnx`.
- Source notebook: `nbs/13_export.ipynb`.
