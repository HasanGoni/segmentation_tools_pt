---
name: production-deployment
description: 'PLACEHOLDER skill — production deployment patterns (ONNX Runtime inference server, TorchScript, batch inference pipeline) are NOT fully implemented in segmentation_tools_pt yet. Use when a user asks "deploy the model", "inference server", "batch inference production", or "TorchScript". Explains options and where they would be added.'
argument-hint: '[optional] target deployment environment'
user-invocable: true
disable-model-invocation: false
---
# Production Deployment — NOT FULLY IMPLEMENTED YET

`segmentation_tools_pt` is primarily a training library. Production serving patterns are **not**
included today, though ONNX export exists (see `export-onnx` skill).

## Deployment options

| Target | Path |
|---|---|
| ONNX Runtime (CPU/GPU) | Export with `export_onnx()`, then load in ONNX Runtime |
| TorchScript | Trace/script the model for C++/mobile deployment |
| Python microservice | FastAPI/Flask wrapper around `predict()` |
| Batch inference pipeline | Apply `predict()` over a folder of images |

## Where native support would be added

- New notebook: `nbs/23_deployment.ipynb` → `deployment.py`.
- Functions: `load_onnx_session(path)`, `predict_onnx(session, image)`, `batch_predict(model, paths)`.
- Add `segpt-predict` CLI entry point in `pyproject.toml` + `cli.py`.

## What the user can do today

- Export to ONNX with `export_onnx()` and write a thin ONNX Runtime wrapper.
- Wrap `inference.predict()` in a FastAPI endpoint.

## Status

**Not fully implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
