---
name: experiment-tracking
description: 'PLACEHOLDER skill — experiment tracking (Weights & Biases, TensorBoard, MLflow) is NOT integrated into segmentation_tools_pt yet. Use when a user asks "log metrics to wandb", "TensorBoard", "experiment tracking", or "compare runs". Explains how to add a TrainCallback for it.'
argument-hint: '[optional] which tracker you want to use'
user-invocable: true
disable-model-invocation: false
---
# Experiment Tracking — NOT IMPLEMENTED YET

Centralized logging of metrics, hyperparameters, and artifacts (checkpoints, overlay JPEGs) is
**not wired** into `segpt-train` today. You can add it via a custom `TrainCallback`.

## Quick custom-callback pattern

```python
import wandb
from segmentation_tools_pt.train import TrainCallback

class WandbCallback(TrainCallback):
    def __init__(self, project, config):
        wandb.init(project=project, config=config)

    def on_epoch_end(self, state):
        wandb.log({'epoch': state.epoch, **state.train_metrics, **state.val_metrics})

    def on_train_end(self, state):
        wandb.finish()
```

Pass `callbacks=[WandbCallback(...)]` to `train()` from a notebook. The CLI does not yet support
this out of the box.

## Where a native integration would live

- New notebook: `nbs/19_experiment_tracking.ipynb` → `experiment_tracking.py`.
- Add a `tracking:` section to `configs/train.yaml` (e.g. `backend: wandb`, `project`, `tags`).
- Wire `build_default_callbacks()` to include the tracker when enabled.

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention. See `custom-callbacks-howto` for the interim
solution.
