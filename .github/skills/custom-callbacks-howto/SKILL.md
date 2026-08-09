---
name: custom-callbacks-howto
description: 'Write custom TrainCallback hooks for segmentation_tools_pt: early stopping, LR scheduling, metric logging, or experiment-tracking integration. Use when a user asks "add early stopping", "log to wandb", "reduce LR on plateau", "custom checkpoint logic", or wants to extend the train loop without modifying train.py.'
argument-hint: '[optional] what callback you want to write'
user-invocable: true
disable-model-invocation: false
---
# Custom Callbacks How-To

Extend `train()` without editing `train.py` by subclassing `TrainCallback`. Hooks fire at
train / epoch / batch / checkpoint / validation granularity; lower `TrainCallback.order` runs first.
Set `state.stop_training = True` to exit after the current epoch.

## Minimal template

```python
from segmentation_tools_pt.train import TrainCallback, train

class MyCallback(TrainCallback):
    order = 0   # lower = earlier

    def on_train_begin(self, state):      pass
    def on_epoch_begin(self, state):      pass
    def on_train_batch_begin(self, state): pass
    def on_train_batch_end(self, state):  pass
    def on_epoch_end(self, state):        pass
    def on_val_begin(self, state):        pass
    def on_val_batch_begin(self, state):  pass
    def on_val_batch_end(self, state):    pass
    def on_val_end(self, state):          pass
    def on_checkpoint(self, state):       pass
    def on_train_end(self, state):        pass

history = train(..., callbacks=[MyCallback()])
```

## TrainState fields (read-only or mutable as documented)

- `state.model`, `state.optimizer`, `state.scheduler`, `state.scaler`
- `state.epoch`, `state.num_epochs`
- `state.batch` (current dict), `state.logits`, `state.loss`
- `state.train_metrics`, `state.val_metrics`
- `state.best_metric`, `state.checkpoint_dir`
- `state.stop_training = True` to stop gracefully

## Common examples

### Early stopping

```python
class EarlyStopping(TrainCallback):
    def __init__(self, patience=10):
        self.patience, self.wait, self.best = patience, 0, 0.0

    def on_epoch_end(self, state):
        s = state.val_metrics[state.best_metric]
        if s > self.best:
            self.best, self.wait = s, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                state.stop_training = True
```

### ReduceLROnPlateau

```python
class ReduceLROnPlateau(TrainCallback):
    def __init__(self, patience=5, factor=0.5):
        self.patience, self.factor, self.wait, self.best = patience, factor, 0, 0.0

    def on_epoch_end(self, state):
        s = state.val_metrics[state.best_metric]
        if s > self.best:
            self.best, self.wait = s, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for pg in state.optimizer.param_groups:
                    pg['lr'] *= self.factor
                self.wait = 0
```

## What this skill emits

- A callback subclass template matched to the user's goal.
- The exact `train(..., callbacks=[...])` call.

## Library anchors

- `src/segmentation_tools_pt/train.py` — `TrainCallback`, `TrainState`, `train`.
- Source notebook: `nbs/10_train.ipynb`.
