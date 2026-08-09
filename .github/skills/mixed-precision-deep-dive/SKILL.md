---
name: mixed-precision-deep-dive
description: 'Tune automatic mixed precision (AMP) training in segmentation_tools_pt: when it helps, when it hurts, and how to debug NaN/Inf. Use when a user asks "use_amp", "mixed precision", "fp16 training", "loss scaling", or "NaN loss with AMP". Emits train.use_amp guidance and debugging checklist.'
argument-hint: '[optional] GPU and model size'
user-invocable: true
disable-model-invocation: false
---
# Mixed-Precision Deep Dive

`train.use_amp: true` enables PyTorch automatic mixed precision (AMP). On modern NVIDIA GPUs this
usually gives faster training and lower memory with no accuracy loss, but it can introduce numerical
instability.

## When to enable

- NVIDIA GPUs with Tensor Cores (V100, T4, RTX 20+/A100/L40).
- Large models / large crop sizes where memory is the bottleneck.
- When you want ~1.3–1.8× speedup for free.

## When to disable

- Older GPUs without Tensor Cores (AMP may slow things down).
- You observe NaN/Inf losses that disappear with AMP off.
- Very small models where the overhead is not worth it.

## Config

```yaml
train:
  use_amp: true
```

## Debugging NaN/Inf

1. Check `loss.gamma` — very high focal gamma + AMP can underflow.
2. Check `boundary_blur` band widths — large values can produce extreme gradients.
3. Try `use_amp: false` for one epoch to confirm AMP is the cause.
4. If confirmed, keep AMP off or switch to full `float32` for the loss head only (requires custom
   callback — see `custom-callbacks-howto`).

## What this skill emits

- A go/no-go decision for `train.use_amp` and a debugging checklist.

## Library anchors

- `src/segmentation_tools_pt/train.py` — AMP context manager in `train_one_epoch`/`validate`.
- Source notebook: `nbs/10_train.ipynb`.
