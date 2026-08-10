# Multi-Label Overlap Segmentation — Planning Doc

**Status:** Phases 0-5 done. Losses, dataset loading, metrics/inference, the patch-build + train +
Hydra wiring, hierarchy enforcement (soft regularizer + hard `apply_hierarchy`), multilabel
prediction overlays, and ONNX export metadata (`task`/`class_names`/`thresholds`/
`hierarchy_pairs` in `metadata_props`) are all task-aware end to end, verified via a real
Hydra-config-driven run (`configs/train_multilabel_3head.yaml`) through to a real
`onnx.load()`-verified export. The consumer-side integration guide for
`void_detection_example_line` is below. Remaining work is entirely on the
`void_detection_example_line` side (pseudo-labeling, real training data, deployment).

**Target:** One lightweight TRT engine with 3 independent sigmoid heads — `chip`, `void`, `lifted` — where void may overlap chip and lifted is mostly coextensive with chip.

**Related product:** `void_detection_example_line` (chip / void / lifted chip on grayscale X-ray).

---

## Development workflow (mandatory)

This project uses **nbdev** and **Donald Knuth literate programming**. These rules are non-negotiable for all multilabel work:

1. **Notebooks first** — explore, explain, and verify in `nbs/` before any export.
2. **One line or small block → print/plot immediately** — never write large untested chunks.
3. **Prose before code** — each notebook section states *why* before *how* (literate style).
4. **Export only stable API** — `#| export` cells in notebooks; never hand-edit `src/segmentation_tools_pt/*.py`.
5. **Suggested notebook order for multilabel:**
   - `nbs/00_core.ipynb` — `TaskType`, `SegConfig.is_multilabel`
   - `nbs/07_losses.ipynb` — `probs_and_targets(..., task='multilabel')`
   - `nbs/06_dataset.ipynb` — K-channel mask loading + albumentations
   - `nbs/08_metrics.ipynb` — per-head IoU/Dice
   - `nbs/11_inference.ipynb` — sigmoid decode, hierarchy post-process
   - `nbs/10_train.ipynb` — thread `task` through train loop
   - `nbs/03_patches.ipynb` + `nbs/04_dataset_build.ipynb` — patch cache for K masks
   - `nbs/15_hydra_pipeline.ipynb` — config wiring
   - `nbs/13_export.ipynb` — ONNX metadata
6. Run `nbdev_export` after each notebook is green; run `nbdev_test` on touched notebooks.

---

## Executive summary

| Layer | Ready today? | Blocker |
|-------|--------------|---------|
| Model (`get_model`, SMP head) | **Mostly yes** | `classes=3` already outputs 3 logits; no arch change needed |
| Config / task semantics | **No** | `num_classes>1` implies softmax everywhere |
| Dataset / patch build | **Yes (Phases 1+3 done)** | `SegmentationDataset`/`DirectDataset` load K-channel masks; `patches.py`/`dataset_build.py` *build* a K-channel patch cache too (`mask_mode='stacked'`/`'per_class_files'`) |
| Losses | **Yes (Phase 0 done)** | `probs_and_targets(..., task='multilabel')` + `SegLoss(task=...)`; synthetic overlap test passes |
| Metrics | **Yes (Phase 2 done)** | `SegMetrics(task='multilabel')` per-channel sigmoid, no argmax, no background exclusion; `subset_violation_rate`/`hierarchy_diagnostics` added |
| Train / callbacks | **Yes (Phase 3+4 done)** | `task`/`mask_mode` threaded through `train()`; `metrics_threshold` actually wired to `SegMetrics`. `PredOverlayCallback` renders real per-channel GT/pred overlay JPEGs for multilabel via `save_multilabel_overlay_grid_jpeg` |
| Inference | **Yes (Phase 2+4 done)** | `predict(task='multilabel')` independent sigmoid; `decode(task='multilabel')` returns `(C,H,W)`, no argmax. `apply_hierarchy(masks, pairs)` hard-zeros sub-channel pixels outside their super-channel |
| Hierarchy | **Yes (Phase 4 done)** | Soft: `hierarchy_regularizer_loss` (`relu(p_sub - p_super).mean()`) wired into `SegLoss` via `w_hierarchy`/`hierarchy_pairs`. Hard: `apply_hierarchy()` at inference/post-process |
| ONNX export | **Yes (Phase 5 done)** | Exports logits (sigmoid applied downstream, not baked into the graph); `task`/`class_names`/`thresholds`/`hierarchy_pairs` written to `metadata_props` so a consumer can decode without hardcoding |
| Skills / docs | **Yes (Phase 5 done)** | `multi-class-segmentation` skill now has an overlap callout pointing to `task: multilabel`; `export-onnx` skill documents the metadata params; this doc's integration section has a runnable consumer-side snippet |

**Bottom line:** `train.num_classes=3` alone still defaults to softmax + argmax (multiclass) and destroys overlap semantics — you must also set `train.task=multilabel` (e.g. via `configs/train_multilabel_3head.yaml`). That combination, hierarchy enforcement, multilabel prediction overlays, and self-describing ONNX export are now implemented and verified end to end (Phases 0-5), including a synthetic run through the real `scripts/build.py`/`scripts/train.py` wiring all the way to a real `onnx.load()`-verified export.

---

## Current activation decision tree (the bug)

```
logits (B, C, H, W)
    │
    ├─ C == 1  → sigmoid → threshold     (binary)
    └─ C > 1   → softmax → argmax        (mutually exclusive — WRONG for overlap)
```

Key locations:
- `src/segmentation_tools_pt/losses.py` — `probs_and_targets`
- `src/segmentation_tools_pt/metrics.py` — `SegMetrics.update`
- `src/segmentation_tools_pt/inference.py` — `predict`, `decode`
- `src/segmentation_tools_pt/core.py` — `SegConfig` documents binary vs multiclass only

---

## Proposed third mode: `train.task`

Decouple **output channel count** from **activation semantics**.

### Config (new)

```yaml
train:
  task: multilabel          # binary | multiclass | multilabel
  num_classes: 3            # output channels K
  class_names: [chip, void, lifted]
  hierarchy:
    lifted_subset_of_chip: true

data:
  mask_mode: stacked        # single | stacked | per_class_files
  mask_channels: 3

metrics:
  thresholds: [0.5, 0.5, 0.5]   # per-head, multilabel only
```

| `task` | Channels | Activation | Targets | Decode |
|--------|----------|------------|---------|--------|
| `binary` | 1 | sigmoid | `(B,H,W)` {0,1} | threshold |
| `multiclass` | N | softmax | `(B,H,W)` int 0..N-1 | argmax |
| **`multilabel`** | K | **sigmoid per channel** | **`(B,K,H,W)` float {0,1}** | **per-channel threshold** |

Backward compatibility: if `task` omitted, `num_classes==1 → binary`, `num_classes>1 → multiclass`.

---

## Data layout

### Today (Phase 1 done)

`SegmentationDataset`/`DirectDataset` return `{'image': (C,H,W), 'mask': (H,W) long}` for
`mask_mode='single'` (unchanged), or `{'image': (C,H,W), 'mask': (K,H,W) float {0,1}}` for
`mask_mode='stacked'`/`'per_class_files'`. `extract_patches_from_sample` still accepts only
one `mask_path` -- Phase 1 covers *reading* K-channel masks, not *building* a K-channel patch
cache (that's Phase 3).

### On-disk layouts (both implemented)

**Option A — stacked mask PNG** (`mask_mode='stacked'`, requires `mask_channels`):

```
raw_dir/masks/sample_001.png   # 3-channel PNG: ch0=chip, ch1=void, ch2=lifted
```

**Option B — per-class files** (`mask_mode='per_class_files'`, requires `class_names`):

```
raw_dir/masks/sample_001_chip.png
raw_dir/masks/sample_001_void.png
raw_dir/masks/sample_001_lifted.png
```

Overlapping pixels are valid (void inside chip: both masks = 1) -- verified by test in
`nbs/06_dataset.ipynb` for both layouts, through the real augmentation pipeline.

---

## Module-by-module change list

| Module | Notebook | Changes |
|--------|----------|---------|
| `core.py` | `00_core.ipynb` | `TaskType`, `is_multilabel`, activation helper |
| `model.py` | `09_model.ipynb` | Doc only; `num_classes=3` logits unchanged |
| `losses.py` | `07_losses.ipynb` | Multilabel sigmoid path; lifted masked loss; subset regularizer |
| `metrics.py` | `08_metrics.ipynb` | Per-channel metrics; overlap diagnostics |
| `dataset.py` | `06_dataset.ipynb` | `(K,H,W)` mask tensor; albumentations for K channels |
| `patches.py` | `03_patches.ipynb` | K-mask patch extraction |
| `dataset_build.py` | `04_dataset_build.ipynb` | `resolve_mask_paths()` for K files |
| `train.py` | `10_train.ipynb` | Thread `task`; fix `metrics_threshold` wiring |
| `inference.py` | `11_inference.ipynb` | Multilabel sigmoid decode; `apply_hierarchy()` |
| `export.py` | `13_export.ipynb` | `task`/`class_names`/`thresholds`/`hierarchy_pairs` → `metadata_props` (JSON) |
| `hydra_pipeline.py` | `15_hydra_pipeline.ipynb` | Parse `task`, `mask_mode`, `resolve_hierarchy_pairs()` |
| `visualize.py` | `12_visualize.ipynb` | Multilabel RGB overlays |

---

## Workaround without library changes (available today)

Train **3 independent binary models** (`train.num_classes: 1`), separate mask dirs per class.
Run `segpt-build` + `segpt-train` three times. Use to bootstrap labels while building `multilabel` mode.

---

## Implementation order

1. **Phase 0 — done** — `task=multilabel`, fix `probs_and_targets` + `SegLoss`; synthetic overlap test in `07_losses.ipynb`
2. **Phase 1 — done** — Dataset K-mask loading in `06_dataset.ipynb`
3. **Phase 2 — done** — Metrics + inference in `08_metrics.ipynb`, `11_inference.ipynb`
4. **Phase 3 — done** — Patch pipeline + `configs/train_multilabel_3head.yaml`
5. **Phase 4 — done** — Hierarchy loss (`hierarchy_regularizer_loss`) + `apply_hierarchy()` + multilabel viz (`show_pred_multilabel`, `save_multilabel_overlay_grid_jpeg`) + `PredOverlayCallback` wiring
6. **Phase 5 — done** — `export_onnx()` metadata (`task`/`class_names`/`thresholds`/`hierarchy_pairs`), `resolve_hierarchy_pairs()` config resolver, `multi-class-segmentation`/`export-onnx` skill updates, `void_detection_example_line` consumer-side integration guide below

---

## Integration with void_detection_example_line

Segpt-side (this repo) is done as of Phase 5. This section is the handoff to the consumer
project, `void_detection_example_line`.

| Asset | Role |
|-------|------|
| `chip_detection.py` | Pseudo-label chip channel |
| `pseudo_labels.py` | Pseudo-label void channel |
| `lifted_chip.py` | Pseudo-label lifted channel |
| `segmentation/inference.py` | Consumer of the exported multilabel ONNX model |

### Training the 3-head model

```bash
segpt-train --config-name train_multilabel_3head \
  data.raw_dir=/path/to/raw data.patch_dir=/path/to/patches \
  train.hierarchy.lifted_subset_of_chip=true \
  export.onnx=true
```

`data/masks/<sample_id>_{chip,void,lifted}.png` (per-class files, `data.mask_mode:
per_class_files`) or a single 3-channel stacked PNG (`data.mask_mode: stacked`) both work — see
`configs/train_multilabel_3head.yaml`. `train.hierarchy.lifted_subset_of_chip: true` both trains
a soft subset regularizer (`hierarchy_regularizer_loss` inside `SegLoss`) and gets baked into the
exported ONNX file's `metadata_props` as `hierarchy_pairs` (see the `export-onnx` skill).

### Consuming the exported model

`checkpoints/model.onnx` carries `metadata_props` (`task`, `class_names`, `thresholds`,
`hierarchy_pairs` — JSON strings) so `segmentation/inference.py` doesn't need to hardcode any of
this:

```python
import json
import numpy as np
import onnx
import onnxruntime as ort

props = {p.key: p.value for p in onnx.load('model.onnx').metadata_props}
sess = ort.InferenceSession('model.onnx')
class_names = json.loads(props['class_names'])   # ['chip', 'void', 'lifted']
thresholds = json.loads(props['thresholds'])     # [0.5, 0.5, 0.5]
hierarchy_pairs = json.loads(props['hierarchy_pairs'])  # [[2, 0]]  (lifted subset of chip)

logits = sess.run(['logits'], {'image': image_batch})[0]
probs = 1 / (1 + np.exp(-logits))                # sigmoid, independent per channel -- NOT softmax
masks = probs >= np.array(thresholds)[None, :, None, None]

# hard-enforce the subset constraint (or use segmentation_tools_pt.inference.apply_hierarchy
# directly if consuming torch tensors instead of the raw ONNX output)
for sub, sup in hierarchy_pairs:
    masks[:, sub] &= masks[:, sup]
```

The equivalent PyTorch-side helper, if the consumer imports `segmentation_tools_pt` directly
instead of running the ONNX graph standalone, is `inference.apply_hierarchy(masks, pairs)`.

---

## Anti-patterns

- `train.num_classes=3` without `task: multilabel` — trains softmax, breaks overlap
- Single integer mask with priority painting — loses compositional semantics
- Training lifted as full chip duplicate mask — use chip-conditional target
- Using `multi-class-segmentation` skill for overlapping chip/void/lifted
- Editing `src/` directly — always `nbs/` → `nbdev_export`

---

## Test plan

| Test | Pass |
|------|------|
| Synthetic overlap | void⊂chip; void channel still learns interior void |
| Softmax guard | `task=multilabel` never calls softmax/argmax in loss/metrics/inference |
| Patch round-trip | K-mask build → dataset reload → `(K,H,W)` |
| Hierarchy | `lifted & chip` post-process → zero subset violations — **done**, `apply_hierarchy()` verified in `11_inference.ipynb` |
| ONNX | Output `(1,3,H,W)`; sigmoid parity PyTorch vs ORT — **done**, `13_export.ipynb`; metadata_props round-trip verified against a real `onnx.load()` |
| Backward compat | `train.num_classes: 1` configs unchanged |
