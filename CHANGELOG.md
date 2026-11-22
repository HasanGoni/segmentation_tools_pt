# Changelog

All notable changes to `segmentation_tools_pt` are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## 2026-11-22

RF-DETR bridge: instance-model data and output converters (#101, map #100, research #99). Nothing imports `rfdetr`; the functions take plain arrays and `pyproject.toml` is unchanged.

- `masks.mask_to_coco_instances(mask, ...)`: one COCO annotation per connected component of an `(H, W)` label map or `(K, H, W)` stack. Polygons are traced on a 2x grid so 1-px lines and 2-px dots keep a non-zero area; `segmentation='auto'` (default) writes RLE only for components with holes, `'polygon'` fills holes, `'rle'` always RLE; `epsilon` simplifies polygons (default 0, lossless). `masks.coco_instances_to_mask` is the inverse; `masks.write_coco_folder(patch_dir, out_dir, class_names)` writes `train/` + `valid/` with 8-bit PNGs and `_annotations.coco.json` from the patch cache (single-channel caches only; 16-bit divided by 257).
- Round trip mask -> COCO -> mask, per case (IoU, wrong pixels), measured with our decoder and again with `pycocotools` (`frPyObjects` + `decode`, throwaway venv; both agree to the pixel):

  | case | `auto` (default) | `polygon` | `rle` |
  |---|---|---|---|
  | 1-px thin diagonal line | 1.0000, 0 wrong px | 1.0000, 0 | 1.0000, 0 |
  | 1-px thin horizontal line | 1.0000, 0 | 1.0000, 0 | 1.0000, 0 |
  | 1-px dot | 1.0000, 0 | 1.0000, 0 | 1.0000, 0 |
  | 2-px dot (2x2) | 1.0000, 0 | 1.0000, 0 | 1.0000, 0 |
  | square with hole | 1.0000, 0 (RLE) | 0.8889, 100 (hole filled) | 1.0000, 0 |
  | island inside hole | 1.0000, 0 | 0.9339, 32 (ring hole filled) | 1.0000, 0 |

  `epsilon=0.5` on a radius-20 disc: 552 -> 104 numbers at IoU 0.975. Not verified: RF-DETR's own loader on the uncompressed RLE that `auto` writes for components with holes, and its resize to training resolution.
- `write_coco_folder` raises `ValueError` when a mask value exceeds the number of `class_names` (it used to write a `category_id` with no entry in `categories`). 16-bit images divided by 257 are lossy.
- `inference.detections_to_semantic(dets, shape, class_ids, rule='score', output='labels'|'stack', threshold)`: `(H, W)` long labels or `(K, H, W)` float best scores; higher score wins overlaps; equal scores: the later detection wins; class ids below `first_class_id` raise (default 1; RF-DETR's 2-class model returned 0-based ids, so pass `first_class_id=0`, see #102); works with `supervision.Detections`-like objects or a plain tuple; consumed by `ObjectMetrics`, `SegMetrics`, `review_report.Sample`.
- `tutorials/09_rfdetr_bridge.ipynb` (rfdetr cells optional, guarded by `RUN_RFDETR`), `docs/TRAINING_CLI.md` section, `choose-model-architecture` and `run-inference` skills. Recipe gotchas recorded: reload with `RFDETR.from_checkpoint` before `predict`; prefer `last_ema.pth` over `checkpoint_best_total.pth` on small data; ONNX is fixed batch 1.

---

## 2026-10-18

Custom UNet clean-ups from the #64/#65 reviews (#97, replaces #94, map #12). No change for valid configs: parameter counts and state_dict keys are pinned by a test for every bottleneck and block style.

- `get_param_groups` no longer probes for `.bottleneck`: a model may define `decoder_side_parameters()` (`CustomUNet` returns decoder, head, bottleneck, in that order); models without it (SMP, DINOv3) keep decoder + head.
- One `_BOTTLENECKS` table row per bottleneck (`build` and `prepare`) replaces the `== 'aspp'` / `== 'attention'` branches in the class, the validator and the factory; unused knobs are still not checked. Adding a bottleneck is one row (tested).
- A list, dict or `None` for `bottleneck`, `norm` or `block_style` now raises `ValueError` naming the key (was `TypeError` for lists and dicts).
- `_DEFAULT_BASE_CHANNELS` / `_DEFAULT_MAX_CHANNELS` replace three copies of the literals; docstring, comment and test hygiene (duplicated gradient/group checks are now shared or covered once for all bottlenecks).

Custom UNet config examples, skill and docs (#66, map #12).

- `configs/train_multilabel_3head.yaml`: `model.source: smp` is now a real key, so the documented `model.source=custom_unet` override works (it failed with "Key 'source' is not in struct" because that config takes its model from a package). `train.yaml` already had it.
- `nbs/09_model.ipynb`: a test composes every documented `custom_unet` command from both shipped configs and builds the model, so the docs cannot drift from the YAML.
- `docs/TRAINING_CLI.md`: the Custom UNet section is one reference (options table, one command per option, depth/norm/ASPP/attention/ONNX notes, random-init caveat: DINOv3/SMP stays stronger for layout shift until a warm-start option exists). `docs/agents/repo-guide.md` and tutorial 08 point to it. Commented `custom_unet` blocks in both configs and the `choose-model-architecture` skill were completed in #62-#65.

Custom UNet attention bottleneck (#65, map #12).

- `nbs/09_model.ipynb`: `model.custom_unet.bottleneck` now accepts `attention` (Step 7). New `_AttentionBottleneck(channels, heads)`: flatten the map to `H*W` tokens, pre-LayerNorm, one q/k/v projection, `scaled_dot_product_attention`, output projection, residual add, fold back; keeps `(B, C, H, W)`. New `attention_heads` (default 4) must be a positive int that divides the bottleneck channels `min(base_channels * 2**encoder_depth, max_channels)`; a bad value raises `ValueError` naming `custom_unet.attention_heads` before any model is built (the `CustomUNet` constructor checks it too). `bottleneck` stays one value, so `aspp` and `attention` are exclusive; `aspp_rates` is ignored for attention and `attention_heads` for the others (unused knobs are not validated; `CustomUNet` now only checks `aspp_rates` when `bottleneck='aspp'`). The `NotImplementedError` branch and `_PLANNED_BOTTLENECKS` are gone; `_BOTTLENECK_BUILDERS` builders take `(channels, rates, heads, norm, num_groups)`.
- Memory: cost is O(tokens^2), tokens = `(crop // 2**encoder_depth)^2`, i.e. `(H*W)^2` of the bottleneck map (512 crop, depth 4 = 1024 tokens fine; depth 3 = 4096 tokens heavy). No warning is raised.
- ONNX (FP32, opset 18, `dynamo=False`): fixed-size export matches onnxruntime to < 1e-4, and export with dynamic H/W also works and matches at other sizes (32, 48, 37x41). This needed the explicit `scaled_dot_product_attention` form: with `nn.MultiheadAttention` the dynamic-H/W graph runs only at the traced size (onnxruntime fails with a Reshape error elsewhere; reproduced in a test, and the attention model also goes through `export_onnx`). Legacy exporter (`dynamo=False`) is the one verified. There is no positional encoding; layout comes from the neighbouring conv features.
- Tests: shapes for odd / non-square / 1x1 maps, every block style x norm, heads-must-divide checks before building, mutual exclusivity and ignored knobs, `none` unchanged, optimizer-group coverage and gradients, positions really mix, ONNX smoke (fixed + dynamic). The old `NotImplementedError` assertions now expect the new behaviour.
- Configs, `choose-model-architecture` skill, `docs/TRAINING_CLI.md`, `CONTEXT.md` and `tutorials/08_custom_unet.ipynb` (section 1e) updated.

## 2026-10-17

Custom UNet ASPP bottleneck (#64, map #12).

- `nbs/09_model.ipynb`: new `model.custom_unet.bottleneck: none | aspp` and `aspp_rates` (default `[1, 6, 12, 18]`). `_ASPP` = parallel dilated 3x3 convs + a whole-map pooling branch, concat, 1x1 project; keeps `(B, C, H, W)`. The model gets a `.bottleneck` module between encoder and decoder (`nn.Identity` for `none`, so default state_dict keys and output are unchanged). New `trim_aspp_rates(rates, size)`: below 16 px per side, rates not smaller than the side are dropped (fallback `[1]`); `CustomUNet(bottleneck_size=...)` applies it itself (the factory passes `crop_size // 2**encoder_depth`); without `bottleneck_size` the rates are used as given. `bottleneck: attention` still raises `NotImplementedError` (#65); unknown values raise `ValueError`.
- `get_param_groups` now puts bottleneck parameters in the decoder group (they would otherwise be in no optimizer group).
- Tests: shapes for every norm/odd sizes, trim table, `none` identity, config checks before building, optimizer-group coverage, gradients reach every branch, dynamic-H/W ONNX smoke for the three norms.
- Configs, `choose-model-architecture` skill, `docs/TRAINING_CLI.md`, `CONTEXT.md` and `tutorials/08_custom_unet.ipynb` (section 1d) updated.

Custom UNet residual block style (#63, map #12).

- `nbs/09_model.ipynb`: new `model.custom_unet.block_style: double | residual`. The residual block is Conv3x3-Norm-ReLU-Conv3x3-Norm, add the skip (bias-free 1x1 conv when channels differ, identity otherwise), ReLU; used in encoder and decoder (decoder blocks take the concatenated `2*ch` input). One `_BLOCK_CLASSES` table maps style to block class; unknown values raise `ValueError` early (was `NotImplementedError` for anything but `double`). `bottleneck != none` still raises `NotImplementedError` (#64-#65).
- Tests: shapes with every norm incl. odd sizes, residual adds exactly the 1x1 projections' parameters (5384 at depth 3, base 8), skip carries signal, the one-epoch `train()` smoke now runs for both styles, residual added to the dynamic-H/W ONNX smoke.
- Configs, `choose-model-architecture` skill, `docs/TRAINING_CLI.md`, `CONTEXT.md` and `tutorials/08_custom_unet.ipynb` updated.

Custom UNet automatic depth and selectable norm (#62, map #12).

- `nbs/09_model.ipynb`: new `resolve_encoder_depth(depth, crop_size, bottleneck_min_size=32)`; `recommend` (nearest step, 512 crop -> 4 downs) and `auto` (`floor(log2(crop/min))`) are clamped to [3, 6], an integer is honoured as-is. New config key `model.custom_unet.bottleneck_min_size`.
- `norm: group | batch | instance` implemented in `_norm_layer`; unknown norms now raise `ValueError` (was `NotImplementedError`). Tests cover the depth table, clamps, GroupNorm divisibility for every stage (incl. base_channels 12/20/24), batch/instance shapes, BatchNorm train vs eval, and a dynamic-size ONNX smoke for both.
- Configs, `choose-model-architecture` skill, `docs/TRAINING_CLI.md` and `tutorials/08_custom_unet.ipynb` updated. Residual blocks and bottlenecks stay `NotImplementedError` (#63-#65).

Custom UNet tracer bullet: `model.source=custom_unet` builds a library-owned UNet (#61, map #12).

- `nbs/09_model.ipynb`: new `CustomUNet` (double-conv blocks, GroupNorm with a divisor-safe group count, explicit integer `encoder_depth`) exposing SMP-compatible `.encoder` / `.decoder` / `.segmentation_head`, so `get_param_groups` / `freeze_encoder` work unchanged.
- New `get_model_from_cfg(cfg, in_channels, crop_size, ...)` branches on `model.source` (`smp` default, behaviour unchanged; `custom_unet`). `recommend`/`auto` depth, residual blocks, batch/instance norm and ASPP/attention raise `NotImplementedError` until tickets #62-#65.
- `scripts/train.py` builds the model through `get_model_from_cfg` (private `_model_kwargs` removed); `configs/train.yaml` gains `model.source: smp` plus a commented `custom_unet` block.
- Review fixes: `custom_unet` runs with `freeze_epochs=0` and `encoder_lr=decoder_lr` (new `resolve_train_schedule`, warns, overridable via `model.custom_unet.freeze_epochs`/`encoder_lr`); `encoder_depth` (None/str/bool/non-positive) and `norm` are validated up front in `get_model_from_cfg`.
- Tests in the notebook: output shape at depth 4 on 1x1x64x64, smp regression, freeze/param groups, one-epoch `train()` smoke, FP32 ONNX smoke (skips without `onnx`/`onnxruntime`).

## 2026-10-13

Tighten the `_poster_masks` device/dtype regression tests (#73); test cells only, no library change.

- New CPU-runnable test in `nbs/12_visualize.ipynb`: a stand-in "device" logits object that rejects any use before `.cpu()`, so removing `.cpu()` fails without a GPU. Covers binary, multiclass, multilabel and `class_idx`.
- fp16/bf16 logits are checked against an fp32 reference, plus a bf16 threshold-boundary case (`sigmoid(-0.001)` rounds to 0.5 in bf16) that fails if `.float()` is removed.
- The CUDA test now runs on `copy.deepcopy` of `ml_model` / `bin_model` and asserts the shared fixtures stay on CPU.
- Audit: every remaining raw `.numpy()` in `visualize.py` is inside `_poster_masks` on a tensor already passed through `.float().cpu()`; all other helpers use `_to_numpy` (`.detach().cpu()`).

Fix `_poster_masks` on multiclass ignore labels and multilabel label maps (#72).

- New private `_one_hot` in `nbs/12_visualize.ipynb`: one-hot stacks are sized from the model's channel count and labels outside `[0, C)` (e.g. `255`) belong to no class instead of raising `IndexError`.
- Pixels with an ignore label also carry no prediction, so the diff column does not paint the ignored border as false alarms.
- A multilabel 2-D label map missing the top class now gives GT the same channel count as the prediction (was sized from the sample's own max label).

Fix `rank_samples`, `rank_samples_per_class`, `compute_patch_losses` and `save_error_poster` side effects (#71).

- New private `_eval_mode(model)` context manager in `nbs/12_visualize.ipynb`: the helpers restore each submodule's own train/eval flag (a frozen BatchNorm stays frozen), so calling them mid-training no longer leaves BatchNorm/Dropout in eval mode.
- `save_error_poster` closes its figure in a `finally`, so a failure while rendering no longer leaks it.

Fix `save_error_poster` crash on a CUDA model (#69).

- `_poster_masks` in `nbs/12_visualize.ipynb` now calls `.float().cpu()` on the logits before `.numpy()`, covering the binary, multiclass, multilabel and `class_idx` branches (and half-precision models).
- Added a CUDA regression test cell (skipped when no GPU is available); `ErrorAnalysisCallback` benefits too since it calls `save_error_poster` on the training device.

## 2026-10-11

Per-epoch train vs validation metrics with per-class IoU (#58).

- Added `flatten_epoch_metrics` in `nbs/10_train.ipynb`: `train_*` / `val_*` loss and metrics plus `train_iou_<class>` / `val_iou_<class>`; background (multiclass label 0) is never listed.
- `train_one_epoch` / `validate` now also return `train_per_class` / `per_class`; `history.json` gains `train_iou`, `train_dice`, `train_precision`, `train_recall`, `val_precision` and the per-class keys (existing keys kept).
- New `metrics.csv` beside `history.json`, rewritten from history each epoch (resume-safe). Console line shows `train_iou` next to `val_iou`; `console_per_class` adds per-class val IoU.
- Config: `monitoring.metrics_log.{per_class,console_per_class,csv}` in `configs/train.yaml` / `train_example.yaml`, wired in `scripts/train.py`; defaults preserve behaviour except the new columns.

---

## 2026-10-12

Class-hierarchy loss wired into training (#67, `lifted ⊂ chip`). Default off — configs without a weight behave exactly as before.

- `train(..., hierarchy_pairs=None, w_hierarchy=0.0)` builds `SegLoss` with the hierarchy term; when on, the epoch log / `metrics.csv` gain `train_loss_hierarchy`.
- `losses.resolve_loss_weights(epoch, phases, w_hierarchy=0.0)` now returns a **5-tuple** `(w_tversky, w_dice, w_boundary, w_bce, w_hierarchy)` (was 4) so `criterion.set_weights(*weights)` takes it directly; a phase may set its own `w_hierarchy`, else the run-wide value, else `0.0`. Callers that unpack exactly four values must be updated.
- New `losses.validate_hierarchy_loss`: with a weight `> 0`, fails before training for a non-`multilabel` task, unresolved pairs or out-of-range indices.
- New `hydra_pipeline.resolve_hierarchy_weight` (`train.hierarchy.weight`, default `0.0`); `scripts/train.py` passes pairs + weight. `configs/train_multilabel_3head.yaml` sets `weight: 0.5`.
- Docs: `docs/TRAINING_CLI.md`, `choose-loss-function` skill, and `tutorials/06_hierarchy_loss.ipynb` (wiring check + A/B with the real `train()` on synthetic data: mean leak 0.144 → 0.029 / 0.013 / 0.007 for weight 0.1 / 0.5 / warm-up; a large weight from epoch 0 cost chip IoU on the toy, so prefer a small weight or a warm-up phase).
- Still open on #67: the A/B on real data (acceptance item 5).

---

## 2026-10-08

Tiled-inference consolidation (#27, second slice of map #23, per spec in #24).

- `inference.predict_tiled` is now the **single** tiled-inference interface. The parallel Oct-1 cluster is deleted outright (no external callers existed, so retirement is deletion, not deprecation): `tiling_core` (`TileConfig`, `InferenceConfig`, `PaddingMode`, `MergeStrategy`, `PatchCoords`, `pad_to_divisible`, `unpad_image`), `tiling` (`TileProcessorONNX` + 3 tile-grid visualizers), `stitching` (`POSTPROCESSING_PRESETS`, `PatchAccumulator`, `TileMerger`, the known-broken `TileMergerONNX`, `build_postprocessing_pipeline`, merge visualizers), `model_wrapper` (ONNX-runtime wrappers), and `onnx_export` (`export_to_onnx` + helpers) — notebooks `nbs/00_tiling_core` … `04_onnx_export` removed with them.
- Salvage 1 — `gaussian_weight_map` moved into `nbs/11_inference.ipynb` as private `_gaussian_weight_map` (single consumer: `predict_tiled`; `inference.py` no longer imports `stitching`).
- Salvage 2 — Oct-1 reflect tile-edge padding folded in as `predict_tiled(..., padding_mode='zero' | 'reflect')` (default `'zero'`, bit-identical to before — parity pinned by a fixed-seed checksum test). `'reflect'` only matters when the image is smaller than one tile: `_reflect_idx`/`_reflect_pad` mirror-extend bottom/right for any pad size (unlike `F.pad`, which raises when pad ≥ dim), so real edge structure is not zero-darkened. Gaussian-weighted stays the only merge strategy (confidence-weighted deferred in the spec, not dropped).
- Public API surface grows by exactly one parameter (`padding_mode`); batch-size invariance (`inference_batch_size=1` ≡ larger) stays bit-identical and under test.
- `nbs/llms.txt` regenerated to list the current modules (it still named the retired ones); `__init__.py` docstring refreshed.
- Test-harness gating: machine-specific demo cells in `nbs/09_model.ipynb` (DINOv3 hub download), `nbs/14_cli.ipynb` (placeholder repo paths, live `segpt_train()` demo, dry-run chain) and `nbs/003_patches_skill_test.ipynb` (placeholder dataset) marked `#| eval: false` so `nbdev-test` is fully green; code stays visible in the notebooks for manual runs. Un-hidden pre-existing rot fixed: `_build_job_script` now emits `uv sync --quiet` after entering the repo — matching its own test cell and the production `hpc/submit.sh`.

---

## 2026-10-03

Shared activation helper + tutorial (#25, first slice of map #23).

- Extracted the sigmoid/softmax activation rule duplicated between `predict` and `predict_tiled` into one private helper `_activate(logits, task, activate)` (`nbs/11_inference.ipynb`); both public functions delegate to it (kept out of `__all__`), behaviour unchanged — notebook tests pin binary `C==1` sigmoid, multilabel sigmoid, multiclass softmax, and `activate=False` logits passthrough.
- Added `tutorials/05_activation_semantics.ipynb` (executed, CPU-only synthetic data): walks the four pinned activation cases, softmax-vs-sigmoid channel semantics, the `C==1`-wins precedence, single-tile `predict_tiled` ≡ `predict` (to float32 rounding), tile batch-size invariance of the stitching, and bit-exact manual `_activate` reproduction. Note: on CPU the conv kernel itself may pick different algorithms per tile-batch size (~1e-7); the library's bit-identity test covers the single-tile case.

FP32 ONNX export + batched tile inference + loader prefetch (#20, from spec in #14).

- Added `predict_tiled` / `tile_starts` (`nbs/11_inference.ipynb`): batched sliding-window inference for large images — tiles forward in chunks of `inference_batch_size` and stitch with `stitching.gaussian_weight_map`. Clamped tile starts cover every pixel (last tile shifted back; images smaller than one tile are zero-padded up); batch size is a pure memory ↔ throughput knob — `inference_batch_size=1` is bit-identical to any larger value (test pins it). Activation semantics match `predict` (sigmoid binary/multilabel, softmax multiclass; `activate=False` logits passthrough); bad `overlap`/`inference_batch_size` raise clear `ValueError`s.
- Added `resolve_inference_cfg` (`nbs/15_hydra_pipeline.ipynb`): maps the `inference.*` config block (`tile_size: 256`, `overlap: 32`, `inference_batch_size: 8`, `prefetch_factor: null`) to `predict_tiled` knobs, validated before the GPU is touched; missing block → defaults. `prefetch_factor` targets loader-driven tiled-inference consumers — `predict_tiled` batches in-process and ignores it.
- FP32 ONNX regression smoke (`nbs/13_export.ipynb`): curated encoders (`convnext_tiny`, `efficientnetv2_s`) build via `get_model(pretrained=False)` and export through `export_onnx` (dynamic batch, opset 18, `task`/`class_names` metadata) with ONNX Runtime output verified against PyTorch. FP32 only — no INT8, no TensorRT. DINOv3 FP32 export at `in_channels=1`, multilabel 3-head was validated separately (timm-hub weights pull is heavier than a smoke).
- DataLoader prefetch knob (`nbs/06_dataset.ipynb` + `nbs/10_train.ipynb`): `prefetch_factor` threads through `build_dataloaders` / `build_direct_dataloaders` and `train()`'s internal patch-mode loaders via guarded `_prefetch_kwargs` (torch rejects the kwarg when `num_workers == 0`, so it is inert without workers). Config: `train.prefetch_factor` (default null = torch's 2 batches/worker), wired through `scripts/train.py`.
- Config blocks `inference.*` + `train.prefetch_factor` added to `configs/train.yaml` / `train_example.yaml` / `train_multilabel_3head.yaml`; documented in `docs/TRAINING_CLI.md` (production tiled-inference section) and the `run-inference` skill.

## 2026-10-02

Class-targeted morph postprocess (#19, from spec in #14).

- Added `apply_class_morph` (`nbs/11_inference.ipynb`): cleans **one named class channel** of a decoded multilabel mask with classical morphology — opening (erode→dilate, outside-image treated as background, so border-touching speckles are removable) followed by connected-component min-area filtering. Other channels pass through untouched; `(C,H,W)` and `(B,C,H,W)`, no input mutation; bad `class_idx`/`kernel_size`/`min_area` raise a clear `ValueError`.
- Fixed inference decode order: threshold decode (`decode`) → class morph (`apply_class_morph`) → hierarchy (`apply_hierarchy`). The order test pins morph-then-hierarchy by cleaning the *super* channel: a `void` speckle under a removed `chip` speckle must die with it.
- Inference/eval/production decode **only** — never applied to the training loss or to raw train metrics (those judge the model as-is).
- Added `resolve_class_morph_cfg` (`nbs/15_hydra_pipeline.ipynb`): maps the config block `postprocess.class_morph.*` (master switch default **false**; `class_name`, `kernel_size: 3`, `min_area: 16`) to `apply_class_morph` args, resolving `class_name` against `data.class_names`. Disabled/missing block → `None` (strict no-op); enabled without a name, no `class_names`, or unknown `class_name` → clear `ValueError`.
- Config block added to `configs/train.yaml` / `train_example.yaml` / `train_multilabel_3head.yaml` (default off; product recipes may set e.g. `class_name: void`).
- Documented in `docs/TRAINING_CLI.md` and the `run-inference` skill.

Batch-level augmentations, YOLO-style pick-one (#17, from spec in #14).

- Wire `mosaic` / `mixup` / `cutmix` / `copy_paste` into `train_one_epoch` (`nbs/10_train.ipynb`): applied on the batch after it lands on the device, before forward. `mosaic` = per-sample 2x2 canvas (self + 3 rolled donors) at `(2H, 2W)` with a random crop back to `(H, W)` (no-op when `B < 4`); `mixup` = convex blend with a permuted partner, scalar `lam ~ Beta(32, 32)` (mild blending, masks stay mostly intact); `cutmix` = one random box (image + mask) swapped with a permuted donor; `copy_paste` (`mode: flip`) = paste a horizontally-flipped donor's foreground, masks merge via maximum. Shape-generic over mask layouts `(B,H,W)`, `(B,1,H,W)`, `(B,K,H,W)`; metrics/loss see augmented targets (YOLO convention), validation untouched.
- YOLO-style **pick-one**: per batch, one uniform draw walks cumulative thresholds in a fixed order (`mosaic → mixup → cutmix → copy_paste`), so each type fires with exactly its configured marginal probability and at most one type applies per batch. All randomness (including donor permutations) flows through the seeded Python `random` instance — runs stay reproducible via `set_seed`.
- Config: `augment.batch.enabled` master switch (default **false**) + per-type probs (default **0**) + `copy_paste_mode` in `configs/train.yaml` / `train_example.yaml` / `train_multilabel_3head.yaml`; stale top-level placeholder probs (`mixup: 0.05`, `copy_paste: 0.11`, `close_mosaic`) removed. Master off or all-zero probs is a strict no-op — notebook test asserts all-zero training numbers are identical to `batch_augs=None`.
- Wiring: `resolve_batch_aug_cfg` (`nbs/15_hydra_pipeline.ipynb`, returns `None` when disabled/zero) → `scripts/train.py` passes `batch_augs=` to `train()` and logs the active set; `train()` threads it into `train_one_epoch`.
- Notebook tests (synthetic): pick-one marginals match configured probs over 20k draws; per-aug pixel/mask semantics (union masks, donor pixels pasted, B<4 mosaic no-op); dispatcher on/off/deterministic-prob-1.0; `train_one_epoch` off==zero equivalence + enabled run finite; `train()` smoke with batch augs.
- Updated `AGENTS.md`, `docs/TRAINING_CLI.md` (recipe note: keep flips off for orientation-sensitive X-ray; multiclass integer masks — prefer cutmix/copy_paste over mixup), and the `choose-augmentation` skill.
- Added `tutorials/04_augmentation_playground.ipynb` (linked from `docs/TRAINING_CLI.md`): point it at your own `IMAGE_PATHS` / `MASK_PATHS` (or its synthetic demo data) to visualize every batch-aug type forced on, realistic pick-one sampling over virtual batches, and the sample-level Albumentations stack. Demo-only — no `#| export`, not run by `nbdev-test`.

Encoder drift check callback (#21, from spec in #14).

- Added `EncoderDriftCheck` (`nbs/10_train.ipynb`, `TrainCallback` order 40): digests the encoder `state_dict` at `on_train_begin` (sorted keys + float32 bytes) and compares at `on_train_end`. Bit-identical → confirmation print; changed → per-layer cosine similarity report (worst layers first, `max_report` capped). With `expect_frozen=True` a changed encoder raises `RuntimeError` — the deterministic guarantee behind freeze-forever (`freeze_epochs: -1`), since AdamW never updates params without grads. Models without `.encoder` are a no-op. Note: activation histograms are *not* a frozen-ness check — they shift because augmentation changes inputs, not because weights moved.
- Wired via `build_default_callbacks(..., drift_check=..., drift_expect_frozen=...)` and `monitoring.encoder_drift_check` (`enabled`, `expect_frozen: null` = derive from `train.freeze_epochs == -1`) in `configs/train.yaml` / `train_example.yaml` / `train_multilabel_3head.yaml` (enabled there for the DINOv3 recipe) / `scripts/train.py`. Default **off**; existing runs unchanged.
- Notebook tests on synthetic encoder/decoder/head data: freeze-forever run passes the guard, drifted run reports and raises under `expect_frozen`, no-`.encoder` no-op, wiring via `build_default_callbacks`. Updated `AGENTS.md` monitoring table and the `training-monitoring-callbacks` skill.

Freeze-forever training (`freeze_epochs: -1`, #16, from spec in #14).

- `train()` (`nbs/10_train.ipynb`) now freezes the encoder whenever `freeze_epochs != 0`: `0` = no freeze (unchanged), `N > 0` = warm-up then unfreeze at epoch `N` (unchanged), `-1` = encoder frozen for the entire run — only the decoder and segmentation head train. The unfreeze condition (`freeze_epochs > 0 and epoch == freeze_epochs`) never fires for `-1` since epochs start at 0, so freeze-forever needs no extra branch.
- Notebook tests on a synthetic encoder/decoder/head model track per-epoch `requires_grad` at the first training batch (batch-level, because `on_epoch_begin` fires before the epoch's unfreeze): `-1` → frozen every epoch, `0` → never frozen, `2` → frozen epochs 0–1, unfrozen at epoch 2.
- Config comments (`configs/train.yaml`, `train_example.yaml`, `train_multilabel_3head.yaml`), `docs/TRAINING_CLI.md` encoder-freeze note, `AGENTS.md`, and the `configure-training-schedule` skill now document the three-way semantics.

Post-training error analysis callback (#18, from spec in #14 / decision in #9).

- Added `ErrorAnalysisCallback` (`nbs/10_train.ipynb`): fires `on_train_end`, ranks the worst samples per split and writes 4-column JPEG posters under `checkpoints/error_analysis/` — image | GT overlay (α from config, default 0.1) | pred overlay | difference overlay (FN green / FP red / overlap magenta) — plus a small `top_ranked.json` (`sample_id`, `split`, `score`, `rank_by`, optional `class`). Default **off**; existing runs unchanged.
- Added ranking helpers in `nbs/12_visualize.ipynb`: `rank_samples` (per-sample worst-first records; `rank_by: loss` highest-first, `iou|dice|recall|precision` lowest-first via `SegMetrics` on single samples), `rank_samples_per_class` (multilabel per-head ranking; loss ranking slices the class channel and scores it with a binary copy of the criterion), and `save_error_poster` (top-k 4-column poster; multilabel per-channel rendering with class legend, multiclass one-hot columns).
- Multilabel `per_class: true` writes one poster per class head (`{split}_top{k}_{class}.jpg`); no combined poster.
- Wired via `build_default_callbacks(..., error_analysis=..., ea_top_k=..., ea_rank_by=..., ea_splits=..., ea_alpha=..., ea_jpeg_quality=..., ea_per_class=...)` and `monitoring.error_analysis` in `configs/train.yaml` / `train_example.yaml` / `scripts/train.py` (explicit kwargs only, same pattern as activation stats / pred overlays).
- Notebook tests on synthetic data: binary train+val posters + worst-first JSON + default-off preserved; multilabel per-class posters ranked per head. Full suite clean for `10_train.ipynb` / `12_visualize.ipynb`.
- Updated `AGENTS.md` monitoring table, `training-monitoring-callbacks` and `segmentation-error-analysis` skills.

## 2026-10-01

Curated encoders + Hydra encoder preset packages (#15).

- `get_model` / `list_smp_encoders` treat curated aliases as first-class: `convnext_tiny` → `tu-convnext_tiny`, `efficientnetv2_s` → `tu-tf_efficientnetv2_s`; raw curated `tu-*` builds without warnings; other `tu-*` warn once; unknown non-`tu-*` hard-error.
- Added Hydra packages under `configs/model/` (`mobilenet_v2`, `dinov3_vit_small`, `convnext_tiny`, `efficientnetv2_s`) setting `model.*` only. `configs/train.yaml` composes MobileNet (library default); `train_multilabel_3head.yaml` composes DINOv3-S (primary recommendation). Shut-off = `model=mobilenet_v2`.
- Updated `choose-model-architecture` skill for curated vs experimental guidance.
- Synthetic nbdev tests: allowlist/warn-once, Hydra compose, curated `export_onnx` smoke via `get_model`.
- Rewrote the curated section in `nbs/09_model.ipynb` as literate **experiment → verify → promote to `#| export`** (alias map, raw `tu-*` forward, allowlist policy, discovery list), then API/Hydra/ONNX tests under the export.
- Fixed ImageNet weight scratch cells in `nbs/09_model.ipynb`: resolve a real file (portable `.pth` or hub `model.safetensors`), never `torch.load` the HF hub directory; save `/tmp/.../mobilenet_v2_encoder.pth` for later air-gapped `get_model(weights_path=...)`.
- Added `validate_mask_mode` in `nbs/15_hydra_pipeline.ipynb` (strict companion-field checks for stacked / per_class_files).
- Agent/domain docs: `CONTEXT.md`, `docs/agents/*`, design specs under `docs/superpowers/specs/`; refreshed learned preferences in `AGENTS.md` (literate+implement, finish-one-map, config-driven class names, skill sync).
- Refined GitHub issue #18 (error-analysis callback): 4-column JPEG posters, train+val, `rank_by` loss|metric (tracker only).


## 2026-09-27

Added a DINOv3 encoder-decoder (`nbs/09_model.ipynb`) — the library had no ViT-based backbone, only smp's CNN encoders.

- Added `Dinov3ReassembleEncoder`: wraps `timm`'s `vit_*_patch16_dinov3` (already available via `timm>=1.0.28`, added as a direct dependency) and reassembles 4 tapped transformer layers into a stride `[2,4,8,16,32]` pseudo-pyramid (DPT-style), the missing stride-2 skip synthesized by upsampling the stride-4 map. Registered into `smp.encoders.encoders` as `dinov3_vit_small` / `dinov3_vit_base`, so it works through the existing `get_model(arch='dinov3_vit_small', architecture='unet'|'fpn'|..., ...)` call — no new API, `get_param_groups` / `freeze_encoder` / `unfreeze_encoder` / Hydra configs unchanged.
- Pretrained DINOv3 weights load via `timm`'s own hub download at construction time, not smp's `encoder_weights='imagenet'` state-dict loader (which would `load_state_dict` the whole encoder against a checkpoint missing the reassemble layers); `get_model()` threads `pretrained` through the registry's construction params instead and forces `encoder_weights=None` for `dinov3_*` encoders.
- Single/multi-channel input (`in_channels=1,2,3,...`) works via the existing `EncoderMixin.set_in_channels` → `patch_first_conv` path, reusing the same weight-averaging convention as every other encoder in the library.
- Fixed a pre-existing `NameError`/`FileNotFoundError` in `nbs/09_model.ipynb`'s local-weights scratch cells (missing `Path` import) by marking them `#| eval: false` so `nbdev-test` skips them instead of failing before reaching the exported cells.

## 2026-07-16

Single-channel patch naming fix + HPC workflow gate + patch-config update for new dataset.

- Fixed `extract_patches_from_sample` in `nbs/03_patches.ipynb` so single-channel patches use identical image/mask base names (`{pid}.png`) with no `_ch0` or `_mask` suffix. Multi-channel sources continue to save images as `{pid}_ch{i}.png`.
- Updated `SegmentationDataset` (`nbs/06_dataset.ipynb`) and `iter_train_images` (`nbs/15_hydra_pipeline.ipynb`) to load both the new plain filenames and legacy `_ch{i}` / `_mask` filenames for backward compatibility.
- Added `.github/skills/hpc-ready-gate/SKILL.md` — gate-keeper skill that caches HPC connectivity and repo context in `.segpt/`, verifies passwordless SSH, runs `configure-repo` once, and routes to the correct downstream HPC skill.
- Added `.copilot-instructions.md` with the mandatory HPC workflow gate: whenever HPC/cluster/bsub/segpt-submit/GPU queue is mentioned, run `hpc-ready-gate` first, ask `local or HPC?`, and ask for `raw_dir` / `patch_dir` before editing configs.
- Updated `configs/data_patch.yaml` and `configs/train.yaml` for the `example_dataset/curated8bit/train` dataset: grayscale `single` mode, 1-channel input, `patch_size=656`, `stride=328` (50% overlap), `crop_size=656`.
- Added `scripts/patch_656_example.py` — standalone script to extract 656×656 patches with 50% overlap from the example curated dataset, with optional `bsub` submission (`--submit`) to the HPC batch queue.
- Updated `.github/skills/choose-channels/SKILL.md` and `.github/skills/choose-tiling-strategy/SKILL.md` to state the default image/mask naming convention explicitly: image and mask must share the same base name unless `data.mask_suffix` or multi-channel `data.data_mode` is configured. Warns against inventing `mask_` prefixes or `_ch0` tags without matching config.
- Verified the single/multi-channel naming fix end-to-end on HPC: a synthetic build + `SegmentationDataset` load check confirms single-channel patches share identical image/mask names with no `_ch0`/`_mask` suffix, multi-channel patches keep `_ch{i}` on images only, and `uv run nbdev-test --path nbs/03_patches.ipynb` / `nbs/06_dataset.ipynb` both pass.
- Consolidated the nbdev CLI command reference into `.github/skills/nbdev-export/SKILL.md`: it now also documents `nbdev-test` (never the legacy `nbdev_test` or `python -m nbdev.test`), with a full v1-vs-v2 command table and matching troubleshooting rows.
- Migrated Cursor-specific HPC skills into `.github/skills/`: added `configure-repo`, `hpc-cpu-submit`, `hpc-gpu-submit`, `hpc-jupyter-connect`, `ssh-hpc-connect`, `nbdev-notebook-verify`, plus `.github/prompts/hpc-submit.prompt.md` and `.github/scripts/{hpc_repo_context.sh,discover_repo_root.sh}`; removed the superseded `.cursor/skills/nbdev-notebook-verify` copy and the unused `docker-containerization` placeholder skill.
- Added `docs/superpowers/{specs,plans}/2026-07-16-patch-656-hpc-*.md` design/plan notes for the 656×656 patch extraction work on the example dataset.

## 2026-07-11

Context: Added repository-local Copilot skill documentation for recurring workflows.

- Added `.github/skills/git-last-change-summary/SKILL.md` to standardize changelog + commit-message generation from real git diffs.
- Added `.github/skills/fast-network-data-copy/SKILL.md` with a parameterized `robocopy` workflow for high-throughput network-to-local dataset transfer.
- Added `.github/skills/lifted-chip-mask-synthesis/SKILL.md` documenting physically motivated synthetic lifted-chip defect and mask generation.
- Added a modular, interview-style skill suite under `.github/skills/` mirroring the library + CLI, framed as a deep-learning and classical-CV expert panel:
  - Orchestrator `segmentation-project-planner` routes a project interview to focused sub-skills and emits matched `build.yaml` / `train.yaml` + `segpt-*` commands.
  - Decision skills `choose-tiling-strategy`, `choose-loss-function`, `choose-augmentation`, `choose-channels`, `choose-model-architecture`, `configure-training-schedule`, `configure-metrics-checkpointing`, `training-monitoring-callbacks`, `run-inference`, `export-onnx`, `submit-to-hpc`.
  - Classical-CV skills `mask-preprocessing` and `feature-channels` (partial) plus placeholder `classical-segmentation-baseline`, all flagging unimplemented features per the "placeholder if not in the library" convention.
  - `choose-augmentation` documents that `augment` preset / `mosaic` / `mixup` / `cutmix` / `copy_paste` and per-knob controls are config-only placeholders (`scripts/train.py` passes `aug=None`); only the built-in transform stack + `boundary_blur` (+ optional `flips`) are active.
- **Corrected `AGENTS.md`** (`transforms.py` module guide): stated that `augment` presets and batch-level augmentations are config-only placeholders that are **not** currently applied, matching the actual `scripts/train.py` behavior (`aug=None`), instead of incorrectly claiming they are applied inside the train loop.
- Added advanced-practitioner skills for features already implemented in the library but not yet covered:
  - `segmentation-eda-validation` — pre-training checks with `sanity_check`, `show_batch`, `stats.json`, `platform_info`.
  - `segmentation-error-analysis` — post-training `show_top_losses`, `compute_patch_losses`, `find_operating_threshold`.
  - `object-level-metrics` — `ObjectMetrics` / `object_metrics`, `overlap_thresh`, detection rate vs pixel metrics.
  - `test-time-augmentation` — `predict(..., tta=True)` via 90° rotation averaging.
  - `multi-class-segmentation` — config differences when `train.num_classes > 1` (softmax, integer labels, argmax decode).
  - `air-gapped-encoder-weights` — download → copy → `model.weights_path` workflow for offline HPC nodes.
- Added `custom-callbacks-howto` skill with `TrainCallback` templates for early stopping, LR scheduling, and experiment-tracking hooks.
- Added `model-ensembling` skill showing how to average multiple checkpoints / TTA probabilities.
- Added `dataset-imbalance-strategies` skill covering patch filtering, loss alpha/beta/gamma, boundary loss, and object-level monitoring for sparse defects.
- Added `mixed-precision-deep-dive` skill for tuning AMP and debugging NaN/Inf.
- Added `learning-rate-finder` placeholder skill documenting the missing LR range test and where it would be implemented.
- Added `pseudo-labeling` placeholder skill for self-training / semi-supervised segmentation.
- Added `active-learning` placeholder skill for uncertainty-based sample selection.
- Added `experiment-tracking` placeholder skill for W&B/TensorBoard/MLflow integration via TrainCallback.
- Added `model-interpretability` placeholder skill for Grad-CAM / attention maps / feature visualization.
- Added `distributed-training` placeholder skill for multi-GPU DDP training.
- Added `model-comparison-ab-test` placeholder skill for statistical comparison of two checkpoints.
- Added `production-deployment` placeholder skill for ONNX Runtime / TorchScript / serving patterns.
- Added `data-versioning` placeholder skill for dataset versioning and reproducible splits.
- Added `continuous-integration` placeholder skill for nbdev test/docs CI pipeline.
- Added `docker-containerization` placeholder skill for containerized training/inference environment.
- Added `security-data-privacy` placeholder skill for sensitive-image handling, anonymization, and encrypted-artifact considerations.
- Added `skill-smoke-test` skill with reusable notebook `.github/skills/skill-smoke-test/notebooks/test_skills.ipynb` to validate `SKILL.md` frontmatter, discoverability, and broken references before committing.
- Added `tutorials/03_testing_skills.ipynb` explaining how to use the smoke-test notebook.
- Added `nbdev-export` skill documenting the HPC nbdev v2 workflow (`nbdev-export`, `nbdev-readme`) and the fact that `README.md` is maintained manually in addition to `nbs/index.ipynb`.
- Updated `README.md` development workflow to reference the `nbdev-export` skill and added a "Testing Copilot skills" section.
- Added `nbdev-uv-new-project` skill for scaffolding a new nbdev project on top of an existing uv-managed Python project, including the temporary `pyproject.toml` rename dance and auto-detected git/system defaults for `nbdev-new`.

## [Unreleased] — 2026-06-19

### Added

- **Activation monitoring** (`nbs/10_train.ipynb`, `nbs/12_visualize.ipynb`) —
  `ActivationStatsCallback` records per-layer activation histograms during training
  and saves fastai-style colorful-dimension PNGs plus near-zero fraction plots under
  `checkpoints/activation_stats/`. Enabled by default via `monitoring.activation_stats`.

- **Validation overlay JPEGs** (`nbs/10_train.ipynb`, `nbs/12_visualize.ipynb`) —
  `PredOverlayCallback` saves YOLO-style `val_batch{N}_labels.jpg`, `*_pred.jpg`, and
  `*_compare.jpg` mosaics each epoch under `checkpoints/predictions/`. Enabled by
  default via `monitoring.pred_overlays`.

- **`monitoring:` Hydra section** (`configs/train.yaml`) — toggles and knobs for both
  callbacks. `scripts/train.py` wires `build_default_callbacks()` into `segpt-train`.

- **Training callbacks** (`nbs/10_train.ipynb`) — `TrainState`, `TrainCallback`, and optional
  `callbacks=` on `train()`, `train_one_epoch()`, and `validate()`. Hooks fire at train / epoch /
  batch / checkpoint granularity; set `state.stop_training = True` for early stopping. Custom
  callbacks can be added alongside the default monitoring pair. Documented in `AGENTS.md`.

- **Any smp architecture + encoder** (`nbs/09_model.ipynb`) — `get_model()` accepts
  `architecture=` (segmentation head: `unet`, `fpn`, `deeplabv3plus`, …) and `arch` / `encoder`
  (backbone). Helpers `list_smp_architectures()` and `list_smp_encoders()`. Hydra:
  `model.architecture` in `configs/train.yaml` (defaults to `unet` for backward compatibility).

- **`hydra_pipeline.py`** (`src/segmentation_tools_pt/`, `nbs/15_hydra_pipeline.ipynb`) — new
  package module that consolidates all shared Hydra config helpers previously inlined in
  `scripts/train.py`. Exports: `CHANNEL_PRESETS`, `resolve_channel_preset`, `resolve_run_seed`,
  `resolve_deterministic`, `validate_patch_training_sizes`, `patches_exist`, `iter_train_images`,
  `iter_raw_train_images`, `load_or_compute_stats`, `load_or_compute_stats_direct`,
  `run_patch_pipeline`, `run_direct_pipeline`, `validate_data_mode_channels`. Both
  `scripts/train.py` and `scripts/build.py` now import from this module.

- **`scripts/build.py`** — standalone Hydra entry point for patch-cache preparation. Driven by
  `configs/build.yaml` (or any build config); validates channel preset, seeds the run, logs
  platform info, and delegates to `run_patch_pipeline`. Complements `scripts/train.py` for
  workflows that separate patch building from model training.

- **`configs/build.yaml`** — Hydra config for `segpt-build` / `scripts/build.py`. Inherits
  shared data+patch settings via `defaults: [data_patch]`; sets `data.build_only=true` and
  HPC defaults tuned for CPU-only patch-generation jobs.

- **`configs/build_example.yaml`** — template build config with `???` placeholders for
  `raw_dir` and `patch_dir`; copy and edit for new datasets.

- **`configs/data_patch.yaml`** — new shared Hydra config fragment for data and patch settings.
  Included by both `train.yaml` and `build.yaml` via `defaults: [data_patch]` so data paths,
  channel mode, patch parameters, and seed stay in sync across build and train runs.

### Fixed

- **`_build_job_script()`** (`cli.py`, `nbs/14_cli.ipynb`) — `segpt-submit` crashed with
  `TypeError: _build_job_script() got an unexpected keyword argument 'build_job'` because
  the `build_job` parameter was added to the call site but never added to the function
  signature. Also fixed two undefined names referenced inside the function body (`cmd_name`,
  `run_cmd`). The function now derives both from `build_job`: `segpt-build` when true,
  `segpt-train` otherwise.

### Added

- **`BCELoss`** (`losses.py`, `nbs/07_losses.ipynb`) — new exported loss component wrapping
  `F.binary_cross_entropy_with_logits` (binary) and `F.cross_entropy` (multi-class) behind
  the same interface as the other components. Use `SegLoss(w_tversky=0, w_dice=0.5, w_bce=0.5)`
  when the foreground region is large (e.g. a chip fills most of the field of view) and the
  class-imbalance correction of Tversky-focal is unnecessary.

- **`SegLoss.w_bce`** (`losses.py`, `nbs/07_losses.ipynb`) — new `w_bce: float = 0.0`
  constructor parameter and corresponding weight in `forward()` and `set_weights()`. Defaults
  to `0.0` so all existing configs are unaffected.

- **`resolve_loss_weights()`** now returns a 4-tuple `(w_tversky, w_dice, w_boundary, w_bce)`.
  The `w_bce` value is read from the phase dict via `phase.get('w_bce', 0.0)`, so existing
  phase configs without `w_bce` continue to work unchanged. The extra value unpacks cleanly
  into the updated `set_weights()`.

### Changed

- **`scripts/train.py`** — refactored to delegate all shared logic (`CHANNEL_PRESETS`,
  `_resolve_channel_preset`, `_resolve_run_seed`, `_resolve_deterministic`,
  `_validate_patch_training_sizes`, `_patches_exist`, `_iter_train_images`,
  `_iter_raw_train_images`, `load_or_compute_stats`, `run_patch_pipeline`, etc.) to the new
  `hydra_pipeline` module. Script is now ~30 lines of orchestration instead of ~260 lines.

- **`configs/train.yaml`** — refactored to use `defaults: [data_patch]` for shared data/patch
  settings; overrides only training-specific values (`data.build_only`, `model`, `train`, `loss`,
  `augment`, `hpc`). Removed verbose inline comments (now live in `data_patch.yaml`).

- **`cli.py`** (`nbs/14_cli.ipynb`) — `segpt_build` now invokes `scripts/build.py` instead of
  `scripts/train.py`; added `_find_build_script()`, `_run_build_script()`, and the generic
  `_run_hydra_script(script, *overrides)` helper to avoid duplicating argv-rewrite logic.

---

## [Unreleased] — 2026-06-17

### Added

- **`segpt-build`** (`pyproject.toml`, `cli.py`, `nbs/14_cli.ipynb`) — new first-class CLI
  entry point for patch-cache creation from the same Hydra config used by training. Forces
  `data.build_only=true`, builds or reuses patches, computes `stats.json`, and exits before
  model creation.

- **`data.build_only`** (`configs/train.yaml`, `configs/train_example.yaml`, `scripts/train.py`) —
  explicit build-only mode for patch workflows. Allows patch generation to be triggered from the
  CLI, from `python scripts/train.py`, or through HPC submission without starting training.

- **`segpt-submit --build-only`** (`cli.py`, `nbs/14_cli.ipynb`, `docs/TRAINING_CLI.md`) — HPC
  submission shortcut for patch creation. Submits an LSF job that forwards
  `data.build_only=true`, making patch generation a first-class cluster workflow for large datasets.

- **Run-level reproducibility controls** (`configs/train.yaml`, `configs/train_example.yaml`,
  `scripts/train.py`, `dataset.py`, `train.py`) — added `train.seed` and
  `train.deterministic` config knobs. The runtime now seeds Python, NumPy, and PyTorch, and
  DataLoader workers are seeded from torch worker seeds for reproducible sampling and train-time
  crops across repeated runs.

### Changed

- **Patch-size versus crop-size UX** (`configs/`, `README.md`, `docs/TRAINING_CLI.md`) — the
  intended workflow is now documented explicitly: `patch.patch_size` controls the reusable patch
  saved on disk, while `train.crop_size` controls the model-view crop. Added guidance for the
  common expert workflow of extracting 512x512 patches and training/validating on 256x256 crops.

- **Patch-mode validation** (`scripts/train.py`) — training now fails early with a clear error when
  `train.crop_size > patch.patch_size`, preventing invalid patch-mode runs before loader
  construction.

- **HPC submit summary** (`cli.py`, `nbs/14_cli.ipynb`) — `segpt-submit` now prints whether the
  submitted job is a normal training run or a build-only patch-cache job, making HPC dry-runs and
  job reviews easier to audit.

## [Unreleased] — 2026-06-14

### Added

- **`model.weights_path`** (`configs/train.yaml`, `scripts/train.py`) — set to a local `.pth` file
  to load pretrained encoder weights without an internet connection; `null` = download from internet.
  Generated by `download_encoder_weights('mobilenet_v2', '/path/to/mobilenet_v2.pth')`.

- **`data.data_mode: single_repeat_3`** (`scripts/train.py`, `configs/`) — new channel preset that
  reads one grayscale PNG and repeats it three times so pretrained RGB encoders (MobileNetV2,
  ResNet, EfficientNet) receive a valid 3-channel input without retraining the first conv.

- **`train.resume_from`** (`configs/train.yaml`, `train.py`) — explicit checkpoint path to resume
  from; `null` = resume from `latest.pth` (existing behaviour). Raises `FileNotFoundError`
  immediately if the path does not exist.

- **`model.in_channels`** (`configs/train.yaml`, `scripts/train.py`) — explicit channel count in
  config; validated against `data_mode` preset at startup to catch mismatches early.

- **`TMPDIR=/tmp` enforcement** (`hpc/submit.sh`, `scripts/train.py`) — `TMPDIR` is now set to
  `/tmp` (local disk) both in the job script (after `.bashrc` so it cannot be overridden) and at
  the very top of `scripts/train.py` (before any worker is spawned). Fixes `OSError: [Errno 16]
  Device or resource busy` from DataLoader workers on NFS home directories.

- **Patch copy to local node storage** (`hpc/submit.sh`) — the job script now reads `data.patch_dir`
  from the config and copies the patch cache to `mktemp -d /tmp/segpt_patches_XXXXXX` before
  training, then passes `data.patch_dir=<local_path>` as a Hydra override. Eliminates NFS read
  latency per batch and keeps GPU utilisation above the HPC 5 % watchdog threshold.

### Changed

- **`get_model()` shape-mismatch tolerance** (`model.py`, `nbs/09_model.ipynb`) — when loading a
  local `.pth` encoder, keys whose tensor shape does not match the current model are silently
  skipped instead of raising a `RuntimeError`. A clear warning is printed listing the skipped keys
  and advising to re-run `download_encoder_weights`. All other pretrained layers still load.

- **`train_one_epoch()`** (`train.py`, `nbs/10_train.ipynb`) — replaced `float(ld[k])` with
  `ld[k].detach().item()` to eliminate `UserWarning: Converting a tensor with requires_grad=True
  to a scalar` on every training step.

- **`train.num_workers`** (`configs/train.yaml`) — restored to `4` (was temporarily set to `0` as
  a workaround for the NFS/TMPDIR issue; now fixed at the root).

- **`hpc/README.md`** — expanded with a dedicated troubleshooting section covering both HPC issues
  encountered (NFS shared-memory OSErrors, GPU watchdog kills) and guidance for large patch datasets.

- **`logs/` untracked** (`.gitignore`) — `logs/` folder added to `.gitignore` and removed from
  git index (`git rm --cached`). HPC job log files (`.out`/`.err`) are kept on disk but will no
  longer be committed or pushed.

- **`tutorials/03_inference_and_top_losses.ipynb`** — new tutorial notebook covering:
  load a checkpoint, build val dataset from existing patch cache, visualise random val predictions
  (`show_batch`), find operating threshold (`find_operating_threshold`), plot top-loss patches
  (`show_top_losses`) for error analysis, and run full-image inference with probability heatmap
  and binary mask output.

---

## [Unreleased] — 2026-06-12

### Added

- **`predict()` `activate` parameter** (`inference.py`) — new `activate: bool = True` flag lets
  callers receive raw logits instead of sigmoid/softmax probabilities; useful for ensembling or
  custom post-processing

### Changed

- **Verbose per-epoch logging** (`train.py`) — each epoch now prints a single summary line:
  ```
  epoch 005/200  train=0.3421  val=0.2987  iou=0.7812  dice=0.8541  recall=0.9103  lr=1.00e-04  *best*
  ```
- **`inference.py`** — fixed import of `ObjectMetrics` to use absolute path
  (`from segmentation_tools_pt.metrics import ObjectMetrics`)
- **`pyproject.toml`** — added `aiop-tools>=0.0.32` dependency; moved `[project.scripts]`
  section after `[[tool.uv.index]]` to satisfy uv ordering requirement
- **`configs/train.yaml`** — reset `train.epochs` to `200`; corrected HPC `repo_dir` path
  to `/path/to/segmentation_tools_pt`

---

## [Unreleased] — 2026-06-03

### Added

#### Direct training mode (`data.patch_mode`)

New config key — choose whether to pre-build a patch cache or train straight from raw images:

| `data.patch_mode` | Behaviour |
|---|---|
| `true` (default) | Build / reuse patch cache in `patch_dir`; hard-example oversampling |
| `false` | Skip patch build; load full images directly from `raw_dir` |

```bash
# patch mode (default)
segpt-train configs/train.yaml

# direct mode — no patch build step
segpt-train configs/train.yaml data.patch_mode=false
```

- **`DirectDataset`** (`dataset.py`) — loads full-res images+masks with on-the-fly random/center crop
- **`split_sample_ids()`**, **`build_direct_datasets()`**, **`build_direct_dataloaders()`** — helpers for the direct path
- **`train()`** now accepts optional `train_loader` / `val_loader` (direct mode) or `root` (patch mode) — fully backward-compatible
- `stats.json` written to `raw_dir` in direct mode (not `patch_dir`)
- `docs/TRAINING_CLI.md` — new "Direct mode" section with when-to-use guidance

---

## [Unreleased] — 2026-06-02

### Added

#### Training pipeline
- **`scripts/train.py`** — full Hydra-driven end-to-end pipeline:
  discover samples → build patches → compute stats → train → optional ONNX export
- **`configs/train.yaml`** — canonical Hydra config with all parameter groups
  (`data`, `patch`, `model`, `train`, `loss`, `metrics`, `augment`, `export`)
- **`configs/train_example.yaml`** — copy-paste template with `???` placeholders
  for required paths; safe to commit alongside user projects
- **`src/segmentation_tools_pt/cli.py`** — `segpt_train()` console entry point;
  maps `segpt-train /path/to/config.yaml` → Hydra `--config-path / --config-name`

#### Documentation
- **`docs/TRAINING_CLI.md`** — full training CLI user guide covering:
  install, raw data layout, config section cheat sheet, conservative augment snippet,
  CLI overrides, output structure, offline HPC weights, troubleshooting table
- **`README.md`** — added "Training CLI" quick start block and updated project structure tree

#### Augmentations (`transforms.py`)
- `AugmentConfig` dataclass with all per-image fields (YOLO26-aligned)
- `YOLO26_PRESETS` dict (`yolo26n/s/m/l/x`, `legacy`) for one-line preset selection
- `get_train_transforms()` builds an Albumentations pipeline from `AugmentConfig`
- Per-image knobs: `translate`, `scale`, `degrees`, `shear`, `perspective`,
  `fliplr`, `flipud`, `hsv_h/s/v`, `bgr`, `erasing` (CoarseDropout), `extra_legacy`
- Batch-level YOLO26 augs: `mosaic`, `mixup`, `cutmix`, `copy_paste`, `close_mosaic`;
  `mixup` automatically skipped for multi-class integer masks

#### Model (`model.py`)
- `download_encoder_weights(arch, dest_path)` — download ImageNet weights on a
  machine with internet and save to disk for air-gapped HPC
- `get_model(weights_path=...)` — loads local `.pth` encoder weights

#### Dataset build (`dataset_build.py`)
- `resolve_mask_path()` — flexible single/multi-channel mask path resolution
- `resolve_channel_paths()` — multi-channel image path resolution
- `discover_samples()` — auto-discover matched image+mask pairs for any `data_mode`

#### Core utilities (`core.py`)
- `platform_info()` — OS / CUDA / PyTorch / OpenCV / Albumentations versions
- `binarize()` — normalises masks to `{0,1}` regardless of `{0,255}` or `{0,1}` input

#### Training loop (`train.py`)
- Configurable loss phases (`resolve_loss_weights`): Tversky/Dice/boundary weights
  change automatically at epoch boundaries
- Configurable `best_metric` (iou | dice | recall | precision)
- `train()` accepts full `AugmentConfig`-based batch augmentation kwargs

#### `pyproject.toml`
- Added `hydra-core>=1.3` dependency
- Registered `segpt-train = "segmentation_tools_pt.cli:segpt_train"` console script
- Fixed `license` to plain SPDX string; removed unsupported `script-files` glob

### Fixed

#### `masks.py` — OpenCV contour dtype bug (Linux)
- `fit_rotated_rects_mask()` produced empty masks on Linux/HPC.
  Root cause: `cv2.drawContours` silently does nothing when contours are `int64`
  (`np.intp` on 64-bit Linux). Fixed by casting with `_CV_CONTOUR_DTYPE = np.int32`.

#### `scripts/train.py` — `_export_best_model` syntax error
- Broken function header caused `SyntaxError` at import time.
  Restored correct `def _export_best_model(cfg, checkpoint_dir, in_channels):` signature.

---

## [0.1.0] — 2026-05-29 · commit `305546f`

*"feature: skeleton + tutorials are added, need to test on hpc"*

### Added

#### Library skeleton (all modules authored as nbdev notebooks)

| Module | Notebook | Description |
|--------|----------|-------------|
| `core` | `00_core.ipynb` | shared utilities |
| `channels` | `01_channels.ipynb` | multi-channel image loading / normalisation |
| `masks` | `02_masks.ipynb` | mask conversion, `fit_rotated_rects_mask`, `binarize` |
| `patches` | `03_patches.ipynb` | sliding-window patch extraction |
| `dataset_build` | `04_dataset_build.ipynb` | discover → extract → cache patch dataset |
| `transforms` | `05_transforms.ipynb` | paired Albumentations augmentation pipeline |
| `dataset` | `06_dataset.ipynb` | `SegmentationDataset` (PyTorch `Dataset`) |
| `losses` | `07_losses.ipynb` | `TverskyLoss`, `DiceLoss`, `BoundaryLoss`, composite |
| `metrics` | `08_metrics.ipynb` | pixel IoU, Dice, recall, precision |
| `model` | `09_model.ipynb` | `get_model()` smp factory (any architecture + encoder) |
| `train` | `10_train.ipynb` | `train()` training loop with checkpointing |
| `inference` | `11_inference.ipynb` | `predict()` with TTA support |
| `visualize` | `12_visualize.ipynb` | overlay / comparison plots |
| `export` | `13_export.ipynb` | ONNX export + verification |

#### Tutorials
- `tutorials/01_training_walkthrough.ipynb` — end-to-end walk-through for all
  four `data_mode` variants: `single`, `single_to_3`, `two_channel`, `three_channel`
- `tutorials/02_next_steps.ipynb` — extension points and production checklist

#### Project infrastructure
- `pyproject.toml` — src layout, nbdev config, dependencies
- `.cursor/rules/cv-expert-panel.mdc` — expert-panel prompting rule
- `.cursor/rules/nbdev-authoring.mdc` — nbdev source-of-truth rule
- GitHub Actions: `deploy.yaml` (docs), `test.yaml` (nbdev tests)
- `uv.lock` — locked dependency tree

---

## [0.0.1] — 2026-05-29 · commit `d4caa48`

*"Initial commit"*

- `README.md` stub — project description and goal statement
