# Multi-head robustness: encoders, augs, postprocess, deploy

**Date:** 2026-10-01  
**Wayfinder map:** [Multi-head robustness: encoders, augs, postprocess, deploy](https://github.com/HasanGoni/segmentation_tools_pt/issues/1)  
**Spec issue:** [Spec: Multi-head robustness](https://github.com/HasanGoni/segmentation_tools_pt/issues/14)  
**Glossary:** `CONTEXT.md`  
**Implementation contract:** `docs/agents/nbdev-implementation-contract.md`

---

## Problem Statement

Users training multilabel industrial segmentation (and general segmentation tasks) need to:

1. Switch curated encoders (DINOv3, ConvNeXt, EfficientNetV2) and shut them off back to the library default without editing Python.
2. Optionally enable layout-diversity batch augmentations (MixUp, CutMix, mosaic, copy_paste) that today are config-only placeholders.
3. Optionally clean one named class channel at inference with classical morphology (opening + min-area), without baking that into loss or raw train metrics.
4. Run post-training error analysis and FP32 ONNX + batched tile inference via config.

Today the library default stays MobileNet; DINOv3 exists in code but presets/docs/`get_model` allowlisting for `tu-*` are incomplete; batch augs are not wired; class-targeted morph is not config-gated; error analysis is notebook-only; production depth stops short of documenting the FP32 batch path as the destination.

## Solution

One design covering the cleared map decisions: Hydra **encoder preset packages** + curated/`tu-*` allowlist; wire **batch-level augmentations** behind a master switch (YOLO-style pick-one); **class-targeted morph postprocess** at inference only; **ErrorAnalysisCallback** on train end; **freeze forever** via `freeze_epochs: -1`; document/config **FP32 ONNX + batched tile inference**. Product recipes (e.g. multilabel Chip path) opt into primary recommendation DINOv3; library `train.yaml` stays MobileNet. Deferred work (LoRA, INT8, TensorRT, multi-target morph, …) is recorded explicitly for later — not silently dropped.

## User Stories

1. As a library user, I want `configs/train.yaml` to keep MobileNet as the library default, so existing runs stay unchanged until I opt in.
2. As a product engineer on a Chip/Void/Lifted recipe, I want a documented primary recommendation of DINOv3 ViT-S, so I know the starting encoder under domain shift.
3. As a product engineer, I want a documented throughput alternative of ConvNeXt-Tiny, so I can trade robustness for speed/memory.
4. As a Hydra user, I want `configs/model/` packages for mobilenet_v2, dinov3_vit_small, convnext_tiny, and efficientnetv2_s, so I switch encoders by composing `defaults` or CLI `model=…`.
5. As a Hydra user, I want shut-off to mean selecting the MobileNet package, so I do not need a special `none` flag.
6. As a Hydra user, I want each encoder package to set `model.*` only, so freeze/LR stay under my train/recipe config.
7. As a multilabel recipe user, I want `train_multilabel_3head.yaml` to compose DINOv3-S by default, so the product path matches the primary recommendation.
8. As a trainer, I want `model.arch: convnext_tiny` (alias) and raw `tu-convnext_tiny` to both work through `get_model` / `segpt-train`, so presets are first-class.
9. As a trainer, I want `efficientnetv2_s` / `tu-tf_efficientnetv2_s` curated the same way, so I can try that curated encoder without raw SMP hacks.
10. As an experimenter, I want other `tu-*` names to warn-and-proceed, so agents can try experimental timm encoders without a hard block.
11. As a discoverability user, I want `list_smp_encoders()` to advertise curated names and aliases (not the whole of timm), so skills and humans see the supported set.
12. As an agent author, I want `choose-model-architecture` updated in the same implement slice as the allowlist, so agents emit curated vs experimental correctly.
13. As a trainer, I want pretrained init + freeze warm-up as the default adaptation path, so encoders start from strong weights.
14. As a trainer, I want `freeze_epochs: -1` to keep the encoder frozen for the whole run, so I can train decoder-only when needed.
15. As a trainer, I want `freeze_epochs: 0` to mean no freeze and `N > 0` to mean warm-up then unfreeze, so the schedule is predictable.
16. As a trainer, I want batch augs behind `augment.batch.enabled` default false, so nothing silent changes today’s training.
17. As a trainer, I want mixup/cutmix/mosaic/copy_paste probabilities defaulting to 0, so enabling the master switch alone does not surprise me until I set probs.
18. As a trainer, I want YOLO-style pick-one (at most one batch aug type per batch), so stacking MixUp then CutMix in one step does not happen.
19. As an X-ray recipe author, I want flips to stay 0 by default and documented off for orientation-sensitive data, so geometry stays physical.
20. As a general CV user, I want to enable flips via config when orientation does not matter, so the library is not X-ray-only.
21. As an inference user, I want `postprocess.class_morph.enabled` default false, so decode stays raw until I opt in.
22. As an inference user, I want to name any class via `class_name` resolved against `data.class_names`, so the library is not Void-branded.
23. As a Chip recipe author, I want to set `class_name: void` in project YAML, so speck FP cleanup targets Void without changing library vocabulary.
24. As an inference user, I want opening + min-area on that channel only, so other heads are untouched.
25. As an inference user, I want morph then hierarchy in a fixed order, so decode is reproducible.
26. As a trainer, I want morph never applied to loss or to metrics that judge the raw model, so dashboards reflect true model behavior.
27. As a trainer, I want `ErrorAnalysisCallback` on `on_train_end` via `monitoring.error_analysis`, default disabled, so I can inspect worst val cases after a run.
28. As a multilabel trainer, I want per-class top-k when `per_class: true`, so Void FPs and Lifted misses can be ranked separately.
29. As a deployer, I want FP32 ONNX export documented and working through `get_model` for curated encoders, so I can ship without INT8 yet.
30. As a deployer, I want batched tile inference and DataLoader prefetch knobs documented/configured, so GPU util stays high on large images.
31. As a maintainer, I want all code changes via nbdev notebooks + export, so `src/` is never hand-edited.
32. As a future planner, I want LoRA, INT8, TensorRT, multi-target morph, active-learning feedback, and ieg-synth-in-library explicitly listed as later/out-of-scope, so I can reopen them without rediscovering the debate.

## Implementation Decisions

### Encoder presets and allowlist

- Hydra config group `configs/model/` with packages: `mobilenet_v2`, `dinov3_vit_small`, `convnext_tiny`, `efficientnetv2_s`.
- `train.yaml` defaults compose `model: mobilenet_v2`; multilabel recipe composes `model: dinov3_vit_small`.
- Packages set model fields only (architecture, arch, pretrained defaults / weights_path guidance); not train freeze/LR.
- `get_model` accepts curated `tu-*` and aliases (`convnext_tiny` → `tu-convnext_tiny`, `efficientnetv2_s` → `tu-tf_efficientnetv2_s`); other `tu-*` warn once; unknown non-`tu-*` hard-error.
- `list_smp_encoders` advertises curated + aliases (+ existing SMP registry and `dinov3_*`).
- Same implement slice updates `choose-model-architecture` (and export skill if needed).

### Training adaptation

- Pretrained + freeze warm-up remains the default documented path.
- `freeze_epochs: -1` = freeze forever; implement in the train/freeze helpers if missing.

### Batch-level augmentations

- Wire MixUp, CutMix, mosaic, copy_paste behind `augment.batch.enabled` (default false).
- Per-type probs default 0; YOLO-style pick-one when enabled.
- Sample-level geometric/photometric stack unchanged; flips default 0.

### Class-targeted morph postprocess

- Config block `postprocess.class_morph.*` with master switch default false.
- Resolve `class_name` → index via `class_names`; fail if missing.
- Ops: morphological opening + connected-component min-area on that channel only.
- Pipeline order after threshold decode: class morph → then hierarchy.
- Inference/eval/production decode only — not loss, not raw train metrics.

### Error analysis

- `ErrorAnalysisCallback` on `on_train_end` via `build_default_callbacks` and `monitoring.error_analysis` (default enabled false).
- Outputs under checkpoint `error_analysis/` (overlays + JSON); multilabel per-class top-k when configured.
- Threshold sweep deferred (out of this map’s implement scope).

### Production depth (this map)

- FP32 ONNX + batched tile inference + DataLoader prefetch: document and ensure config/docs surface; build on existing export/inference APIs.
- No INT8 or TensorRT implementation in this spec’s delivery.

### Process

- Implement via `/to-tickets` vertical slices; first slice = encoder allowlist + preset packages + skill sync (per map destination).
- Literate nbdev only; never hand-edit generated package sources.
- Custom UNet / `model.source` factory is **map 12**, not this spec.

## Testing Decisions

Good tests assert **external behavior** on synthetic tensors/configs (shapes, warnings, enable/disable, artifacts on disk), not private helpers.

Agreed seams:

1. `get_model` / `list_smp_encoders` — curated aliases build; experimental `tu-*` warns; unknown non-`tu-*` errors.
2. Hydra model packages — compose yields expected `model.*`; library default package remains MobileNet.
3. `train_one_epoch` + batch augs — pick-one when enabled; master off / probs 0 matches prior behavior.
4. `freeze_epochs: -1` and `build_default_callbacks(error_analysis=…)` — encoder stays frozen; callback writes top-k under a temp checkpoint dir.
5. Inference decode chain — threshold → class morph (named class) → hierarchy; morph off is no-op; bad `class_name` fails clearly.
6. `export_onnx` smoke for curated builds via `get_model` (fixed spatial size; dynamic batch where already supported).

Prior art: in-notebook tests beside exports in model, train, transforms, inference, export, and hydra_pipeline notebooks; run via `nbdev-test` on the owning notebook path.

## Out of Scope

### Explicitly out of this effort (will not graduate from this map)

- TensorRT packaging / GPU-locked engines (deployment-env; near serving).
- Full inference microservice / FastAPI serving.
- Rewriting the multilabel Chip/Void/Lifted head API (already present).
- Replacing or embedding ieg-synth inside this library.
- Custom UNet / `model.source` factory (owned by map [Custom UNet + SMP factory](https://github.com/HasanGoni/segmentation_tools_pt/issues/12)).

### Deferred — document for later (fog / future map; do not implement now)

Recorded so they are easy to reopen after this spec ships:

| Item | Status | When to reopen |
|------|--------|----------------|
| LoRA on DINOv3 (or other encoders) | Fog on map 1 | Only if freeze warm-up + low encoder LR underfits on real runs |
| INT8 calibration policy and quantized export | Fog on map 1 | After FP32 batched inference is validated and still misses latency |
| TensorRT | Out of scope (above) | Separate deployment map if needed |
| Multi-target `class_morph` list (many class names) | Fog | When one named class is not enough |
| Active-learning hard-case feedback into training | Fog | When an AL loop exists |
| Synthetic layout placement from ieg-synth inside this library | Fog | If external synth is insufficient |
| Broader “every notebook change must sync every related skill” process rule | Fog | Beyond the encoder-factory skill sync locked for Allowlist |
| Error-analysis threshold sweep | Deferred from error-analysis decision | Optional later monitoring slice |

These rows must stay visible on the wayfinder map’s **Not yet specified** / **Out of scope** sections and in this spec so later sessions do not treat them as forgotten.

## Further Notes

- Parent map destination: hand off to `/to-tickets` with **encoder presets as the first implementation slice**.
- Vocabulary: Library default, Primary recommendation, Throughput alternative, Curated encoder, Experimental tu-* encoder, Encoder preset package, Batch-level augmentation, YOLO-style pick-one, Freeze warm-up, Freeze forever, Class-targeted morph postprocess — see `CONTEXT.md`.
- Research fact: FP32 ONNX already verified for DINOv3 via `get_model` and for ConvNeXt/EffV2 via raw SMP before allowlist; post-allowlist path should use `get_model` for curated names.
- Sibling map 12 owns custom UNet; sequence implementation so both do not thrash the model notebook in parallel without coordination.
