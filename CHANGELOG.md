# Changelog

All notable changes to `segmentation_tools_pt` are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

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
