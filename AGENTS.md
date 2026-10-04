# AGENTS.md — Repository Overview

> Read this first. One screen of orientation + links. **This file is intentionally lean —
> when detail is needed, put it in a linked doc (`docs/agents/repo-guide.md` is the default
> home), never inline it here.** Deep guide: `docs/agents/repo-guide.md`.

---

## 1. What this repo is

**`segmentation_tools_pt`** is a PyTorch library for **industrial image segmentation**.

It was built to solve a recurring problem: every new segmentation project
(pin detection, wafer maps, PCB inspection, …) required re-writing the same boilerplate —
mask loading, patch extraction, paired augmentations, IoU metrics, training loops, ONNX export.
This library collects all of that into one clean, tested, reusable package so each new project
starts from a solid foundation, not from copy-pasted scripts.

**Domain:**
- Input images are typically grayscale or multi-channel industrial X-ray / optical inspection images
- Ground-truth masks are binary (defect / no-defect) or multi-class label maps
- Models are U-Net style encoder–decoder networks (via `segmentation-models-pytorch`)
- Inference runs on HPC GPU nodes (LSF cluster) or locally

---

## 2. Development methodology — nbdev

This project uses **[nbdev](https://nbdev.fast.ai/)**. The golden rule:

```
Edit notebooks in nbs/ → nbdev_export → src/ is auto-generated
```

**Never edit files inside `src/segmentation_tools_pt/` directly.**
The notebooks in `nbs/` are the source of truth; the `.py` files in `src/` are generated artifacts.
The notebook IS the documentation — every function ships with its docstring, example, and test.

```bash
nbdev-export    # regenerate src/ from nbs/
nbdev-test      # run all in-notebook tests
nbdev-preview   # local docs preview (Quarto)
```

On NFS mounts, verify agent edits landed on disk before export — see the `nbdev-notebook-verify` skill.
Implementation contract: `docs/agents/nbdev-implementation-contract.md`.

---

## 3. Repository map

```
segmentation_tools_pt/
├── nbs/          ← SOURCE OF TRUTH (edit here); 00_core … 15_hydra_pipeline notebooks → src/
├── src/segmentation_tools_pt/  ← AUTO-GENERATED (do not edit)
├── scripts/      # build.py (segpt-build), train.py (segpt-train) — Hydra entry points
├── configs/      # data_patch.yaml (shared base), build.yaml, train.yaml, *_example.yaml, model/ presets
├── docs/         # TRAINING_CLI.md, agents/ (issue-tracker, domain, repo-guide), adr/, superpowers/specs/
├── hpc/          # submit.sh (bsub job script) + README (HPC guide, large-dataset strategies)
├── tutorials/    # 01 walkthrough, 02 next steps, 03 inference+top-losses, 03 testing skills,
│                 # 04 augmentation playground, 05 activation semantics
├── outputs/      # Hydra run outputs (gitignored)
├── CONTEXT.md    # domain vocabulary (single source of truth for shared language)
└── AGENTS.md     # ← this file
```

Notebook → module mapping: `00_core`, `01_channels`, `02_masks`, `03_patches`,
`04_dataset_build`, `05_transforms`, `06_dataset`, `07_losses`, `08_metrics`, `09_model`,
`10_train`, `11_inference`, `12_visualize`, `13_export`, `14_cli`, `15_hydra_pipeline` —
each notebook generates the correspondingly named module in `src/`.

---

## 4. Module map (one line each — deep guide in `docs/agents/repo-guide.md`)

| Module | Responsibility |
|---|---|
| `channels.py` | assemble source channels from image files (`data_mode`: single, single_repeat_3, single_to_3, two_channel, three_channel) |
| `masks.py` | mask binarization; rotated-rect fitting (`fit_rotated_rects_mask`) |
| `patches.py` | sliding-window patch extraction with empty-patch filtering |
| `dataset_build.py` | discover → (fit rects) → patch → split → `metadata.json`/`stats.json` |
| `transforms.py` | sample-level augmentations (preset knobs are config-only placeholders); batch augs live in `train.py` |
| `dataset.py` | `SegmentationDataset` (patch mode), `DirectDataset` (direct mode), loaders |
| `losses.py` | Tversky+Dice+boundary phase-scheduled loss; newer stack (`TverskyFocalLoss`, `ForeGroundAwareLoss`, `SegLoss`, …) |
| `metrics.py` | pixel IoU/Dice/recall/precision + object-level `ObjectMetrics`; per-epoch tracking |
| `model.py` | `get_model()` factory (SMP + custom UNet, curated + DINOv3 encoders); air-gapped weight download |
| `train.py` | resumable AMP training loop; freeze epochs; batch augs; `TrainCallback` hooks; monitoring callbacks |
| `inference.py` | `predict` (sliding window), `predict_tiled` (batched tiles), decode/morph/hierarchy, threshold sweep |
| `visualize.py` | overlay masks, top-loss grids, error posters |
| `export.py` | ONNX export + verification, task/class_names metadata |
| `cli.py` | `segpt-build` / `segpt-train` / `segpt-submit` entry points |
| `hydra_pipeline.py` | shared Hydra resolvers (batch augs, mask, inference, class-morph configs) |

**Gotchas and per-module detail** (dataset dict-return, `channel_names` mismatch, patch-cache
naming, callback hooks table, monitoring outputs): `docs/agents/repo-guide.md`.

---

## 5. The two training modes

| Mode | When | Flow |
|---|---|---|
| Patch mode (`data.patch_mode: true`, default) | Large images, reusable cache, hard-example oversampling | `raw_dir` → extract patches → `patch_dir/{train,val}` → loaders |
| Direct mode (`data.patch_mode: false`) | Small images, fast prototyping | `raw_dir` → random/center crop on-the-fly |

Data layout and the end-to-end flow diagram: `docs/agents/repo-guide.md`.

---

## 6. Config files

Two Hydra configs share a common base (`configs/data_patch.yaml`): `configs/build.yaml`
(`segpt-build`, CPU-friendly) and `configs/train.yaml` (`segpt-train`, full training config).
`configs/*_example.yaml` are annotated templates for new projects; encoder presets live in
`configs/model/`. **The YAML files are the source of truth** — the annotated walkthrough of
every block lives in `docs/agents/repo-guide.md`, and daily driving is covered by
`docs/TRAINING_CLI.md`. Any key can be overridden on the command line:
`segpt-train configs/train.yaml train.epochs=80 metrics.best_metric=recall`.

---

## 7. CLI commands

| Command | What it does |
|---|---|
| `segpt-build` | Hydra-driven patch-cache preparation (`configs/build.yaml`) |
| `segpt-train` | Hydra-driven end-to-end training (`configs/train.yaml`) |
| `segpt-submit` | Submits a build/training job to LSF HPC via `bsub`; reads `hpc:` from config; `--dry-run` to verify |

Examples:
```bash
segpt-build configs/build.yaml data.force_rebuild=true patch.patch_size=512
segpt-train configs/train.yaml train.epochs=80 data.patch_mode=false
segpt-submit configs/train.yaml hpc.queue=gpu_compute hpc.host=gpu-node-01
```

---

## 8. Installation and environment

This project uses **[uv](https://docs.astral.sh/uv/)**:

```bash
uv sync                    # first setup: creates .venv/ + installs CLI commands
source .venv/bin/activate
uv lock && uv sync         # after changing pyproject.toml
```

On HPC the job script runs `uv sync --quiet` automatically before starting training.

---

## 9. Coding conventions

1. **Notebook first** — explore in a notebook cell, confirm it works, then wrap in a function.
   One function = one clear responsibility.
2. **Test in the same notebook** — every exported function has a test cell directly below it
   using real or synthetic data. Tests are cheap; they run via `nbdev_test`.
3. **Docstring format** — fastai/nbdev style (one-line summary, no "Returns:" header).
4. **No comments that narrate code** — comments explain *why*, not *what*.
5. **Progressive complexity** — simple implementation first; add complexity only when needed.
6. **`__all__`** — every exported module defines `__all__` explicitly at the top.

---

## 10. Key design decisions and trade-offs

| Decision | Rationale |
|---|---|
| nbdev over plain Python | Exploration + tests + docs in one place; critical for industrial code that must be maintainable long-term |
| Hydra for config | Hierarchical overrides, run reproducibility (`.hydra/` saves exact config), composable configs |
| Phase-scheduled loss | Coarse Dice early → boundary sharpening late; works better than a static loss for irregular industrial defects |
| YOLO26 augmentation presets | Consistent baseline; easy to tighten (set flips=0 for fixed-camera X-ray) or loosen |
| Two training modes | Patch cache for large images + hard-example oversampling; direct mode for fast prototyping |
| `segpt-submit` in Python | Shell scripts break; Python reads the same YAML config the trainer uses → single source of truth |
| uv over conda | Faster, reproducible lockfile, no channel conflicts, works in air-gapped environments with Artifactory mirror |
| `TMPDIR=/tmp` in job script and Python | HPC `.bashrc` sets TMPDIR to NFS; NFS doesn't support POSIX shared memory → DataLoader workers crash. Setting it after `.bashrc` and again in Python ensures local disk is always used |
| Patch copy to `/tmp` at job start | NFS patch reads cause long GPU idle gaps and trigger HPC watchdog (< 5 % GPU util for 5 min). Copying once to local SSD makes reads ~10× faster and keeps GPU utilisation high |

---

## 11. Common workflows

Quick recipes; full versions in `docs/agents/repo-guide.md`, CLI details in `docs/TRAINING_CLI.md`,
HPC in `hpc/README.md`.

- **Before starting any change**: run `git status --short` + `git diff --stat` and check the top of
  `CHANGELOG.md` — a fix from a prior session often exists uncommitted; treat that as the start.
- **New project**: copy `configs/*_example.yaml`, edit paths, `segpt-build` then `segpt-train`.
- **New feature**: edit `nbs/` → run cells → `nbdev_export` → `nbdev_test` → commit `nbs/` + `src/`.
- **HPC run**: `segpt-submit configs/train.yaml --dry-run` then without `--dry-run`.
- **Resume**: `train.resume: true` (+ `train.resume_from:` for a specific checkpoint).
- **Large patch sets**: scratch-prestage / NFS-workers / direct-mode strategies — `docs/agents/repo-guide.md` §4.

---

## Agent skills

### Issue tracker

GitHub Issues are the official board (`gh` CLI); `.scratch/` is for private drafts only. See `docs/agents/issue-tracker.md`.

### Triage labels

Default roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.

### nbdev implementation

All feature work: literate `nbs/` → test steps → `nbdev-export` (never hand-edit `src/`). See `docs/agents/nbdev-implementation-contract.md`.

---

## Memory maintenance policy

The `Learned` sections below are maintained by the continual-learning hook. Rules:
- **AGENTS.md stays lean** — add a one-line fact, not prose; detail goes in the linked doc
  (`docs/agents/repo-guide.md`) with a pointer added here if needed.
- Never restructure or re-title the `Learned` sections (the updater and other agents rely on them).
- Exclude one-off/transient details and secrets.

## Learned User Preferences

- Treat notebooks as literate programs (Knuth/nbdev): narrative + bullet plan first; then **experiment → verify** each step on synthetic data; only after every verify passes promote to `#| export` (dump-and-export is not literate). Tests sit as footnotes under the export. `/implement` (and TDD/code-review) still apply, but never skip this experiment→verify→function order — see `docs/agents/nbdev-implementation-contract.md`.
- Prefer plain language over jargon; define unfamiliar terms briefly when first used (e.g. avoid unexplained words like “canonical”).
- Prefer a wayfinder decision map before writing a design spec; ship phased tickets (umbrella/design first, then one implementable slice); finish one map through implement before starting the next; document deferred/out-of-scope items in the map or spec for later; create map/ticket/spec via the repo’s wayfinder flow rather than waiting indefinitely on grill answers.
- Prefer reviewing draft agent-config files before they are written to disk.
- Prefer GitHub Issues as the official tracker (`gh`); keep `.scratch/` for private drafts only.
- Prefer the default Matt Pocock triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
- On wayfinder grill questions (A/B/C), default to the agent’s expert recommendation for every question unless they explicitly override.
- Include post-training error analysis in wayfinder/planning (grill-with-docs or related tickets), not only training and export.
- Prefer config-only control so any segmentation task works by editing Hydra YAML without code changes (including class/head names and optional inference post-processing on/off).
- Prefer independently toggleable batch-level augmentations in config (mixup, cutmix, mosaic, copy_paste, alone or combined).
- Prefer default ImageNet encoder weights with freeze warmup, plus a config path to keep the encoder frozen for the full run.
- When changing library behavior in notebooks, also update related agent skills so non-power-users can ask the agent and get working scripts, and reference the feature from the GitHub issue, user docs (e.g. `docs/TRAINING_CLI.md`), and an interactive tutorial notebook when hands-on testing applies (e.g. augmentation playground in `tutorials/`).
- Close implementation issues explicitly (`gh issue close` + wrap-up comment mapping deliverables to commits) as soon as work is done, even while commits sit on an unpushed feature branch — `Closes #N` keywords only fire on merge to the default branch. Split commits into logical per-issue chunks (e.g. `73c7577` / `93f48a4`) so issue closure stays auditable.

## Learned Workspace Facts

- DINOv3 ViT-S/16 and B/16 are available via `get_model()` as `dinov3_vit_small` / `dinov3_vit_base`; curated aliases `convnext_tiny` / `efficientnetv2_s` (and raw `tu-convnext_tiny` / `tu-tf_efficientnetv2_s`) are first-class in `get_model` / `list_smp_encoders`; Hydra packages live under `configs/model/`.
- Mask config is resolved with `resolve_mask_cfg(cfg)` → `{mask_mode, mask_channels, class_names}`; `validate_mask_mode(cfg)` validates companions and returns `(mask_mode, mask_channels, class_names)`.
- This is a general segmentation library: class/head names are config-driven; Chip/Void/LiftedChip is an example multi-head use case, not hard-coded product vocabulary. Active focus also includes MatriX↔SAKI domain shift, post-training error analysis, and a config-selectable SMP vs custom UNet factory (ASPP/attention knobs).
- GitHub remote for this repo is `HasanGoni/segmentation_tools_pt`.
- FP32 ONNX export works for `dinov3_vit_small` (and raw `tu-*` Unets) at `in_channels=1`, multilabel `num_classes=3`; `onnx`/`onnxruntime` are optional/commented in `pyproject.toml`; INT8/TensorRT not validated.
- Shared nbdev implementation contract lives at `docs/agents/nbdev-implementation-contract.md`; wayfinder map for multi-head encoder/robustness is tracked on GitHub Issues.
- Design specs live in `docs/superpowers/specs/`: custom UNet/SMP factory design (`2026-10-01-custom-unet-smp-factory-design.md`; `model.source: smp | custom_unet`; ASPP/attention are v1 options, default off) and multi-head robustness design (`2026-10-01-multi-head-robustness-design.md`).
- DINOv3 pretrained weights load via timm's hub inside `Dinov3ReassembleEncoder.__init__` (`pretrained=True` threaded through the SMP encoder registry); `encoder_weights=None` only switches off smp's state-dict loader, which would fail because the DINOv3 checkpoint has no keys for the added reassemble layers.
- `ErrorAnalysisCallback` was decided in closed issue #9: fires on `on_train_end`, config under `monitoring.error_analysis`, default off, val top-k JPEG/JSON; implemented and closed via issue #18 (fuller poster spec folded into it: 4-column image | GT overlay α≈0.1 | pred overlay | diff overlay; train+val; rank by loss or metric; configurable top-k).
- `losses.py` exports a loss stack newer than the Tversky+Dice+boundary combiner: `TverskyFocalLoss`, `ForeGroundAwareLoss`, `hierarchy_regularizer_loss`, `SegLoss`, `resolve_loss_weights`.
- Multi-head robustness map #1 progress: #15 (DINOv3 encoders), #18 (`ErrorAnalysisCallback`), #16 (`freeze_epochs: -1` freeze-forever, commit `793dfbb`), #17 (batch-level augs: YOLO-style pick-one under `augment.batch.*`, default off; augmentation playground tutorial made canonical and referenced from the issue + `docs/TRAINING_CLI.md`), #21 (`EncoderDriftCheck` at train end: bit-identical digest verdict for frozen runs, per-layer cosine drift report otherwise, raises on drift when `expect_frozen=true`), #19 (class-targeted morph postprocess: `inference.apply_class_morph` + `hydra_pipeline.resolve_class_morph_cfg` + `postprocess.class_morph` config, default off), and #20 (FP32 ONNX for curated encoders via `get_model` + `export_onnx` regression smoke; batched tile inference `predict_tiled`/`tile_starts` with `inference.inference_batch_size` — batch-size invariant output; `prefetch_factor` loader knob in `train()`/`build_*_dataloaders` + `hydra_pipeline.resolve_inference_cfg`) are implemented and closed (#20 closed via commits `73c7577` run-inference skill + `93f48a4` skill-library fixes) — map #1 is fully retired. Active map is #23 ("Deep seams: tiled inference + ONNX export consolidation", labeled `wayfinder:map`): #24 (spec), #25 (`_activate` seam in `nbs/11_inference.ipynb`, implemented + closed via commits `6ad939f`/`afbe820` + `tutorials/05_activation_semantics.ipynb`), #26 (single ONNX export seam), #27 (consolidate tiled inference on `predict_tiled`, retire Oct-1 cluster); #26/#27 blocked by #24; #27 recommended next; active work branch remains `feat/onnx-batched-tile-inference` (commits local, not yet pushed); `configs/train_multilabel_3head.yaml` is the DINOv3 3-head recipe with `encoder_drift_check` enabled.
- All 41 skills in `.github/skills/` were audited (41/41 PASS); `.agents/skills` is a symlink to `.github/skills`; the smoke-test harness `.github/skills/skill-smoke-test/notebooks/test_skills.ipynb` was recreated and is paired with `tutorials/03_testing_skills.ipynb`; dispatch (skill loading) can only be verified in a fresh chat, not by static checks.
