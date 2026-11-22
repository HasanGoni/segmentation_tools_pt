# Research: where does RF-DETR segmentation fit in segmentation_tools_pt?

**Date:** 2026-11-22
**Issue:** [#99](https://github.com/HasanGoni/segmentation_tools_pt/issues/99) (`wayfinder:research`)
**Status:** findings and recommendation only. No code in `src/`, `nbs/`, `pyproject.toml`, `uv.lock` or the repo `.venv` was touched.
**Glossary:** `CONTEXT.md`

How to read the citations:

- `[wheel]` = file inside the installed PyPI release **rfdetr 1.12.1** (path under `rfdetr/`, line number from that install).
- `[main]` = file in a shallow clone of `github.com/roboflow/rf-detr` at commit `1268d0a` (2026-11-21, version `1.13.0.dev`, not released).
- `[docs]` = the markdown docs inside that clone (`docs/...`), which are what rfdetr.roboflow.com is built from.
- `[repo]` = this repository, `src/segmentation_tools_pt/...`.
- `[run]` = something I ran myself (Appendix A).
- **UNVERIFIED** = I could not confirm it from a primary source or a run.

---

## 1. Summary and recommendation

**What RF-DETR seg is.** An *instance* model: it outputs up to 100 objects per image, each with a box, a class, a confidence score and a mask. It is not a per-pixel classifier. Our library is the other kind (*semantic*): the model returns a per-pixel score map `(B, C, H, W)` and everything else (loss, metrics, tiling, ONNX, reports) is built on that map.

**Recommendation: C now, with two small bridge tools; move to B only after a real-data test shows RF-DETR is worth it. Do not do A.**

1. **A (full peer in `get_model()` + `train()`): no.** The `get_model` contract, `train_one_epoch`, `SegLoss`, `SegMetrics`, `predict_tiled`, batch augs and `export_onnx` all assume a `(B, C, H, W)` score map. RF-DETR has no such tensor in its training path, and its training loop lives inside PyTorch Lightning (`RFDETR.train()` builds a Lightning `Trainer`). A "peer" would mean forking or bypassing nearly every seam.
2. **B (separate `segpt-train-instance`-style entry sharing build/report/metrics): possible, not yet justified.** Cost is real (new data export, new callbacks, new report adapter) and the benefit on *our* defect domain is unmeasured. It becomes worth it only if RF-DETR beats our U-Net/DINOv3 on real patches.
3. **C (documented external recipe): do this now.** The smoke test shows the whole path works with one `pip install` and a COCO folder, including **1-channel grayscale input** (Section 3.2).
4. **Add two small pure functions, no new entry point:** (a) our mask PNG to COCO-JSON converter, (b) RF-DETR `Detections` to our `(H, W)` / `(K, H, W)` semantic masks. They make the A/B comparison possible, are testable without a GPU, and are needed by B anyway.
5. **Keep `rfdetr` out of `pyproject.toml`.** Install it in a separate environment (reasons in Section 7).

**Why not simply "yes, add it"?** Three hard facts from my runs: it is an instance model (thin/touching defect shapes are not what it is tuned for), input resolution is fixed per size (312 to 768 px, masks at 1/4 of that), and the exported ONNX is a different signature (`dets`/`labels`/`masks`, batch 1, fixed size) from our `image` -> `logits` convention.

**What I could not settle:** whether RF-DETR is *better* than our models on our data. No real data was used (Section 9).

---

## 2. Findings: training API (issue point 1)

| Question | Answer | Source |
|---|---|---|
| Seg model classes | `RFDETRSegNano/Small/Medium/Large/XLarge/2XLarge` (+ deprecated `RFDETRSegPreview`) | [wheel] `detr.py`/`variants.py`; [main] `src/rfdetr/variants.py:150-232` |
| Native resolution | 312 / 384 / 432 / 504 / 624 / 768 px square | [main] `README.md:115-120` |
| Resolution rule | Must be divisible by `patch_size * num_windows`: multiples of 24 (Nano: 12) | [docs] `docs/learn/train/training-parameters.md:103,110`; [main] `variants.py:150-156` |
| How to train | `model.train(dataset_dir=..., epochs=..., batch_size=..., ...)`, all kwargs go to a pydantic `TrainConfig` (seg adds mask loss coefficients) | [wheel] `detr.py:1285`; [wheel] `config.py:1880-1900` |
| Lightning | **Yes, hard dependency of `train()`**: it imports `RFDETRDataModule, RFDETRModelModule, build_trainer` and calls `trainer.fit(...)`. Without `rfdetr[train]` it raises ImportError. | [main] `detr.py:1325-1390`; [wheel] `training/trainer.py:17-19`; [run] (Lightning banner and `Trainer.fit` in the traceback/log) |
| Checkpoints | `last.ckpt`, `checkpoint_best_total.pth`, `checkpoint_best_ema.pth`, `last_ema.pth`, `checkpoint_<n>.ckpt`; `checkpoint_interval` default 10 | [docs] `training-parameters.md:157-171`; [run] output dir listing |
| Resume | `resume=<path>`; full `.ckpt` restores optimizer + scheduler, lightweight `.pth` restarts them | [docs] `training-parameters.md:42` |
| EMA | `use_ema=True` default, `ema_decay=0.993`; validation runs on EMA weights by default | [docs] `training-parameters.md:137-147` |
| Early stopping | `early_stopping`, `_patience=10`, `_min_delta=0.001`, `_use_ema` | [docs] `training-parameters.md:179-184` |
| Best-model rule | Mask mAP for seg (`best_model_metric="map"`) | [docs] `training-parameters.md:173` |
| Augmentation | Built in: torchvision default (resize/crop jitter + horizontal flip), optional Albumentations/Kornia via `aug_config` and `rfdetr[augment]`; includes `AUG_INDUSTRIAL` preset | [docs] `docs/learn/train/augmentations.md:1-45` |
| Hydra-drivable? | Yes in the narrow sense: `train(**kwargs)` is one function call, so a Hydra block can map onto kwargs. Not through our `train()`. | [wheel] `detr.py:1285` |

**Can training run without Lightning leaking?** Not through the public API. Lightning owns the loop, checkpoints, logging, the progress bar and the `output_dir` layout. We could wrap `RFDETRSegXxx.train(**cfg)` in a function, and that wrapper would be clean, but it would *replace* our `train()` rather than plug into it: our `TrainCallback` hooks (`on_train_batch_end`, `state.logits`, ...) have nothing to attach to. Lightning would be a second training framework inside the repo.

**Gotchas seen in my runs** ([run]):

- `train(...)` on a 1-channel model, followed by `predict(...)` **on the same in-memory object**, failed with `ValueError: Make sure that the channel dimension of the pixel values match with the one set in the configuration. Expected 3 but got 1.` Loading the saved checkpoint with `RFDETR.from_checkpoint(path)` and predicting from that worked. Treat "train, save, reload, predict" as the supported path.
- `checkpoint_best_total.pth` was written at epoch 1 (EMA mask mAP 0.90 on a toy set) and gave scores of about 0.03, so nothing passed a 0.3 threshold. `last_ema.pth` from the same run gave scores about 0.93 and correct instances. The best-metric rule is rank-based (mAP), so it can pick a very early, badly-calibrated checkpoint on tiny data. Confirm on real data.
- Weights are downloaded to `~/.roboflow/models/` by default (Section 7).

---

## 3. Findings: data (issue point 2)

### 3.1 Formats

- **Auto-detect:** `train/_annotations.coco.json` => COCO; else `data.yaml` + `train/images` => YOLO. COCO wins if both exist. [docs] `docs/learn/train/dataset-formats.md:7-14`. Splits are named `train/`, `valid/` (and optional `test/`). [docs] same file, "Directory Structure".
- **COCO segmentation:** each annotation needs `bbox`, `area`, `iscrowd`, and `segmentation` as a list of flat polygons `[x1,y1,x2,y2,...]`. [docs] `dataset-formats.md` "Segmentation Annotations".
- **RLE:** the loader accepts it. `convert_coco_poly_to_mask` handles polygon lists and RLE dicts (`{"counts","size"}`, compressed or uncompressed). [wheel] `datasets/coco.py:446-500`. The docs only *show* polygons, and [main] `datasets/coco.py:319,361` states RLE cannot be rescaled by `scale_coco_annotation`. **UNVERIFIED** whether every training-time augmentation path works with RLE; I did not run it. Polygon is the documented path.
- **YOLO-seg:** `class x1 y1 x2 y2 ...` normalised polygons, `data.yaml` with `names`. [docs] `dataset-formats.md` "Segmentation Labels (YOLO-Seg)".
- **Classes and background:** categories are foreground only; "Number of output classes (background-free)" ([main] `config.py:662`). `num_classes` is auto-aligned to the dataset ([main] `detr.py:1385-1392`). In my run a 1-class dataset gave `class_names == ['defect']` and `class_id == 1`; the ONNX `labels` output had width 2 ([run]), i.e. one extra "no-object" slot. [docs] `docs/exports/onnx.md` warns the background slot position depends on the checkpoint layout.

### 3.2 Channels and resolution

- **Channels:** `num_channels` is a model-config field (default 3, `ge=1`). [wheel] `config.py:719`. For non-3 values the patch-embedding weights are adapted: 1 channel = mean over the RGB filters; other counts = tile + rescale. [wheel] `inference.py:352-377`. Docs state a one-channel model reads grayscale PIL/file inputs (`L`). [docs] `docs/learn/inference.md:82`.
- **Smoke result** ([run]): `RFDETRSegNano(num_channels=1)` trained for 60 epochs on 8-bit single-channel PNGs with polygon masks, and a reloaded checkpoint predicted from PIL `L`, NumPy `(H, W)` and NumPy `(H, W, 1)` inputs. Pretrained weights were loaded with a warning: `PretrainWeightsCompatibilityWarning ... num_channels: 1 (variant default: 3) ... will leave significant portions of the model randomly initialised`. [wheel] `config.py` warning; [main] `inference.py:437-457` contains the adaptation code. **UNVERIFIED** how much of the pretrained benefit survives with 1 channel; the warning text suggests it is partial.
- **Our 2-channel / multi-channel images** (`two_channel`, `three_channel` modes in `channels.py`): the tile/rescale path exists in `_adapt_input_conv`, but **I did not run 2 or 4+ channels** and the on-disk COCO dataset loader reads normal image files. **UNVERIFIED.**
- **16-bit images:** **UNVERIFIED** (I only ran 8-bit PNG). Our own review-report spec already notes 16-bit loses range at 8-bit reads.
- **Resolution is fixed per size.** The model is trained and run at 312 to 768 px square. Our patches are typically 256 to 656+ (`docs/superpowers/specs/2026-07-16-patch-656-hpc-design.md`), so a 656 patch fed to Medium (432) is downscaled by ~1.5x. The mask head works at `mask_downsample_ratio = 4` ([wheel] `config.py:741`), so at Nano the native mask grid is 78x78 ([run], ONNX output `masks (1,100,78,78)`).
- **Mask to polygon fidelity** (thin lines, holes, 1-3 pixel defects): **UNVERIFIED**, I did not measure it. Known by construction: polygons cannot hold holes (RLE can), and any polygon simplification loses sub-pixel width. This is the first thing a follow-up must measure on real masks.

---

## 4. Findings: inference and output (issue point 3)

`model.predict(images, threshold=0.5, shape=None, ...)` ([wheel] `detr.py:3149`; docstring [main] `detr.py:3215-3300`):

- Input: path, URL, PIL, NumPy or `torch.Tensor (C,H,W)` in [0,1]; a list is batched into one forward.
- Output: one `supervision.Detections` per image (a list for a list input). Fields seen in my run ([run]): `xyxy (N,4)`, `confidence (N,)`, `class_id (N,)`, **`mask (N, H, W) bool` at the source image size**, `data = {source_shape, class_name}`, `metadata = {source_image}`.
- `threshold` is a score cutoff on up to 100 queries per image (even at `threshold=0.001` the run returned exactly 100 detections). It can also be a per-class `ClassThresholds`. [main] `detr.py:3231-3237`.
- **Semantic map:** no built-in. I rasterised with `sem[d.mask[k]] = d.class_id[k]` (or an `OR` for binary). On the toy set (instance counts matched GT 1,2,1,3) the binary union had IoU 0.94 to 0.97 against the bright-pixel ground truth ([run] `PRED` lines). Overlapping instances need a rule (higher score wins); touching instances merge in a semantic map, which is the opposite of why one would use an instance model.
- **Large images:** `predict` resizes the whole image to the model resolution (`shape=None` = model resolution). A 1152x1632 image ran without error and returned `mask (N, 1152, 1632)` ([run], `BIG` line; it had 0 detections since the image was noise). There is **no tiling or stitching** in the library. Thin or small defects in a 1152x1632 image at 312 px would shrink by ~4x or more, so tiling is on us. Tiles would produce instances cut at tile borders, and merging them (union? NMS?) is not provided. This is the main reason B costs real effort.
- **Batching:** a list of images runs as one batch ([run], `BATCH list 2`). Eval batch via `eval_batch_size`.

---

## 5. Seam-by-seam: what assumes per-pixel logits (issue point 4)

"Carries over" means usable unchanged by that option. Option A = full peer in `get_model()`/`train()`; B = separate instance pipeline sharing build/report/metrics; C = external recipe.

| Seam (repo) | Assumption in the code | A | B | C |
|---|---|---|---|---|
| `get_model()` return contract, [repo] `model.py:358-410` | returns `nn.Module` mapping `(B, in_ch, H, W)` -> `(B, num_classes, H, W)` logits | **Breaks.** RF-DETR's `nn.Module` returns boxes/logits/masks per query (ONNX shows `dets`, `labels`, `masks`). A wrapper could stack masks into a map, but then the Hungarian-matched training is gone | n/a (own factory) | n/a |
| `train_one_epoch`, [repo] `train.py:484-541` | `logits = model(images); criterion(logits, targets)`; AMP scaler, `clip_grad_norm_`, one optimizer | **Breaks.** RF-DETR trains with set-matching loss, EMA, per-layer LR decay inside Lightning | Not used | n/a |
| `train()`, [repo] `train.py:722+` | AdamW + cosine, two LR groups (encoder/decoder), `freeze_epochs`, `latest.pth`/`best.pth`, resume | **Breaks** (all of it is re-implemented by `RFDETR.train`) | Not used; Hydra block maps to `RFDETR.train(**kw)` | n/a |
| Losses, [repo] `losses.py` (`SegLoss`, `TverskyFocalLoss`, `ForeGroundAwareLoss`, hierarchy) | per-pixel probs vs `(H,W)` long or `(K,H,W)` float targets | **Lost.** RF-DETR has fixed mask CE + Dice + box + class losses (coefficients only) | Lost (no Tversky/boundary/hierarchy) | Lost |
| `SegMetrics`, [repo] `metrics.py:19+` | accumulates from logits + targets | Needs logits | **Carries over** if instances are rasterised to a semantic map first | By hand |
| `ObjectMetrics` / `object_metrics`, [repo] `metrics.py:105-150` | works on binary `(H,W)` masks via connected components | Carries over (post-raster) | **Carries over** after rasterising; compare with RF-DETR's own instance mAP | Carries over |
| `predict` / `predict_tiled` / `_activate`, [repo] `inference.py:23-319` | `_activate` = sigmoid/softmax on `(B,C,H,W)`; tile stitching averages per-pixel probs with Gaussian weights | **Breaks.** Instances cannot be averaged across overlapping tiles; scores are per object | Needs a new instance-tile merge; reuse only `tile_starts` | n/a |
| Dataset, [repo] `dataset.py` | returns `{'image','mask'}` tensors from patch cache | Breaks (needs boxes + polygons per instance) | Needs a one-way exporter: patch cache -> COCO folder | Same exporter, run manually |
| Batch augs, [repo] `train.py:617-720` (mosaic/mixup/cutmix/copy_paste) | operate on `(images, targets)` tensors; targets are masks | Breaks (RF-DETR has its own aug stack) | Not applicable; use `aug_config` | n/a |
| `ErrorAnalysisCallback`, [repo] `train.py:297+` | needs `state.model`, ranks per-sample loss/metric from logits | **Breaks** (no `TrainCallback` in Lightning) | Reusable *after training* if given rasterised predictions (the poster only needs image, GT, pred) | Reusable offline |
| `EncoderDriftCheck`, [repo] `train.py:385-434` | `state.model.encoder` state dict | Breaks (RF-DETR backbone is `model.model.backbone`, DINOv2 with windowed attention); freezing is `freeze_encoder` in config | Could be a ~15-line port | n/a |
| `review_report`, [repo] `review_report.py:30-48` | consumes `Sample(image, gt, pred)` NumPy arrays; `pred` same layout as `gt` | n/a | **Carries over** if `pred` is the rasterised semantic map | **Carries over** |
| `export_onnx`, [repo] `export.py:21-71` | traces `model(dummy) -> 'logits'`, dynamic batch, metadata props, ORT parity check | **Breaks** (different outputs) | Use RF-DETR's own `export(format="onnx")` | Same |
| Config (`segpt-train`, `hydra_pipeline`) | `model.*`, `loss.*`, `mask_mode`, `class_names` | Needs a new `model.source: rfdetr` branch that mostly ignores `loss.*`, `freeze_epochs`, `batch_augs` | New `configs/train_instance.yaml` | none |

Net: A breaks nine of fourteen rows; B reuses `build`, `review_report`, `SegMetrics` and `ObjectMetrics` (via rasterising) and `ErrorAnalysisCallback` offline; C keeps the repo untouched.

---

## 6. Option comparison (issue point 5)

| | A: full peer | B: thin instance pipeline | C: external recipe (+ 2 bridge functions) |
|---|---|---|---|
| What it is | `model.source: rfdetr` inside `get_model()`/`train()` | Own entry point, shares patch build, metrics, report | A doc + a notebook + converters; separate env |
| Effort | Large (reimplement or bypass `train_one_epoch`, loss, callbacks, ONNX) | Medium (exporter, tile-merge, rasteriser, config, 1 callback port, tests) | Small (2 pure functions + doc/tutorial) |
| Risk | High: two training frameworks in one loop; every future `train.py` change must consider RF-DETR | Medium: dependency pin conflicts (Section 7); upstream API churn (1.12.0 to 1.13.0.dev changed several defaults) | Low |
| Lost vs our stack | Tversky/boundary/hierarchy losses, `freeze_epochs`, batch augs, `TrainCallback`, dynamic-batch ONNX | Same losses and callbacks; keeps our tooling for build/report/metrics | Same, plus no `segpt-train` ergonomics |
| Gained | Instance outputs, DINOv2 backbone with seg head, built-in EMA/early stop | Instance outputs and a fair side-by-side on our metrics | Ability to try it today |
| Air-gap fit | Same as B; pulls transformers/hf-hub into the main env | Separate env avoids touching main env | Separate env |
| Recommended | **No** | **Later, gated on a benchmark** | **Yes, now** |

---

## 7. Dependency, licence and air-gap risk (issue point 6)

### 7.1 Versions

- **Release facts** (PyPI JSON, `rfdetr 1.12.1`): `requires_python >=3.10`, `torch>=2.2.0`, `transformers>=5.1.0,<6.0.0`, `supervision>=0.29.0`, `pydantic>=2.0,<3`. Extras: `train` (Lightning `>=2.6,<3`, torchmetrics `<1.9`, several COCO eval backends, `peft`, `roboflow`, ...), `onnx` (`onnx<2`, `onnxruntime`). [PyPI] `https://pypi.org/pypi/rfdetr/json`.
- **Install defect found** ([run]): `uv pip install 'rfdetr[train,onnx]==1.12.1'` **fails to resolve** with `Because there is no version of torch-hungarian==0.1.0rc0 ... rfdetr[train]==1.12.1 cannot be used`; it needs `--prerelease=allow`. The `[train]` extra pins a release-candidate package (`torch-hungarian==0.1.0rc0`). With `--prerelease=allow` it installed, but that flag also let `uv` pick other pre-releases (it installed `protobuf 7.37.0rc3`, `safetensors 0.9.0rc1`, `sympy 1.15.0rc3`). In a locked repo this is a real hazard.
- **Pins that would hit our env** ([run] throwaway venv): `huggingface-hub 2.4.0` (ours is 1.27.0), `transformers 5.24.0`, `numpy 2.5.5`, `torch 2.15.0` (the resolver picked a newer torch than the repo's 2.13.0 because the venv was empty; not an rfdetr requirement), `opencv-python-headless 5.1.0.95`.
- **Next release looks lighter (not shipped):** [main] `CHANGELOG.md:21-31` (section `[Unreleased]`) says "`rfdetr` no longer depends on `transformers`" for a core install; `rfdetr[train]` still gets it through `peft`. Core deps in [main] `pyproject.toml` no longer list `transformers`. This helps only the core, not training. **Do not plan on it until released.**
- **Does our DINOv3 path survive hf-hub 2.4?** Yes in a test ([run]): in the throwaway venv with `timm 1.0.28`, `segmentation-models-pytorch 0.5.0`, `huggingface-hub 2.4.0`, `transformers 5.24.0`, I imported the repo's `model.py` (read-only, bytecode off) and built `get_model` for `resnet18`, `convnext_tiny` and `dinov3_vit_small` at `in_channels=1, num_classes=3`; `dinov3_vit_small` with `pretrained=True` downloaded weights through hub 2.4 and returned `(1, 3, 256, 256)`. This is a build-and-forward check only; I did not run the repo test-suite or a training step.

### 7.2 Licence

- `rfdetr` package: Apache 2.0 ([wheel] `rfdetr-1.12.1.dist-info/METADATA`; [main] `LICENSE`).
- **All RF-DETR-Seg sizes (Nano to 2XLarge) are Apache 2.0**, including XL and 2XL. [main] `README.md:23,115-120`; [docs] `docs/learn/run/segmentation.md` ("Every RF-DETR-Seg size ... Apache 2.0 ... No `rfdetr[plus]` install is needed").
- PML 1.0 (Roboflow Platform Model License) applies to Atto/Femto/Pico/XL/2XL **detection** models in `rfdetr[plus]` and the NAS models. [main] `README.md:23,312-322`. Not relevant to seg, but do not install `[plus]` without legal review.
- Pretrained seg weights are COCO-trained and served from `repo.roboflow.com` ([wheel] `assets/model_weights.py`). Whether the weight files carry any extra terms beyond the stated Apache 2.0 is **UNVERIFIED**; the README states the models are Apache 2.0.

### 7.3 Pretrained weights, offline, HPC

- Mechanism: on first use `download_pretrain_weights()` fetches e.g. `https://repo.roboflow.com/rfdetr/rf-detr-seg-n-ft.pth` and caches it under `~/.roboflow/models/` with MD5 validation. [wheel] `assets/model_weights.py:38,319`; [main] `assets/model_weights.py:243-260`. In my run `rf-detr-seg-nano.pt` is 128 MB.
- **Pre-staging works** ([run]): passing `pretrain_weights='<local path>'` to the constructor loaded the file with an "already exists with correct MD5 hash" log and no new download. For HPC, copy the file to the node and pass the path (or place it in `~/.roboflow/models/`). I did not test with the network blocked, so a full offline run is **UNVERIFIED**; the code path checks the local file first.
- `rfdetr[train]` also pulls `roboflow` (a hosted-platform client) and `torchmetrics[detection]`; these need no network at train time as far as I saw, but I did not audit for telemetry. **UNVERIFIED.**
- Our existing air-gapped Artifactory mirror (`uv` with a mirror) would need to carry `torch-hungarian 0.1.0rc0` and the other extras listed above.

---

## 8. ONNX export (issue point 7)

`model.export(format="onnx", output_dir=...)` ([wheel] `detr.py:2072`; docs [main] `docs/exports/onnx.md`). In my run ([run]) for a 1-channel Nano seg model:

| Item | RF-DETR | Our `export_onnx` ([repo] `export.py`) |
|---|---|---|
| Input | `input`, shape `[1, 1, 312, 312]` (fixed batch and fixed size) | `image`, dynamic batch, size `input_size` |
| Outputs | `dets (1,100,4)` normalised cxcywh, `labels (1,100,2)` raw logits (per-class sigmoid), `masks (1,100,78,78)` raw mask logits | `logits (B, C, H, W)` |
| Batch 2 / size 384 in ORT | `INVALID_ARGUMENT: Got invalid dimensions for input` for both | batch is dynamic |
| Dynamic batch | Opt-in `dynamic_batch=True` exists ([wheel] `detr.py:1983,2081`); not tested by me | built in |
| Opset / exporter | opset 17 in the file I produced | opset 18 default, `dynamo=False` |
| Metadata | JSON under key `rfdetr_inference` (schema, producer version, task `segment`) | `task`, `class_names`, `thresholds`, `hierarchy_pairs` props |
| Decode | Needs top-k + sigmoid + threshold + mask upsample + paste at the consumer, or `RFDETRInference("model.onnx")` (Python only, supervision output) | consumer does sigmoid/softmax per pixel |

ORT inference at the fixed shape ran fine on CPU ([run], `ORT1`). So FP32 ONNX works, but **it cannot satisfy our `export_onnx` conventions** (the `logits` name, per-pixel layout, dynamic batch, our metadata props). The downstream line consumer would need a second decoder. INT8 and TensorRT are out of scope here; RF-DETR documents both ([docs] `docs/exports/onnx.md` "INT8 Quantization": an expected cost of about 2.7 mAP on COCO Nano/Small).

---

## 9. Evidence on value (issue point 8)

**What the published numbers say.** COCO instance-seg mask AP50:95 for RF-DETR-Seg is 40.3 (N) to 49.9 (2XL), T4 TensorRT FP16 latency 3.4 to 21.8 ms, against YOLO26-N/L/X-Seg 34.7 / 45.5 / 46.8 ([main] `README.md:113-135`, measured by Roboflow itself). The paper (arXiv 2511.09554) is about detection NAS and, from its abstract, does not cover instance segmentation, so the seg numbers come from the README only. These are COCO object benchmarks (large, everyday, countable objects). They say nothing about U-Net-style semantic models or thin industrial defects, and are not comparable to our IoU/Dice.

**Where an instance model is likely good (reasoning, not measured here):** separate, countable objects that touch or overlap (pins, solder balls, voids as individual blobs), where you want a count and a per-object score and a box.

**Where it is likely weak for us:** thin or tiny defects, because (a) input is 312 to 768 px with masks at a quarter of that, (b) large images need tiling that the library does not provide, (c) the per-pixel losses we tuned (Tversky recall weighting, boundary loss, hierarchy constraint) do not exist there, and (d) a semantic "chip / void / lifted" multi-head with a subset rule is not an instance task.

**Smoke test result** ([run], details in Appendix A): install, 3-channel and 1-channel training on GPU, reload, predict, raster, ONNX all worked on synthetic discs. It proves **plumbing only**. It proves nothing about accuracy on real data: the discs are easy, the set is 12+4 images, and the semantic IoU of 0.94 to 0.97 is against a bright-pixel proxy.

**Not done:** comparison with our U-Net/DINOv3 on real defect data (no real data in this task; per `docs/superpowers/specs/2026-10-08-review-report-design.md`, real data is read from config paths and is never committed). A small A/B with `ObjectMetrics` is the right follow-up.

---

## 10. Open questions

1. Does RF-DETR-Seg beat our DINOv3-UNet on `ObjectMetrics.detection_rate` and false alarms on real patches (use the same val ids)?
2. How much accuracy does the mask-to-polygon step lose on thin defects (polygon vs RLE)? Is RLE safe through the train-time augmentations?
3. Do 2-channel and 16-bit inputs work, and how much of the pretrained gain remains with `num_channels != 3`?
4. What merge rule for instances cut by tile borders (union, NMS on mask IoU, or keep tile-local)?
5. Is `checkpoint_best_total` ever the right one on small datasets? (My run picked epoch 1.)
6. Can the `[train]` extra install without `--prerelease=allow` in the Artifactory mirror, and what is the network behaviour with `HF_HUB_OFFLINE=1` and no outbound access (including `roboflow`)?
7. Is a separate virtualenv acceptable operationally on the HPC nodes (`segpt-submit` uses one `uv sync`)?

---

## 11. Proposed follow-up tickets

Order: umbrella first, then one implementable slice. Each slice is notebook-first (`nbs/`), then `nbdev-export`; none edits `src/` by hand.

1. **Umbrella / design: "RF-DETR seg as an instance side-pipeline"** (`wayfinder:map`). Records: recommendation C now, B gated, A rejected, the Section 5 table, and the decision rule "go to B if RF-DETR beats the best semantic model on `detection_rate` at equal false alarms on real val patches". Blocked by nothing. Includes the dependency/air-gap checks from Section 7 as acceptance notes.
2. **Slice 1 (implementable): bridge functions + external recipe.**
   - `masks.py`: `mask_to_coco_instances(mask, ...)` (connected components -> polygons, RLE option, holes handled) and a COCO-folder writer from the patch cache.
   - `inference.py` (or a new small module): `detections_to_semantic(dets, shape, class_ids, rule='score')` producing `(H, W)` long or `(K, H, W)` float, so `SegMetrics`, `ObjectMetrics` and `review_report.Sample` can consume RF-DETR output.
   - A tutorial notebook `tutorials/` (train outside, load back, compare) and a `docs/TRAINING_CLI.md` pointer. Tests are synthetic; no `rfdetr` import in the library (the functions take plain arrays).
   - Acceptance: round-trip mask -> COCO -> mask IoU >= 0.99 on synthetic thin lines, holes and 2-px dots; document the loss for each.
3. **Slice 2 (only if ticket 1's rule says go): benchmark ticket** on real patches (config paths only), with the external env, producing a `review_report` side by side. This is the decision input for B.
4. **Slice 3 (only after 3): B entry point** `segpt-train-instance` + `configs/train_instance.yaml`, instance tile-merge, `ErrorAnalysisCallback` port, its own ONNX docs. Not scheduled now.

Do not file: an A ticket, or any change to `pyproject.toml`.

---

## Appendix A: what I ran and what happened

Environment: one throwaway venv outside the repo (`<scratchpad>/venv`, Python 3.12.3, uv 0.11.26), NVIDIA GB10 GPU, aarch64. Source clone in its own directory. Python run with `-I`. Nothing was installed into the repo env; `git status --short` in the repo was unchanged (only the pre-existing untracked entries) before this file was added.

| Step | Result |
|---|---|
| `git clone --depth 1 roboflow/rf-detr` | `1268d0a`, version `1.13.0.dev` |
| PyPI JSON | latest `1.12.1`, Apache 2.0, requires Python >=3.10 |
| `uv pip install 'rfdetr[train,onnx]==1.12.1'` | **Failed**: `no version of torch-hungarian==0.1.0rc0 ... cannot be used` (needs `--prerelease=allow`) |
| Same with `--prerelease=allow` | OK (about 5 min, torch + CUDA wheels dominate); `huggingface-hub 2.4.0`, `transformers 5.24.0`, `pytorch-lightning 2.6.6`, `supervision 0.32.0`, `onnx 1.23.2`, `onnxruntime 1.32.0`, `numpy 2.5.5` |
| Synthetic COCO sets | 12 train + 4 valid images, 192x192, 1-3 filled discs, 16-point polygons, one class; one RGB-PNG copy, one 8-bit single-channel copy |
| RGB Nano, 1 epoch, bs 2, CUDA | Pretrained weights downloaded (128 MB, MD5 OK); trained; `predict` returned `Detections` with `mask (0,192,192) bool` (no hits at 0.05, 1 epoch only) |
| 1-channel Nano, 12 epochs, `predict` in the same process | **ValueError**: `Make sure that the channel dimension of the pixel values match with the one set in the configuration. Expected 3 but got 1.` |
| Same, `RFDETR.from_checkpoint(...)` then `predict` | Worked for PIL `L`, NumPy 2-D, NumPy 3-D `(H,W,1)`; `class_names == ['defect']` |
| 1-channel Nano, 60 epochs, lr 3e-4, bs 4 (117 s on GPU) | `checkpoint_best_total.pth` (epoch 1): scores max about 0.03, nothing above 0.3. `last_ema.pth`: scores about 0.93, instance counts 1, 2, 1, 3 (= GT), union-mask IoU 0.942 / 0.951 / 0.966 / 0.943 vs bright-pixel proxy; validation EMA mask mAP50:95 0.20 (epoch 0) to 0.87 (epoch 59) |
| Large image 1152x1632 | `predict` ran, returned masks at source size; no tiling API |
| List of 2 images | returns a list of 2 `Detections` |
| `export(format="onnx")` | `rfdetr-seg-nano.onnx` 138 MB; input `input [1,1,312,312]`; outputs `dets [1,100,4]`, `labels [1,100,2]`, `masks [1,100,78,78]`; opset 17; ORT CPU ran at batch 1 / 312 px; batch 2 and 384 px raised `INVALID_ARGUMENT: Got invalid dimensions for input` |
| `pip install timm==1.0.28 segmentation-models-pytorch==0.5.0` into the throwaway venv, then import repo `model.py` read-only | `get_model` OK for `resnet18`, `convnext_tiny`, `dinov3_vit_small` (pretrained False and True, 1-channel input, 3 classes) under `huggingface-hub 2.4.0` |
| arXiv 2511.09554 abstract | "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers"; the abstract does not mention instance segmentation |

Cleanup: the weight file the library cached at `~/.roboflow/models/rf-detr-seg-nano.pt` was deleted afterwards. Everything else is under the scratchpad directory.

## Appendix B: not verified

- Real-data accuracy, thin-defect fidelity of mask -> polygon, RLE training path, 2-channel / 4+ channel / 16-bit inputs.
- A full run with the network blocked; telemetry of the `roboflow` client.
- `dynamic_batch=True` ONNX export; INT8/TensorRT (out of scope).
- Any claim about rfdetr 1.13.0 (unreleased); the `transformers`-free core install is from the `[Unreleased]` changelog only.
- The Roboflow documentation website itself (rfdetr.roboflow.com) was not fetched; the `docs/` folder in the repository, which it is built from, was read instead.
