# Segmentation tools

Industrial and general image segmentation with this library: users drive behaviour through Hydra config; product recipes (e.g. Chip/Void/Lifted) are documented opt-ins, not forced library defaults.

## Language

**Library default**:
The silent starting config in `configs/train.yaml` (today MobileNetV2) so existing and generic runs stay unchanged unless the user opts in.
_Avoid_: primary, recommended backbone (when meaning the YAML default)

**Primary recommendation**:
The documented starting encoder for a named product path (e.g. Chip/Void/Lifted under domain shift), chosen via config/preset — not the silent library default.
_Avoid_: library default, always-on backbone

**Throughput alternative**:
A documented faster/cheaper encoder for the same product path when speed or memory matters more than maximum robustness.
_Avoid_: secondary default, fallback (unless meaning failure recovery)

**Curated encoder**:
An encoder the library treats as first-class for config and agents: advertised by `list_smp_encoders()`, documented in skills, and usable through `get_model` / Hydra without warnings (including friendly aliases such as `convnext_tiny`).
_Avoid_: experimental tu-* encoder, unsupported backbone

**Encoder preset package**:
A Hydra config group file under `configs/model/` that sets `model.*` only (`architecture`, `arch`, pretrained defaults) for one curated encoder. Composed via `defaults:`; shut-off means selecting the library-default package (`mobilenet_v2`). Does not bake `train.freeze_epochs` / LRs — those stay on the calling train/recipe config.
_Avoid_: backbone_preset flag, train-schedule-in-encoder-pack

**Experimental tu-* encoder**:
Any other SMP `tu-*` (timm) encoder accepted by `get_model` with a warning — allowed so agents can still build scripts, but not curated or smoke-guaranteed.
_Avoid_: curated encoder, power-user-only path

**Model source** (`model.source`):
The config switch that says where the model comes from: `smp` (default; a `segmentation-models-pytorch` architecture and encoder built by `get_model`) or `custom_unet` (the library's own UNet). `get_model_from_cfg` reads it; when it is absent the run is `smp`.
_Avoid_: model type, backbone flag

**Custom UNet** (`custom_unet`):
The library-owned classic UNet (block style selectable via `block_style`, `double` conv by default or `residual`; norm selectable via `norm`, GroupNorm by default; explicit `encoder_depth`; optional `aspp` or `attention` bottleneck via `bottleneck`) selected with `model.source: custom_unet` and tuned under `model.custom_unet.*`. It starts from random weights, so training gives it no freeze warm-up and the same learning rate for encoder and decoder unless the user sets them.
_Avoid_: pretrained encoder, smp UNet, curated encoder

**Freeze warm-up**:
Training with the encoder frozen for the first `freeze_epochs` epochs while the decoder learns, then unfreezing the encoder at a lower learning rate.
_Avoid_: freeze forever, LoRA

**Freeze forever**:
Keeping the encoder frozen for the entire run (`freeze_epochs: -1`) so only the decoder (and head) train.
_Avoid_: freeze warm-up, scratch training

**Batch-level augmentation**:
Training transforms that mix or compose multiple samples in a batch (MixUp, CutMix, mosaic, copy-paste), gated by `augment.batch.enabled` and per-type probabilities.
_Avoid_: geometric/photometric stack (crop, affine, noise) — those are sample-level

**YOLO-style pick-one**:
At most one batch-level augmentation type is applied per training batch, chosen by the configured probabilities among types with probability greater than zero.
_Avoid_: stacking MixUp then CutMix in the same step

**Class-targeted morph postprocess**:
Inference/eval-only classical cleanup (morphological opening + connected-component min-area) applied to one user-named class channel looked up in `data.class_names`, gated by a config master switch (default off). Not applied to loss or raw training metrics; product recipes (e.g. `class_name: void`) opt in via YAML.
_Avoid_: Void-head postprocess (as library vocabulary), train-time morph, whole-mask stitching presets alone

**Review report**:
The single self-contained HTML page for checking the visual result of a ticket or evaluation run (images, metrics, gates) without starting a kernel. Evaluation and embedding work add sections to it; they do not build their own pages.
_Avoid_: dashboard, experiment tracker, notebook output

**Gate**:
A pass/fail rule on one named metric (for example `dice_min: {void: 0.85}`), read from a gates file and shared by the review report and the run-to-run regression check. A gate whose metric is missing fails.
_Avoid_: threshold (that means the decision threshold on probabilities), check

**Section**:
One block of a review report, produced by a registered function that returns HTML plus flat metrics.
_Avoid_: tab, panel, plugin page

**Feedback item**:
One click-and-comment on a review report, stored as JSON with coordinates in original image pixels (not thumbnail pixels), so the agent can fix exactly that spot.
_Avoid_: annotation, label, screenshot note
