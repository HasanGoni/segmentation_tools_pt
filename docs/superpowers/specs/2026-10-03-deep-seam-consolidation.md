# Deep seams: tiled inference + ONNX export consolidation

**Date:** 2026-10-03 (spec written 2026-10-08)  
**Wayfinder map:** [Deep seams: tiled inference + ONNX export consolidation](https://github.com/HasanGoni/segmentation_tools_pt/issues/23)  
**Spec issue:** [Spec: deep-seam consolidation](https://github.com/HasanGoni/segmentation_tools_pt/issues/24)  
**Implementation tickets:** [#27 tiled-inference seam](https://github.com/HasanGoni/segmentation_tools_pt/issues/27) (first), [#26 ONNX-export seam](https://github.com/HasanGoni/segmentation_tools_pt/issues/26)  
**Glossary:** `CONTEXT.md`  
**Implementation contract:** `docs/agents/nbdev-implementation-contract.md`

---

## Problem Statement

The library has **two parallel interfaces for the same job** in the tiled-inference / ONNX path:

1. `inference.predict_tiled` — one call: clamped sliding-window starts, batched forward, Gaussian-weighted stitching, batch-size-invariant output, config knob `inference.inference_batch_size`. Used by the CLI path, skills, tutorials, and tested (batch invariance, small-image single tile).
2. The **Oct-1 cluster** — `tiling_core` / `tiling` / `model_wrapper` / `onnx_export` / most of `stitching`: `TileConfig` + `InferenceConfig` + `PaddingMode` + `MergeStrategy` + `PatchCoords` + `TileProcessorONNX` + 3 visualizers + 2 ONNX-runtime wrapper classes + 5 `onnx_export` functions ≈ **15+ surface items before any behaviour**. Repository-wide grep (scripts/, configs/, docs/, tutorials/, skills/, notebooks) finds **zero callers** outside the cluster's own tests and `_modidx.py`; its own test for `TileMergerONNX` already fails (`_forward_average` AttributeError) — knowledge spread across files means bugs concentrate nowhere.

The ONNX-export seam is forked the same way: `export.export_onnx` (CLI-facing, metadata-decodable, runtime-verified, #20 smoke-tested) vs `onnx_export.export_to_onnx` (cfg dataclass, onnxsim simplify, checker) with divergent parameter names (`output_path`/`input_size` vs `save_path`/`sample_input`).

Left alone, every future inference/export change pays twice: two interfaces to read, two to test, two to document — and the fork with zero users keeps rotting.

## Solution

**Consolidate each seam onto the one interface that has users and passing tests.**

1. **Tiled-inference seam (#27):** `inference.predict_tiled` becomes the single external interface. The Oct-1 cluster is retired; the only Oct-1 behaviour worth keeping — `reflect` tile-edge padding — is folded in as an internal option (`padding_mode: 'zero' | 'reflect'`, default `'zero'`, preserving today's behaviour). `gaussian_weight_map` moves into `nbs/11_inference.ipynb` as a private helper so `inference.py` loses its last dependency on the cluster and `02_stitching.ipynb` can be deleted outright.
2. **ONNX-export seam (#26):** `export.export_onnx` is the single interface. It absorbs the two `onnx_export` features with real value — `simplify: bool = False` (onnxsim, graceful no-op if the package is missing) and explicit `onnx.checker` validation (already present, stays unconditional). Current parameter names, metadata props, and the runtime verify step are kept unchanged.
3. **Internal `_activate` seam:** already shipped in #25 (commits `6ad939f`/`afbe820` + `tutorials/05_activation_semantics.ipynb`); recorded here only so the spec matches the map.

Retirement is *deletion, not deprecation*: no external callers exist, so there is nobody to warn.

## Implementation Decisions

### Per-module disposition

| Module (notebook → src) | Decision | Detail |
|---|---|---|
| `00_tiling_core` → `tiling_core.py` | **Retire** | `TileConfig`, `InferenceConfig`, `PaddingMode`, `MergeStrategy`, `PatchCoords`, `pad_to_divisible`, `unpad_image`, `PadLTRB` all go. The one behaviour with a use case — reflect tile-edge padding — is re-implemented in `predict_tiled` as a plain string option, not an enum. |
| `01_tiling` → `tiling.py` | **Retire** | `TileProcessorONNX` and the three visualizers (`visualize_tile_grid`, `visualize_tile_overlay`, `visualize_tile_positions`) go. Tile-layout debugging, if ever needed again, is a ~10-line matplotlib plot over the public `tile_starts` — trivially reopenable. |
| `02_stitching` → `stitching.py` | **Retire after salvage** | Move `gaussian_weight_map` into `nbs/11_inference.ipynb` as private `_gaussian_weight_map` (single consumer: `predict_tiled`; zero external callers). Then delete the notebook/module: `POSTPROCESSING_PRESETS`, `PatchAccumulator`, `TileMerger`, `TileMergerONNX` (contains the known-broken `_forward_average` path), `build_postprocessing_pipeline`, both tile-overlay visualizers, `plot_weight_maps`, `compare_binary_mask_merge`. |
| `03_model_wrapper` → `model_wrapper.py` | **Retire** | `TiledInferenceModelONNX`, `DualOutputTiledInferenceModelONNX`, `default_model_loader` go. These wrap the retired cluster's ONNX graph; without the cluster they have no input. Reopen if a concrete ONNX-runtime deployment path materializes (see Deferred). |
| `04_onnx_export` → `onnx_export.py` | **Retire; absorb two features** | `OnnxExportConfig`, `export_to_onnx`, `validate_with_onnxruntime`, `load_onnx_model`, `run_onnx_inference` go. Absorbed into `export_onnx`: `simplify` knob (onnxsim, try-import + warn-and-skip like the existing `verify` ImportError handling) and unconditional `onnx.checker` validation (already in `export.py` today — keep). |
| `11_inference` → `inference.py` | **Keep = the seam** | `tile_starts` + `predict_tiled` unchanged except: new `padding_mode` option; `_gaussian_weight_map` moved in; `from .stitching import …` removed. Public API gains zero new names beyond the `padding_mode` parameter. |
| `13_export` → `export.py` | **Keep = the seam** | `export_onnx` signature unchanged (same names: `output_path`, `input_size`, `verify`, `task`, `class_names`, `thresholds`, `hierarchy_pairs`); adds `simplify: bool = False`. |

### Tiled-inference seam details (#27)

- `predict_tiled(model, image, device, tile_size=256, overlap=32, inference_batch_size=8, activate=True, task=None, padding_mode='zero')`.
- `padding_mode`: `'zero'` (default — today's behaviour, bit-identical) or `'reflect'` (Oct-1 salvage: tile extraction at image edges reflects instead of zero-filling; useful when edge tiles contain real structure that zero-pad darkens). Applies to **small-image padding and tile extraction at edges**; it is a decoding detail, not a config surface — no Hydra key unless a project asks.
- Merge strategy: **Gaussian-weighted only.** `MergeStrategy` alternatives (overwrite, majority-vote, confidence-weighted, average) are retired; confidence-weighted is explicitly deferred (see Fog) rather than dropped silently.
- Batch invariance property is preserved and stays under test: `inference_batch_size=1` output ≡ any larger value.

### ONNX-export seam details (#26)

- `export_onnx(model, output_path, in_channels=3, input_size=256, opset=18, verify=True, simplify=False, task=None, class_names=None, thresholds=None, hierarchy_pairs=None)`.
- `simplify=True` runs `onnxsim.simplify` after export and before checker/verify; if `onnxsim` is not installed, warn once and continue unsimplified (mirrors the existing `verify` soft-fail on missing `onnxruntime`). `onnx`/`onnxsim` stay optional deps in `pyproject.toml`.
- Order after export: (optional) simplify → `onnx.checker.check_model` (unconditional) → metadata props (already present) → runtime verify (already present).
- No new metadata keys; the metadata-decode chain (`decode → apply_class_morph → apply_hierarchy`) is untouched.

### Notebook and index hygiene

- Delete the five notebooks above; run `uv run nbdev-export` so `src/`, `_modidx.py`, `llms.txt`, and Quarto docs regenerate without them.
- Update `docs/agents/repo-guide.md` notebook→module mapping and the AGENTS.md module table (drop `tiling*`/`stitching`/`onnx_export`/`model_wrapper` rows; note `_gaussian_weight_map` lives in `inference`).
- Sync skills that touch the seams: `run-inference` (already `predict_tiled`-centric — verify only), `export-onnx` (add `simplify`), `choose-tiling-strategy` (point at `predict_tiled`/`tile_starts` only). Update `docs/TRAINING_CLI.md` if it names any retired symbol.

### Process

- Two implement slices in order: **#27 first** (largest deletion; unblocks the import graph), then **#26** (small, isolated). Both follow the nbdev contract: narrative → experiment on synthetic tensors → verify → `#| export` → test cell → `nbdev-export` → `nbdev-test`.
- Landed on `feat/onnx-batched-tile-inference`; one commit per ticket, closing comments map deliverables to commits.

## Testing Decisions

Tests assert external behaviour on synthetic tensors; retired modules need no migration tests — their absence is the test.

Agreed seams:

1. `predict_tiled` parity — zero-pad output unchanged vs pre-refactor on a fixed-seed synthetic image (regression guard).
2. `padding_mode='reflect'` — differs from `'zero'` on an edge-structure synthetic image, raises on unknown mode, and both modes agree when no edge tiles exist (small image, single tile).
3. Batch invariance — `inference_batch_size=1` vs `8` max-abs residual < 1e-5 (existing test, kept green).
4. `export_onnx` smoke — binary and multilabel 3-head builds via `get_model`; `simplify=False` byte-path unchanged; `simplify=True` with onnxsim absent warns and still exports + verifies; checker runs unconditionally.
5. Import-graph hygiene — `python -c "import segmentation_tools_pt.inference, segmentation_tools_pt.export"` succeeds after deletions; `grep` confirms no residual `stitching|tiling|model_wrapper|onnx_export` referents outside history.
6. Full `uv run nbdev-test` green at the end of each slice.

## Out of Scope

### Explicitly out of this effort

- Any new inference features (TTA inside `predict_tiled`, non-square tiles, arbitrary rotation).
- Changes to `predict`, `decode`, `apply_class_morph`, `apply_hierarchy`, or the config adapter `resolve_inference_cfg` (beyond what the deletions force).
- INT8 / TensorRT export.

### Deferred — fog items from the map, resolved or parked

| Item | Resolution | When to reopen |
|------|-----------|----------------|
| `merge_strategy: confidence-weighted` | **Deferred** (not folded, not silently dropped) | A concrete case appears where Gaussian weighting under-blends high-confidence tiles — then fold as an option, with a test image where it beats Gaussian. |
| ONNX-runtime wrapper classes (`TiledInferenceModelONNX`, `DualOutput…`) | **Retired** | A real ONNX-runtime deployment path (no-PyTorch consumer) materializes. |
| Tile-grid visualizers | **Retired** | Someone needs tile-layout debugging — re-add as a thin adapter over public `tile_starts`. |
| `padding_mode: reflect` as a Hydra config key | **Deferred** (library option only for now) | A project config asks for it; then add `inference.padding_mode` + `resolve_inference_cfg` wiring. |
| Custom `InferenceConfig`/`TileConfig` dataclasses | **Rejected** | Optionality already fits in `predict_tiled`'s signature; dataclasses add surface without behaviour (the original design-review finding). |

## Further Notes

- Design-review findings and expert-default grill live on map [#23](https://github.com/HasanGoni/segmentation_tools_pt/issues/23) (2026-10-03); this spec records them as binding decisions.
- Vocabulary: seam, deep/shallow module, deletion test, tile-edge padding, merge strategy — see `CONTEXT.md`.
- Internal-seam work (#25) shipped before this spec; `_activate` in `nbs/11_inference.ipynb` is the pattern the remaining consolidation follows: one behaviour, one private helper, public surface unchanged.
