# nbdev implementation contract

All library changes in `segmentation_tools_pt` follow **nbdev literate programming** (see `AGENTS.md`, skill `literate-nbdev-notebooks`, skill `nbdev-export`). There is no separate export path: **`src/segmentation_tools_pt/` is generated; edit `nbs/` only.**

## Per feature / ticket slice

1. **Narrative** — In the owning notebook (`nbs/NN_*.ipynb`), write markdown first: problem, constraints, and a **step-wise bullet plan** (what we want, in order).
2. **Experiment** — One code cell at a time on **synthetic** data (no external datasets required for tests). Print or plot so the step is visible.
3. **Verify each step** — Before promoting, assert or inspect output in the notebook; fix the story if a step fails.
4. **Export** — When the step sequence works, promote to `#| export` function(s) with a fastai-style docstring and a one-line usage example.
5. **Test cell** — Directly under the export, a test cell on the **same** synthetic sample (`nbdev-test` runs these).
6. **Sync** — On Linux/HPC: `uv run nbdev-export`, then `uv run nbdev-test --path nbs/<notebook>.ipynb`. Do not hand-edit `src/`.

## Where work lands

| Kind of change | Primary notebook(s) |
|----------------|-------------------|
| Model / encoders | `nbs/09_model.ipynb` |
| Train / callbacks | `nbs/10_train.ipynb` |
| Visualize / error analysis | `nbs/12_visualize.ipynb` |
| Transforms / augs | `nbs/05_transforms.ipynb`, `nbs/10_train.ipynb` |
| Hydra / CLI wiring | `nbs/15_hydra_pipeline.ipynb`, `nbs/14_cli.ipynb` |
| Export / ONNX | `nbs/13_export.ipynb` |
| Inference / postprocess | `nbs/11_inference.ipynb`, stitching notebooks as applicable |
| Config-only presets | `configs/*.yaml` + skill/docs; no `src/` unless a resolver needs code |

Wayfinder **decision** tickets close with a decision comment; **implementation** tickets (from `/to-tickets`) must cite this contract in the test plan.
