# segmentation_tools_pt

PyTorch utilities for image segmentation — data preparation, mask handling, augmentation, and evaluation — built for production use.

## Goal

Image segmentation workflows involve a lot of repeated, error-prone boilerplate: loading masks, converting label formats, applying paired augmentations, computing IoU metrics. This library collects those building blocks into a clean, tested API so that each new segmentation project starts from a solid foundation instead of copy-pasted scripts.

Built with **nbdev** — every module is authored as a Jupyter notebook under `nbs/`, then exported to `src/`. The notebooks are the documentation.

> **New here? Read [AGENTS.md](AGENTS.md) first.**
> It covers the full repo structure, every module's purpose, the two training modes, the config format, and key design decisions — everything you need to understand or extend this library without reading the entire codebase.

## Project structure

```
segmentation_tools_pt/
├── nbs/                        # source notebooks (edit here, not in src/)
│   ├── index.ipynb             # library overview / README source
│   ├── 00_core.ipynb           # core utilities
│   └── ...
├── src/
│   └── segmentation_tools_pt/  # auto-exported Python package
│       ├── __init__.py
│       ├── core.py
│       └── ...
├── configs/                    # Hydra training configs
│   ├── train.yaml              # default config used by scripts/train.py
│   └── train_example.yaml      # template to copy into your project
├── docs/
│   └── TRAINING_CLI.md         # segpt-train user guide
├── scripts/
│   └── train.py                # Hydra training entry (also via segpt-train)
├── tutorials/                  # step-by-step notebooks
├── pyproject.toml
└── uv.lock
```

> **Rule:** never edit files inside `src/` directly. Edit the corresponding notebook in `nbs/` and re-export.

## Install

```bash
git clone https://github.com/HasanGoni/segmentation_tools_pt.git
cd segmentation_tools_pt
uv sync               # creates .venv/, installs all deps + CLI commands
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates `.venv/`, installs all dependencies, and installs the package itself (including the `segpt-build`, `segpt-train`, and `segpt-submit` entry points). No separate `pip install -e .` step is needed.

Requires Python >= 3.12 and PyTorch (install separately to match your CUDA version).

## Quick start

### Training CLI (recommended)

Set up the environment, copy the example config, point at your data, train:

```bash
uv sync
source .venv/bin/activate
cp configs/train_example.yaml /path/to/my_train.yaml
# edit data.raw_dir and data.patch_dir in my_train.yaml
segpt-train /path/to/my_train.yaml
```

Build patches only from the same config file:

```bash
segpt-build /path/to/my_train.yaml
segpt-build /path/to/my_train.yaml data.force_rebuild=true patch.patch_size=512
```

Recommended CV workflow for large industrial images:
- extract larger reusable patches with `patch.patch_size` for spatial context
- train with smaller random crops via `train.crop_size`
- validate with deterministic center crops of the same `train.crop_size`

Example: save 512x512 patches, then train and validate on 256x256 crops.

Full guide: [docs/TRAINING_CLI.md](docs/TRAINING_CLI.md)

### Python API

```python
from segmentation_tools_pt.model import get_model
from segmentation_tools_pt.train import TrainCallback, train

model = get_model(arch="mobilenet_v2", architecture="unet", in_channels=1, num_classes=1)
history = train(root="/path/to/patch_dataset", model=model, num_epochs=60)
```

Optional **training callbacks** hook into each train/val batch and epoch — subclass
`TrainCallback`, pass `callbacks=[...]` to `train()`. See `AGENTS.md` → *Training callbacks*.

Notebook walkthrough: [tutorials/01_training_walkthrough.ipynb](tutorials/01_training_walkthrough.ipynb)

Full API docs: <https://hasangoni.github.io/segmentation_tools_pt/>

## Development workflow

Before contributing or extending the library, read [AGENTS.md](AGENTS.md). It describes every module, the nbdev golden rule, coding conventions, and common workflows — you won't need to read the full codebase.

This project uses [nbdev](https://nbdev.fast.ai/). Edit notebooks, export, then commit both.

Because the repo lives on a slow network share, run `nbdev-export` on a remote host (`hpc-cluster`) where the share is mounted at `/path/to/segmentation_tools_pt`:

```bash
ssh hpc-cluster 'cd /path/to/segmentation_tools_pt && uv run nbdev-export'
```

For the full HPC nbdev workflow, invoke the `nbdev-export` skill or see [AGENTS.md](AGENTS.md).

## Testing Copilot skills

If you add a skill under `.github/skills/<name>/SKILL.md`, run the smoke-test notebook to catch common structural mistakes (bad YAML frontmatter, missing description keywords, broken references) before committing. Step-by-step guide: [tutorials/03_testing_skills.ipynb](tutorials/03_testing_skills.ipynb).

The notebook prints **PASS / FAIL** per skill and suggested chat prompts so you can verify the skill actually loads in Copilot Chat.

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Author

Goni Mohammed Hasan
