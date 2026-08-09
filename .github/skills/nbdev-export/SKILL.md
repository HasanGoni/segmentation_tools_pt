---
name: nbdev-export
description: 'Regenerate the Python package from the nbs/ notebooks and run nbdev tests on a remote host using the correct hyphenated nbdev v2 commands (nbdev-export, nbdev-test, nbdev-readme) — never the legacy underscored v1 names (nbdev_export, nbdev_test, nbdev_readme) or python -m nbdev.test. Use when a user edits a notebook in nbs/, needs to sync src/ with nbs/, runs or asks about "nbdev_export", "nbdev-export", "nbdev_test", "nbdev-test", "regenerate src", or "run nbdev tests".'
argument-hint: '[notebook-name.ipynb]'
user-invocable: true
disable-model-invocation: false
---

# nbdev Export

## When to Use

- You edited a notebook under `nbs/` and need to update `src/segmentation_tools_pt/`.
- You want to regenerate the README from `nbs/index.ipynb`.
- You are unsure whether to use `nbdev_export` (old nbdev v1) or `nbdev-export` (new nbdev v2).

## Important: Run on HPC

This repo lives on a slow network-mounted drive. `nbdev-export` and git operations are much faster on a remote host (`hpc-cluster`) where the same share is mounted at:

```text
/path/to/segmentation_tools_pt
```

Do **not** run `nbdev-export` locally on the Windows-mounted path.

## Procedure

### 1. Export all notebooks to `src/`

```bash
ssh hpc-cluster 'cd /path/to/segmentation_tools_pt && uv run nbdev-export'
```

This regenerates every `.py` file in `src/segmentation_tools_pt/` from the matching notebook in `nbs/`.

### 2. Regenerate `README.md` from `nbs/index.ipynb`

```bash
ssh hpc-cluster 'cd /path/to/segmentation_tools_pt && uv run nbdev-readme'
```

> Note: in this repository `README.md` is maintained manually in addition to `nbs/index.ipynb`. After `nbdev-readme`, verify the README still reads correctly and patch it by hand if needed.

### 3. Verify the diff

```bash
ssh hpc-cluster 'cd /path/to/segmentation_tools_pt && git status --short && git diff --stat'
```

You should see:

- `nbs/<edited>.ipynb` (your edits)
- `src/segmentation_tools_pt/<module>.py` (regenerated)
- `README.md` (if you ran `nbdev-readme`)

### 4. Run tests (optional but recommended)

```bash
# Full suite
ssh hpc-cluster 'cd /path/to/segmentation_tools_pt && uv run nbdev-test'

# Scoped to just the notebook(s) you edited (faster feedback loop)
ssh hpc-cluster 'cd /path/to/segmentation_tools_pt && uv run nbdev-test --path nbs/03_patches.ipynb --do_print'
```

Do **not** run `nbdev_test` (underscore) or `python -m nbdev.test` — see the table below.

## Old vs new nbdev command

| nbdev version | Export | Test | README |
|---|---|---|---|
| v1 (legacy — NOT used here) | `nbdev_export` | `nbdev_test` | `nbdev_readme` |
| v2 (current — this project) | `nbdev-export` | `nbdev-test` | `nbdev-readme` |

This project uses **nbdev v2**, so always use the hyphenated commands, run via `uv run` so the locked dependency in `uv.lock` is used regardless of shell state (e.g. `uv run nbdev-export`, `uv run nbdev-test`, `uv run nbdev-readme`).

## What to commit

Always commit **both** the notebook and the generated files together:

```bash
ssh hpc-cluster 'cd /path/to/segmentation_tools_pt && git add nbs/ src/ README.md && git commit -m "feat(<module>): <what changed>"'
```

Never commit only `src/` changes without the corresponding `nbs/` notebook changes — the notebook is the source of truth.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nbdev-export: command not found` | Use `uv run nbdev-export` so it runs inside the project venv. |
| `nbdev_test: command not found` / `nbdev_export: command not found` | You used the legacy v1 underscored name. Use the hyphenated v2 form: `nbdev-test`, `nbdev-export`. |
| `No such file or directory: .venv/bin/activate` | The HPC venv is managed by `uv`. Use `uv run <command>` instead of `source .venv/bin/activate`. |
| Huge diff in `src/` although you only edited one notebook | A newer nbdev version changed export metadata. Commit the regeneration as its own chore commit. |
| README.md does not update | Run `nbdev-readme` separately; if still missing, edit `README.md` manually. |
