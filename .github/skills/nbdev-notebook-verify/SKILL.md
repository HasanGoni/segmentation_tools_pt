---
name: nbdev-notebook-verify
description: >-
  Edit nbdev source notebooks (nbs/*.ipynb) with mandatory on-disk verification.
  Use when changing nbs/, 10_train.ipynb, or any nbdev notebook; when EditNotebook
  may not persist (NFS, remote mount, open editor buffer); or when the user asks
  to verify notebook changes landed on disk.
---

# nbdev notebook edit + verify

`nbs/*.ipynb` is the source of truth. `src/` is generated — never edit `src/` directly.

## Why edits disappear

- **NFS / network mount** (e.g. `//mucsdv829...`) — slow writes, stale reads, `os error 59`.
- **Open notebook in IDE** — unsaved buffer can overwrite agent writes on save.
- **EditNotebook success ≠ disk updated** — always verify from the filesystem.

## Mandatory workflow

1. **Before editing** — read the target notebook from disk (`Read` or `Grep` on the `.ipynb` path). Note which markers are already present vs missing.

2. **Prefer JSON patch for multi-cell changes** — load `json`, replace whole cell `source` arrays, `write_text` once. Use `EditNotebook` only for single small cells.

3. **After every edit batch** — run verification (step 4). Do **not** tell the user it is done until verification passes.

4. **Verify on disk** — from repo root:

```bash
python .cursor/skills/nbdev-notebook-verify/scripts/verify_notebook.py nbs/10_train.ipynb --expect MARKER ...
```

Or `Grep` the `.ipynb` file for unique strings you just added. If missing, re-apply via JSON patch — do not assume success.

5. **Report to user** — list markers found / missing. If the file is open in the editor, ask them to **reload from disk** before `nbdev_export`.

## Verification markers (10_train callbacks example)

After callback work, all of these must appear in `nbs/10_train.ipynb`:

| Marker | Meaning |
|--------|---------|
| `class TrainState` | State dataclass exported |
| `class TrainCallback` | Callback base class |
| `state: Optional[TrainState]` | `_call_callbacks` null-safe |
| `callbacks: Optional[Sequence[TrainCallback]] = None` | on `train_one_epoch` / `validate` / `train` |
| `on_train_batch_begin` | batch hook wired in loop |
| `on_train_begin` | epoch loop wired |
| `on_checkpoint` | checkpoint hook wired |
| `stop_training` | early-stop flag checked |
| `_CountBatches` | smoke test exercises callbacks |

## Patch script pattern

For large notebook updates, add a one-shot script under `scripts/` that:

1. Loads the `.ipynb` as JSON
2. Replaces entire cells by content marker (`def train_one_epoch`, etc.)
3. Writes atomically (`path.tmp` → `replace`)
4. Runs `verify_notebook.py` and exits non-zero on failure

Delete or keep the patch script per user preference; the skill matters more than keeping every patch script.

## Do not

- Run `nbdev_export` unless the user asks (they may run on HPC).
- Edit `src/segmentation_tools_pt/*.py` when the change belongs in `nbs/`.
- Trust the Read tool's notebook cell view without a fresh `Grep`/`verify_notebook.py` after writes on NFS paths.

## Utility

```bash
python .cursor/skills/nbdev-notebook-verify/scripts/verify_notebook.py nbs/10_train.ipynb \
  --expect "class TrainState" --expect "on_train_begin" --expect "_CountBatches"
```

Exit `0` = all markers present; non-zero = list missing markers.
