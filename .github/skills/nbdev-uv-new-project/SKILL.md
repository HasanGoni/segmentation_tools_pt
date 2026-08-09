---
name: nbdev-uv-new-project
description: "Scaffold a new nbdev project on top of an existing uv-managed Python project. Use when starting a fresh nbdev library, running nbdev-new together with uv, or setting up notebooks-as-source with uv + nbdev v2."
argument-hint: "<project-dir>"
user-invocable: true
disable-model-invocation: false
---

# nbdev + uv New Project

## When to Use

- You want a brand-new nbdev project but manage dependencies with `uv`.
- You already ran `uv init` and have a `pyproject.toml`, then want to layer nbdev on top.
- You are unsure about the order: `uv add nbdev` → rename `pyproject.toml` → `nbdev-new` → restore dependency metadata.

## Assumptions

- `uv` and `nbdev` are available in your base environment (outside any project venv).
- You are comfortable with the terminal and with renaming files temporarily.
- The target project directory already exists or you will create it.

## Procedure

### 1. Create or enter the project directory

```bash
mkdir -p my_project
cd my_project
```

### 2. Initialize uv

If the directory does not yet have a `pyproject.toml`:

```bash
uv init
```

This creates `pyproject.toml`, `.python-version`, and a minimal project structure.

### 3. Add nbdev with uv

```bash
uv add nbdev
```

This installs nbdev into the project environment and records it in `pyproject.toml`.

### 4. Activate the project environment

On Linux / macOS / HPC:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

> Make sure you are now using the venv's nbdev. Run `which nbdev-new` (or `Get-Command nbdev-new` on Windows) to confirm.

### 5. Rename the existing `pyproject.toml`

`nbdev-new` writes its own `pyproject.toml` and fails if one already exists. Temporarily rename the uv-generated file:

```bash
mv pyproject.toml pyproject.toml.uv_backup
```

On Windows:

```powershell
Rename-Item pyproject.toml pyproject.toml.uv_backup
```

> Keep `uv.lock` in place; it will be regenerated later.

### 6. Run `nbdev-new`

First, auto-detect sensible defaults from your git or system config so you do not have to type them every time:

```bash
NB_USER="$(git config user.name 2>/dev/null || echo "$USER")"
NB_EMAIL="$(git config user.email 2>/dev/null || echo "${USER}@example.com")"
NB_PY="$(cat .python-version 2>/dev/null || python3 --version | cut -d' ' -f2 | cut -d. -f1,2)"
```

Then run `nbdev-new` and paste the values:

```bash
nbdev-new
```

| Prompt | Value to paste |
|---|---|
| User name | `$NB_USER` (system `git config user.name` or `$USER`) |
| User email | `$NB_EMAIL` (system `git config user.email` or `$USER@example.com`) |
| Python version | `$NB_PY` (from `.python-version` or current Python) |

> For throwaway tests, accept the defaults or use `test` / `test@example.com`.

This creates:

```text
nbs/
  index.ipynb
  nbdev.yml
  _quarto.yml
  styles.css
.pytest.ini (or equivalent)
.github/
  workflows/
    deploy.yaml
    test.yaml
.gitignore
LICENSE (optional)
```

and a fresh `pyproject.toml` configured for nbdev.

### 7. Merge the uv dependencies back into the new `pyproject.toml`

The new `pyproject.toml` does not know about the packages uv installed earlier (e.g. `nbdev`). You have two options:

#### Option A — let uv re-add everything (recommended)

```bash
uv add nbdev
```

This adds nbdev to the nbdev-generated `pyproject.toml` and updates `uv.lock`.

#### Option B — manually copy dependency metadata

Open `pyproject.toml.uv_backup` and copy the relevant dependency entries (usually under `[project] dependencies` or `[tool.uv]`) into the new `pyproject.toml`. Then run:

```bash
uv lock
uv sync
```

### 8. Clean up the backup

After confirming the project works:

```bash
rm pyproject.toml.uv_backup
```

On Windows:

```powershell
Remove-Item pyproject.toml.uv_backup
```

### 9. Verify the setup

```bash
nbdev-export
nbdev-test
nbdev-readme
```

All three should run without errors. `nbdev-export` regenerates `src/` from `nbs/`, `nbdev-test` runs notebook tests, and `nbdev-readme` generates `README.md` from `nbs/index.ipynb`.

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `nbdev-new` fails because `pyproject.toml` already exists | You skipped step 5. | Rename the file and rerun `nbdev-new`. |
| `nbdev-new` not found | You are not in the activated venv. | Activate `.venv` and confirm with `which nbdev-new`. |
| `nbdev-export` cannot import the package | `uv add nbdev` was not run after `nbdev-new`. | Run `uv add nbdev` and `uv sync`. |
| `uv.lock` conflicts | The lock file was created with the old `pyproject.toml`. | Delete `uv.lock` and run `uv lock` again. |
| Git shows `pyproject.toml.uv_backup` as untracked | Backup file was not removed. | Delete it or add it to `.gitignore`. |

## Quality Checks

Before considering the project ready:

- [ ] `pyproject.toml` contains both nbdev config (`[tool.nbdev]`) and uv config.
- [ ] `uv lock` completes without errors.
- [ ] `nbdev-export` regenerates `src/`.
- [ ] `nbdev-test` passes.
- [ ] No `.uv_backup` files remain.

## Variations

### Fully automated, non-interactive defaults

If you want to skip the prompts, set environment variables before `nbdev-new` (when nbdev supports them) or pipe answers:

```bash
printf "test\ntest@example.com\n3.12\n" | nbdev-new
```

### Existing code you want to convert to nbdev

If you already have a `src/` package, run `uv add nbdev`, rename `pyproject.toml`, run `nbdev-new`, then move your source files into `nbs/` as notebook cells and export them back to `src/`.

## Example Prompts to Try

- "Create a new nbdev+uv project in `my_new_lib`."
- "Set up nbdev-new on top of an existing uv project."
- "Fix nbdev-new failing because pyproject.toml already exists."
