---
name: continuous-integration
description: 'PLACEHOLDER skill — continuous integration for segmentation_tools_pt (nbdev tests, lint, docs build) is NOT set up yet. Use when a user asks "CI", "GitLab CI", "GitHub Actions", "run tests automatically", or "nbdev_test in CI". Explains what a CI pipeline would do and where it would live.'
argument-hint: '[optional] CI platform'
user-invocable: true
disable-model-invocation: false
---
# Continuous Integration — NOT IMPLEMENTED YET

An automated CI pipeline (run tests, lint, build docs) is **not configured** for
`segmentation_tools_pt` yet.

## What a CI pipeline should do

1. `uv sync` to install dependencies.
2. `nbdev_test` to execute in-notebook tests.
3. `nbdev_export` and verify `src/` is in sync with `nbs/`.
4. `nbdev_readme` / `nbdev_docs` to ensure docs build.
5. Optional: `ruff` lint and `mypy` type check on `src/`.

## Where it would live

- GitLab CI: `.gitlab-ci.yml`
- GitHub Actions: `.github/workflows/ci.yml`

## Suggested GitLab CI skeleton

```yaml
stages: [test]

test:
  stage: test
  image: python:3.12
  script:
    - pip install uv
    - uv sync
    - uv run nbdev_test
    - uv run nbdev_export
    - git diff --exit-code src/  # fail if src/ is out of sync
```

## Status

**Not implemented yet.** This skill exists to set expectations and route the user, per the
"placeholder if not in the library" convention.
