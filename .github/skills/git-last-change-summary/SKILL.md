---
name: git-last-change-summary
description: 'Summarize latest git changes, update CHANGELOG.md, and draft a commit message. Use for release notes, handoff summaries, and pre-commit documentation.'
argument-hint: '[optional] scope or commit SHA; default analyzes working tree, fallback to latest commit'
user-invocable: true
disable-model-invocation: false
---
# Git Last Change Summary

Create a concise, traceable change summary from git, write it into CHANGELOG.md, and produce a high-quality commit message.

## When To Use

- You want a quick summary of what changed in the repository.
- You need to append a changelog entry before committing.
- You need a commit message that matches the actual diff.
- You are preparing a status update or handoff.

## Inputs

- Optional scope argument.
- If scope is omitted:

1. Analyze uncommitted changes first.
2. If working tree is clean, analyze the latest commit (`HEAD`).

## Procedure

1. Determine source of truth:

- Run `git status --short`.
- If there are modified files, collect staged and unstaged diffs.
- If no local changes, inspect latest commit with `git show --name-status --stat --oneline -1`.

2. Build structured summary:

- Group changes by file and intent:

1. Added
2. Updated
3. Removed
4. Refactor/cleanup

- Extract key behavioral impact:

1. Feature behavior changes
2. Pipeline/inference changes
3. Config/data path changes
4. Docs/tests updates
5. Update CHANGELOG.md:

- Append under a new dated heading: `## YYYY-MM-DD`.
- Add a short context line, then flat bullet points describing concrete changes.
- Keep wording factual and tied to files/diff.

4. Draft commit message:

- Subject line format: `<type>(<scope>): <summary>`.
- Candidate types: `feat`, `fix`, `refactor`, `docs`, `chore`.
- Add body bullets for major points when change set spans multiple files.
- Include one explicit line for user-facing or pipeline-impacting behavior.
- Always provide a terminal-ready copy/paste command using repeated `-m` flags, for example:
  `git commit -m "<type>(<scope>): <summary>" -m "<bullet 1>" -m "<bullet 2>" -m "Pipeline impact: <impact>."`

5. Final validation:

- Ensure summary matches actual changed files.
- Ensure CHANGELOG.md was appended, not rewritten.
- Ensure commit message maps to strongest change intent.
- Ensure no sensitive data appears in changelog or commit text.

6. Commit, and push if authorized:

- Run the commit for the user rather than only printing it, unless they ask to review first.
- Default to NOT pushing. Only run `git push` when the user has explicitly said to push (this
  time, or as a standing preference stated earlier in the conversation/session) — pushing
  shares a commit outside the local repo and is harder to reverse than a local commit.
- If unsure whether push is authorized, ask once rather than assuming either way.
- Any environment-specific speed tricks (e.g. running git over SSH on a faster remote checkout
  instead of a slow local mount) are a personal workflow choice, not something to hardcode into
  this skill — keep those in your own persistent notes/config, since other users of this skill
  won't have the same remote host set up.

## Decision Logic

- If code + docs changed, prioritize code intent in commit type and list docs in body.
- If only docs changed, use `docs` type.
- If mostly renaming/restructure without behavior changes, use `refactor` or `chore`.
- If thresholds/configs alter runtime outputs, treat as behavior-impacting and call it out.

## Completion Criteria

- Changelog entry exists for current run date with concrete bullets.
- Summary captures all primary changed files.
- Commit message subject is one line and under 72 chars.
- Commit body (if present) is concise and consistent with diff.
- The commit was actually created (not just printed), unless the user asked to review first.
- The commit was pushed only if push was explicitly authorized; otherwise it was left local.

## Output Format

1. Change summary (3-8 bullets)
2. Changelog text added
3. Commit message used
4. Confirmation the commit was created (SHA + branch), and whether it was pushed
