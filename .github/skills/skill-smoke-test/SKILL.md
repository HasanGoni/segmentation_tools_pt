---
name: skill-smoke-test
description: 'Test whether a Copilot SKILL.md is well-formed and discoverable. Use when a user asks "is my skill working?", "test a skill", "skill smoke test", "validate SKILL.md", or "debug why my skill is not loaded". Runs structural checks and emits chat prompts to verify behavior manually.'
argument-hint: '<skill-name-or-folder>'
user-invocable: true
disable-model-invocation: false
---

# Skill Smoke Test

## Before Creating a New Skill

Search existing skills for overlapping scope **before** writing a new `SKILL.md`:

```bash
grep -ril "<topic keyword>" .github/skills/*/SKILL.md
```

Also scan the skill name/description list itself (it is fully loaded into every agent's
context). If an existing skill already covers the topic — even partially — **extend that
skill's SKILL.md** instead of creating a new folder. Splitting one topic across two skills
means they drift out of sync and one of them has to be merged/deleted later (wasted work for
whoever notices first). Example: nbdev test-command guidance belongs inside `nbdev-export`,
not a separate `nbdev-test` skill, since both are steps in the same notebook → package
workflow.

## When to Use

- You just created or edited a `SKILL.md` under `.github/skills/<name>/` and want to know if it will be discovered.
- Copilot is not invoking your skill for queries you think should match.
- You want a fast, repeatable sanity check before committing a new skill.

## What It Checks

This skill points to a reusable notebook at [`./notebooks/test_skills.ipynb`](./notebooks/test_skills.ipynb). When you run it, it checks every `SKILL.md` in `.github/skills/` for the structural problems that most often cause silent failures:

| Check | Why it matters |
|---|---|
| No overlapping existing skill | Two skills covering the same topic drift out of sync; extend the existing one instead. |
| YAML frontmatter parses cleanly | A colon inside an unquoted value or a tab indent breaks the loader silently. |
| `name` matches the folder name | Required by the skill loader; mismatch = invisible skill. |
| `description` exists and is keyword-rich | The description is the discovery surface. If the trigger words are not in it, the model will never load the skill. |
| `user-invocable` / `disable-model-invocation` are consistent | Avoids a skill that is neither slash-invokable nor auto-loadable. |
| Referenced `./scripts/`, `./assets/`, or `./notebooks/` paths exist | Prevents broken relative links. |
| Placeholder skills explicitly say "NOT implemented" | Keeps honesty consistent across the skill library. |

## Procedure

1. Open [`notebooks/test_skills.ipynb`](./notebooks/test_skills.ipynb).
2. Run all cells.
3. Read the report:
   - **PASS / FAIL** per skill
   - A list of broken references
   - Suggested manual test prompts for each skill
4. Pick one of the generated prompts, paste it into Copilot Chat, and confirm the skill loads and follows its procedure.
5. Fix any FAIL items, re-run the notebook, then commit.

## Manual Smoke Test

If you cannot run the notebook right now, copy this prompt into Copilot Chat (replace `<skill-name>`):

```text
Invoke the <skill-name> skill and follow its procedure for a toy example.
```

If the skill is `user-invocable: true`, you can also type:

```text
/<skill-name>
```

If the skill does not load, check that its `description` contains the exact words you used in your query.

## Notes

- This validates structure, not model dispatch. A structurally perfect skill can still fail to load if the model decides another skill or the default agent is a better match.
- For full behavior verification, always perform the manual chat test from step 4.
