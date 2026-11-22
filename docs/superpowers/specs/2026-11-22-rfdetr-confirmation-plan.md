# Confirming RF-DETR vs the semantic baseline: plan and decision rule (#106)

Written and committed **before** the new runs finish, so the rule cannot be fitted to the results. Follows `2026-11-22-rfdetr-benchmark-results.md` (#102). Map #100.

## What was wrong with #102

The RF-DETR threshold was picked on the val set it was scored on; training budgets differed (15 vs 6 epochs); one RF-DETR run vs two baseline seeds; the baseline was not necessarily the best semantic model.

## Runs

| arm | what | seeds |
|---|---|---|
| RF-DETR Small | `scripts/rfdetr_benchmark.py train`, 15 epochs, same settings as #102 | 3 (the #102 run, plus `--seed 1`, `--seed 2`) |
| U-Net ResNet18, 15 epochs | `w0.0` config of the A/B notebook, only `train.epochs=15` | 42, 43 |
| U-Net ConvNeXt-Tiny, 15 epochs | same, `model=convnext_tiny` | 42, 43 |
| U-Net ResNet18, 6 epochs | the existing #102 baseline, kept as a reference | 42, 43 |

Same patch cache (`outputs/hierarchy_ab/patches`), same train/val split. "The best semantic model" = the semantic arm with the highest held-out score below.

## Held-out threshold

1. Split the 210 val **source images** (not patches: patches of one image stay together) into halves A and B by sorting `source_id` on `sha1(source_id)` and alternating.
2. Per model run and per direction (select on A, score on B; select on B, score on A): evaluate the threshold grid on the selection half and take the threshold with the highest **lifted object F1**, where `F1 = 2 * detected / (2 * detected + missed + false_alarms)`, summed over the half's patches (`object_metrics`, as in #102). Ties go to the higher threshold.
   Grids: RF-DETR `0.01 0.03 0.05 0.1 0.2 0.3 0.5 0.7` (confidence); semantic `0.05 0.1 0.2 0.3 0.5 0.7 0.9` (sigmoid).
3. Report on the held-out half only: lifted F1, lifted recall, lifted false alarms, lifted IoU, chip IoU. Average the two directions per run, then average over an arm's seeds; spread = half the range over seeds (or the std for 3 seeds).

## Decision rule (primary metric: held-out lifted object F1)

**Go to B (instance pipeline)** only if all hold:

- RF-DETR's mean F1 beats the best semantic arm's mean by more than `max(0.01, 2 * the larger of the two arms' seed spreads)`;
- RF-DETR's held-out chip IoU is not below the best semantic arm's by more than 0.01.

Otherwise **do not build B** (no clear gain, or a gain smaller than the noise). Secondary, not part of the rule: recall at equal false alarms and the false-alarm reduction at equal recall, read off the full curves.

## Not in scope

Polygon-vs-RLE loss on thin defects and the #101 carry-over checks (RF-DETR's RLE loader, 1-px features after resize, 16-bit input): only matter if the answer is Go.

## Amendment 1 (after a smoke test on the existing arms, before the new runs finished)

The smoke test (one RF-DETR run vs the two 6-epoch U-Nets; not the final comparison) showed the semantic arms selecting **0.9, the top of their grid**, in every direction, i.e. their false alarms were still falling when the grid ended. A grid that clips the baseline's optimum favours RF-DETR, so the semantic grid is extended to `0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.95 0.99 0.995`. Nothing else in the rule changes. If any arm's selected threshold is still the last grid point in the final analysis, the result is reported as grid-limited and the grid is extended again before the rule is applied.

## Amendment 2 (same smoke test, second look)

With the Amendment 1 grid one 6-epoch seed still selected its last point (0.995). Per Amendment 1 the grid is extended again, to `0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.95 0.99 0.995 0.999 0.9995 0.9999`, and **fixed from here on**: it is not extended again after the final runs are in. An arm whose selected threshold is the last point is then reported as grid-limited next to the verdict (sigmoid probabilities above 0.9999 are no longer distinguishable in float32 for most pixels, so a model that wants more is saturated, not under-tuned). Both amendments only help the semantic arms.

