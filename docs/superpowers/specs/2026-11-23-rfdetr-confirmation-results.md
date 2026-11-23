# RF-DETR vs semantic baseline: confirmation result (#106)

Rule fixed beforehand in `2026-11-22-rfdetr-confirmation-plan.md` (incl. Amendments 1-2). Map #100. Raw numbers: `data/2026-11-23-rfdetr-confirmation-heldout.json`.

## Verdict: NO-GO. Do not build the instance pipeline (option B).

The best semantic arm (U-Net ConvNeXt-Tiny, 15 epochs) beats RF-DETR on held-out lifted object F1 by 0.032. RF-DETR would have needed to be ahead by more than 0.025 to get a go.

## Held-out results (threshold chosen on one half of the 105+105 val source images, scored on the other, both directions; mean over seeds)

| arm | seeds | lifted F1 (± spread) | lifted recall | lifted false alarms / half | lifted IoU | chip IoU* |
|---|---|---|---|---|---|---|
| RF-DETR Small, 15 ep | 3 | 0.936 ± 0.012 | 0.901 | 6.2 | 0.931 | 0.907 |
| U-Net ResNet18, 15 ep | 2 | 0.950 ± 0.001 | 0.948 | 12.3 | 0.934 | 0.414* |
| **U-Net ConvNeXt-Tiny, 15 ep** | 2 | **0.968 ± 0.002** | 0.959 | 5.5 | 0.958 | 0.777* |
| U-Net ResNet18, 6 ep (#102 baseline) | 2 | 0.840 ± 0.011 | 0.922 | 72.0 | 0.905 | 0.871 |

Rule: gain = 0.936 - 0.968 = -0.032; required > max(0.01, 2 x 0.0124) = 0.025. **Not met.**

## What changed since #102

- Equal training budget removes RF-DETR's apparent edge. The #102 baseline (6 epochs) was undertrained: 15 epochs cuts its false alarms from 72 to 12 (ResNet18) or 5.5 (ConvNeXt-Tiny) per half.
- Same budget plus a larger encoder: ConvNeXt-Tiny is ahead of RF-DETR on F1, recall and lifted IoU, with about the same false alarms (5.5 vs 6.2 per half, within seed noise).
- RF-DETR seeds spread by ±0.012 F1 (0.919 / 0.939 / 0.949), so a single run (#102) was a weak basis.

## Caveats

- **Grid-limited arms:** in the 15-epoch arms, 5 of the 8 per-direction selections were the last grid point (0.9999: ResNet18 s42 both directions, ConvNeXt s42 both, ConvNeXt s43 one); the rest were 0.9995 or 0.99. Per Amendment 2 the grid is not extended again. This can only understate the semantic arms, which already win, so it does not threaten the no-go. The 6-epoch arm selected 0.995 / 0.9. RF-DETR never selected an edge point (0.1-0.3).
- *Chip IoU of the 15-epoch U-Nets is low (0.41, 0.78) only because one shared threshold is chosen for lifted F1 and a threshold of 0.9999 cuts the chip class. It is not a model-quality number; the chip check passed because RF-DETR is higher, and it does not change the verdict. A per-class threshold was not part of the pre-registered rule.
- RF-DETR seed 0 (the #102 run) was unseeded and was scored with the same 8-point grid as seeds 1-2; one RF-DETR size (Small) only, no larger RF-DETR tried.
- The first launch of RF-DETR seeds 1 and 2 was killed by SIGTERM (no checkpoint); they were re-run from scratch, 15 epochs each.
- Spread for 3 seeds is the population std (ddof=0) as implemented; the sample std would give about 0.015 and a required gain of about 0.030. The verdict does not change.
- The plan's secondary metrics (recall at equal false alarms, false-alarm reduction at equal recall from the full curves) are not reported here; they are not part of the rule and the primary metric already decides.
- Still unmeasured, and moot given no-go: polygon-vs-RLE loss, RF-DETR's RLE loader, 1-px features, 16-bit input. #102 stays open for them only if B is ever revisited.
- Instance separation never mattered on this data (large, well-separated regions), which is where RF-DETR would help.

## Reproduce

```bash
python scripts/semantic_threshold_sweep.py outputs/confirm_106/RUN --out F/RUN.json     # also writes F/RUN_per_patch.json
PYTHONPATH=src python scripts/rfdetr_benchmark.py train --cache outputs/hierarchy_ab/patches --out RUN --model small --epochs 15 --seed N
PYTHONPATH=src python scripts/rfdetr_benchmark.py eval  --cache outputs/hierarchy_ab/patches --out RUN --thresholds 0.01 0.03 0.05 0.1 0.2 0.3 0.5 0.7 --primary 0.05
python scripts/heldout_compare.py --metadata outputs/hierarchy_ab/patches/metadata.json --candidate rfdetr --arm rfdetr=RF0/per_patch.json,RF1/per_patch.json,RF2/per_patch.json --arm unet=F/RUN42_per_patch.json,F/RUN43_per_patch.json --out heldout.json
```
