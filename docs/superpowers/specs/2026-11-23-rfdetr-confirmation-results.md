# RF-DETR vs semantic baseline: confirmation result (#106)

Rule fixed beforehand in `2026-11-22-rfdetr-confirmation-plan.md` (incl. Amendments 1-2). Map #100. Raw numbers: `data/2026-11-23-rfdetr-confirmation-heldout.json`.

## Verdict: NO-GO. Do not build the instance pipeline (option B).

The best semantic arm (U-Net ConvNeXt-Tiny, 15 epochs) beats RF-DETR on held-out lifted object F1 by 0.032, the opposite direction to the rule's required gain for RF-DETR (> 0.025).

## Held-out results (threshold chosen on one half of the 105+105 val source images, scored on the other, both directions; mean over seeds)

| arm | seeds | lifted F1 (± spread) | lifted recall | lifted false alarms / half | lifted IoU | chip IoU* |
|---|---|---|---|---|---|---|
| RF-DETR Small, 15 ep | 3 | 0.936 ± 0.012 | 0.901 | 6.2 | 0.931 | 0.907 |
| U-Net ResNet18, 15 ep | 2 | 0.950 ± 0.001 | 0.948 | 12.3 | 0.934 | 0.414* |
| **U-Net ConvNeXt-Tiny, 15 ep** | 2 | **0.968 ± 0.002** | 0.959 | 5.5 | 0.958 | 0.777* |
| U-Net ResNet18, 6 ep (#102 baseline) | 2 | 0.840 ± 0.011 | 0.922 | 72.0 | 0.905 | 0.871 |

Rule: gain = 0.936 - 0.968 = -0.032; required > max(0.01, 2 x 0.012) = 0.024. **Not met.**

## What changed since #102

- Equal training budget removes RF-DETR's apparent edge. The #102 baseline (6 epochs) was undertrained: 15 epochs cuts its false alarms from 72 to 12 (ResNet18) or 5.5 (ConvNeXt-Tiny) per half.
- Same budget plus a larger encoder: ConvNeXt-Tiny is ahead of RF-DETR on F1, recall and lifted IoU, with fewer false alarms.
- RF-DETR seeds spread by ±0.012 F1 (0.919 / 0.939 / 0.949), so a single run (#102) was a weak basis.

## Caveats

- **Grid-limited arms:** all four semantic runs of the 15-epoch arms selected the last grid points (0.99-0.9999; ResNet18 s42 and ConvNeXt s42/s43 picked 0.9999). Per Amendment 2 the grid is not extended again. This can only understate the semantic arms, which already win, so the verdict is conservative for RF-DETR's rival.
- *Chip IoU of the 15-epoch U-Nets is low (0.41, 0.78) only because one shared threshold is chosen for lifted F1 and a threshold of 0.9999 cuts the chip class. It is not a model-quality number; the chip check passed because RF-DETR is higher, and it does not change the verdict. A per-class threshold was not part of the pre-registered rule.
- RF-DETR seed 0 (the #102 run) was unseeded and was scored with the same 8-point grid as seeds 1-2; one RF-DETR size (Small) only, no larger RF-DETR tried.
- The first launch of RF-DETR seeds 1 and 2 was killed by SIGTERM (no checkpoint); they were re-run from scratch, 15 epochs each.
- Still unmeasured, and moot given no-go: polygon-vs-RLE loss, RF-DETR's RLE loader, 1-px features, 16-bit input. #102 stays open for them only if B is ever revisited.
- Instance separation never mattered on this data (large, well-separated regions), which is where RF-DETR would help.

## Reproduce

```bash
python scripts/semantic_threshold_sweep.py outputs/confirm_106/RUN --out RUN.json     # per-patch file written next to it
PYTHONPATH=src python scripts/rfdetr_benchmark.py train --cache outputs/hierarchy_ab/patches --out RUN --model small --epochs 15 --seed N
PYTHONPATH=src python scripts/rfdetr_benchmark.py eval  --cache outputs/hierarchy_ab/patches --out RUN --thresholds 0.01 0.03 0.05 0.1 0.2 0.3 0.5 0.7 --primary 0.05
python scripts/heldout_compare.py --metadata outputs/hierarchy_ab/patches/metadata.json --candidate rfdetr --arm rfdetr=a,b,c --arm unet=d,e --out heldout.json
```
