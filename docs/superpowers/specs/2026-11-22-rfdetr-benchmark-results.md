# RF-DETR seg vs the semantic baseline: benchmark result (#102)

Part of map #100. Research: `2026-11-22-rfdetr-seg-integration-research.md`.

## Verdict

**Not a clean go and not a no-go. Do not build the instance pipeline (option B) yet; run one confirmation step first.**

The rule in map #100 says: go to B if RF-DETR beats the best semantic model on `detection_rate` at equal false alarms on the same val patches.

- At the threshold fixed **before** looking (score 0.5, the baseline's own 0.5): RF-DETR loses. Lifted recall 0.811 vs 0.925.
- After lowering the threshold (chosen on the val set itself): at about 175 false alarms RF-DETR reaches lifted recall **0.936** vs the baseline's **0.925** (+0.011; the two baseline seeds differ by 0.006-0.007). The cost: lifted IoU drops to 0.815 at that point (baseline 0.903). So the rule is met only at a post-hoc operating point, marginally, and not for free.
- The larger practical difference is false alarms: at threshold 0.05 RF-DETR has **59** lifted false alarms at recall 0.917 and lifted IoU 0.904; the baseline never goes below about 150 at any threshold it was tried at (0.05-0.9; recall 0.923). That is a different operating point from the rule's "equal false alarms", not a win under it.
- Chip IoU: 0.900 at threshold 0.05 vs 0.864-0.870, but at the fixed 0.5 it is a tie (0.872 vs 0.869). Do not quote the 0.900 without the threshold.

Why this is not enough to commit to B:

1. The RF-DETR threshold was picked on the same val set it is scored on (an 8-point sweep). The baseline was swept over 7 points too, but its curve is flat, so it gains nothing from tuning; RF-DETR's curve is steep, so selection on val flatters it.
2. The training budgets differ: RF-DETR Small (33.4M parameters, 15 epochs, 43 min) vs the U-Net (ResNet18, 6 epochs).
3. Two baseline seeds, one RF-DETR run; "± half the seed range" from two seeds is a weak error bar.
4. The baseline is the `w0.0` arm of the A/B notebook (ResNet18, 6 epochs), the run that existed, not necessarily the best semantic model. No DINOv3 or larger encoder was tried, which is what map #100's rule names.
5. 529 lifted objects in val; +0.011 recall is about 6 objects.

**Confirmation step (one follow-up ticket):** pick the RF-DETR threshold on one half of val (split by `source_id`), score on the other half; train the U-Net for the same 15 epochs (or a larger encoder); 2-3 RF-DETR seeds. If RF-DETR still wins on recall at equal false alarms, or keeps its false-alarm advantage at equal recall, go to B.

## Setup

- Data: `outputs/hierarchy_ab/patches` (512 px patches, 2 overlapping classes `chip`, `lifted`; 3294 train / 821 val). RF-DETR trained on the same train patches, scored on the same val patches with the same `object_metrics` and IoU code.
- RF-DETR: `RFDETRSegSmall(num_channels=1)`, 15 epochs, batch 8, lr 1e-4, pretrained weights, scored from `last_ema.pth` (`checkpoint_best_total.pth` is not used, see the research report). COCO folder from `write_coco_folder` (`layout='per_class'`, `segmentation='auto'`; 4 of 15,912 annotations are RLE).
- Baseline: `w0.0` arm of `tutorials/07_hierarchy_ab_real_data.ipynb`, seeds 42 and 43 (U-Net ResNet18, 6 epochs), scored by `scripts/semantic_threshold_sweep.py` with one shared threshold for both classes. At 0.5 it reproduces the notebook exactly (IoU 0.869 / 0.903, recall 0.925, 176 false alarms).
- Thresholds are not on the same scale: RF-DETR's is a confidence score (`>=`, applied by `detections_to_semantic(threshold=)`), the baseline's a sigmoid probability (`>`). "Both 0.5" is nominal; the sweeps, not the 0.5 row, carry the comparison.
- Same val patches: both scripts write a fingerprint of their val ids (`val_ids_sha1`, 821 ids) and they are identical (`d761c5a0cb...`); the baseline report refuses `--ids` that are not in its val split.
- Metrics: IoU per class (mean over patches); `object_metrics` on `lifted` summed over val (recall = detected / 529; false alarms = predicted components not overlapping GT).

## Results (full val, 821 patches)

RF-DETR Small, `last_ema.pth`, score threshold swept:

| threshold | IoU chip | IoU lifted | lifted recall | lifted false alarms | chip recall |
|---|---|---|---|---|---|
| 0.03 *(post-hoc)* | 0.889 | 0.815 | 0.936 | 175 | 0.983 |
| 0.05 *(post-hoc)* | 0.900 | 0.904 | 0.917 | 59 | 0.975 |
| 0.1 *(post-hoc)* | 0.907 | 0.923 | 0.894 | 22 | 0.959 |
| 0.2 *(post-hoc)* | 0.902 | 0.916 | 0.866 | 6 | 0.945 |
| 0.3 | 0.892 | 0.903 | 0.841 | 4 | 0.933 |
| **0.5 (fixed in advance)** | 0.872 | 0.888 | 0.811 | 4 | 0.909 |
| 0.7 | 0.847 | 0.871 | 0.771 | 4 | 0.879 |

(0.01 gives recall 0.981 but 1448 false alarms and lifted IoU 0.369.)

Baseline, mean of 2 seeds (± half the seed range):

| threshold | IoU chip | IoU lifted | lifted recall | lifted false alarms | chip recall |
|---|---|---|---|---|---|
| 0.05 | 0.864 | 0.889 | 0.926 ± 0.006 | 250 ± 25 | 0.932 |
| 0.1 | 0.866 | 0.896 | 0.926 ± 0.006 | 210 ± 3 | 0.931 |
| 0.3 | 0.868 | 0.901 | 0.925 ± 0.007 | 189 ± 8 | 0.931 |
| **0.5** | 0.869 | 0.903 | 0.925 ± 0.007 | 176 ± 10 | 0.930 |
| 0.7 | 0.869 | 0.904 | 0.924 ± 0.006 | 172 ± 14 | 0.929 |
| 0.9 | 0.870 | 0.906 | 0.923 ± 0.005 | 154 ± 11 | 0.929 |

At matching false-alarm levels: about 175 false alarms, RF-DETR 0.936 vs baseline 0.925 / 0.924 (thresholds 0.5 / 0.7); about 59 false alarms the baseline has no operating point.

## Side-by-side review report

Same 80 patches in both (40 where RF-DETR did worst on `lifted`, 40 random): RF-DETR at threshold 0.05 and the baseline at 0.5 (seed 42). **Do not compare their pooled numbers**: the worst-40 are picked against RF-DETR, so the set is for looking at failures, not for scoring. The full-val tables above are the comparison.

## Findings that change the library

- **`write_coco_folder` did not work on the real cache** (per-class mask files, overlapping classes): fixed with `layout='auto'|'label_map'|'per_class'` (`4e606f9`).
- **RF-DETR class ids are 0-based on this 2-class model** (`class_names == ['chip', 'lifted']`, chip = 0, lifted = 1). The research's 1-class toy run returned id 1, so ids are model-dependent. `detections_to_semantic` crashed on class 0; fixed with `first_class_id` (`65a769b`); the research report and docs no longer assert one numbering.
- `tutorials/07_hierarchy_ab_real_data.ipynb`'s evaluation cell calls `scripts/train.py`'s `get_model` / `_model_kwargs`, which no longer exist (`train.py` now uses `get_model_from_cfg`). It was not touched here; `scripts/semantic_threshold_sweep.py` uses the library function.

## Not verified

- Mask-to-polygon loss on thin defects, polygon vs RLE (asked for in #102): not measured. The research and #101 measured it with our own decoder and `pycocotools` on synthetic masks only; on this cache only 4 of 15,912 annotations are RLE and the thinnest region is far larger than 1-3 px, so it could not matter here.
- RF-DETR's own loader on RLE (only 4 annotations here), 1-px features after its resize, 2-channel and 16-bit input.
- Chip false alarms are computed (`chip_false_alarms` in the JSON output) but not discussed.
- Whether the gain comes from RF-DETR's architecture or its larger size and longer training.
- Instance separation, the reason to use an instance model, never mattered here: the regions are large and well apart.

## Reproduce

```bash
# separate env with rfdetr[train]; the library is on PYTHONPATH
PYTHONPATH=src python scripts/rfdetr_benchmark.py train --cache outputs/hierarchy_ab/patches --out RUN --model small --epochs 15
PYTHONPATH=src python scripts/rfdetr_benchmark.py eval  --cache outputs/hierarchy_ab/patches --out RUN --thresholds 0.01 0.03 0.05 0.1 0.2 0.3 0.5 0.7 --primary 0.05 --report
# repo env
python scripts/semantic_threshold_sweep.py outputs/hierarchy_ab/w0.0_s43 --out sweep_s43.json
python scripts/semantic_threshold_sweep.py outputs/hierarchy_ab/w0.0_s42 --out sweep_s42.json \
    --ids RUN/sample_ids.json --report-dir RUN/report_baseline --report-threshold 0.5
```
