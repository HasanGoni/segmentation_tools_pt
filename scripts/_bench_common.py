"""Scoring shared by `rfdetr_benchmark.py`, `semantic_threshold_sweep.py` and `heldout_compare.py` (issues #102, #106), so every model is measured by one code path."""
import hashlib

import numpy as np

from segmentation_tools_pt.metrics import object_metrics

CLASSES = ('chip', 'lifted')


def iou(pred, gt):
    "IoU of two boolean masks; two empty masks score 1.0."
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union else 1.0


def patch_scores(pred, gt, classes=CLASSES):
    "One patch's row: IoU plus connected-component object counts (`object_metrics`) per class. `pred`, `gt` are `(K, H, W)` boolean."
    row = {}
    for k, name in enumerate(classes):
        o = object_metrics(pred[k].astype(np.uint8), gt[k].astype(np.uint8))
        row[f'iou_{name}'] = iou(pred[k], gt[k])
        row[f'{name}_total'] = o['total_gt']
        row[f'{name}_detected'] = o['detected']
        row[f'{name}_false_alarms'] = o['false_alarms']
    return row


def aggregate(rows, classes=CLASSES):
    "Mean IoU and summed object counts over patch rows, plus recall per class."
    rows = list(rows)
    out = {}
    for name in classes:
        total = sum(r[f'{name}_total'] for r in rows)
        detected = sum(r[f'{name}_detected'] for r in rows)
        out[f'iou_{name}'] = float(np.mean([r[f'iou_{name}'] for r in rows]))
        out[f'{name}_total'] = total
        out[f'{name}_detected'] = detected
        out[f'{name}_recall'] = detected / max(total, 1)
        out[f'{name}_false_alarms'] = sum(r[f'{name}_false_alarms'] for r in rows)
    return out


def object_f1(detected, total, false_alarms):
    "Object-level F1 = 2*detected / (2*detected + missed + false alarms); 1.0 when there is nothing to find or flag."
    denom = 2 * detected + (total - detected) + false_alarms
    return 2 * detected / denom if denom else 1.0


def ids_sha1(ids):
    "Fingerprint of a set of patch ids, to check two scripts scored the same val patches."
    return hashlib.sha1('\n'.join(sorted(str(i) for i in ids)).encode()).hexdigest()
