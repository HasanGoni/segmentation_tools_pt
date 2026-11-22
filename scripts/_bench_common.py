"""Scoring shared by `rfdetr_benchmark.py` and `semantic_threshold_sweep.py` (issue #102), so both models are measured by one code path."""
import hashlib

import numpy as np

from segmentation_tools_pt.metrics import object_metrics

CLASSES = ('chip', 'lifted')


def iou(pred, gt):
    "IoU of two boolean masks; two empty masks score 1.0."
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union else 1.0


def new_tally():
    return dict(total=0, detected=0, false_alarms=0)


def add_objects(tally, pred, gt):
    "Add one patch's connected-component detection stats (`object_metrics`) to `tally`."
    o = object_metrics(pred.astype(np.uint8), gt.astype(np.uint8))
    tally['total'] += o['total_gt']; tally['detected'] += o['detected']; tally['false_alarms'] += o['false_alarms']


def summarise(ious, tallies, classes=CLASSES):
    "One result row: mean IoU per class, object recall and false alarms per class. `ious` and `tallies` are per-class lists."
    row = {}
    for k, name in enumerate(classes):
        row[f'iou_{name}'] = float(np.mean(ious[k]))
        row[f'{name}_total'] = tallies[k]['total']
        row[f'{name}_detected'] = tallies[k]['detected']
        row[f'{name}_recall'] = tallies[k]['detected'] / max(tallies[k]['total'], 1)
        row[f'{name}_false_alarms'] = tallies[k]['false_alarms']
    return row


def ids_sha1(ids):
    "Fingerprint of a set of patch ids, to check two scripts scored the same val patches."
    return hashlib.sha1('\n'.join(sorted(str(i) for i in ids)).encode()).hexdigest()
