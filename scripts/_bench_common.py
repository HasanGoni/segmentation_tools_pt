"""Scoring shared by `rfdetr_benchmark.py`, `semantic_threshold_sweep.py` and `heldout_compare.py` (issues #102, #106).

The code lives in the library (`nbs/17_eval.ipynb` -> `segmentation_tools_pt.eval`, issue #109); this module only fixes the
chip/lifted class tuple those scripts were written for, so every model is still measured by one code path.
"""
from segmentation_tools_pt.eval import ids_sha1, iou, object_f1  # noqa: F401  (re-exported)
from segmentation_tools_pt.eval import aggregate as _aggregate, patch_scores as _patch_scores

CLASSES = ('chip', 'lifted')


def patch_scores(pred, gt, classes=CLASSES):
    "`segmentation_tools_pt.eval.patch_scores` with the chip/lifted default."
    return _patch_scores(pred, gt, classes)


def aggregate(rows, classes=CLASSES):
    "`segmentation_tools_pt.eval.aggregate` with the chip/lifted default."
    return _aggregate(rows, classes)
