"""Score a finished semantic run (`best.pth`) on the val split at several probability thresholds (issue #102).

Thin wrapper around `segmentation_tools_pt.eval.sweep_run` (issue #109); `segpt-eval RUN` is the general command. This script
keeps its original output files (`--out` JSON plus `<out>_per_patch.json`) so `heldout_compare.py` and older notes still work.

    python scripts/semantic_threshold_sweep.py outputs/hierarchy_ab/w0.0_s42 --out sweep_w0.0_s42.json
"""
import argparse, json
from pathlib import Path

import numpy as np

from segmentation_tools_pt.eval import DEFAULT_THRESHOLDS, sweep_run


def sweep(run_dir, thresholds, report_ids=None, report_threshold=0.5, device=None):
    "`(by_threshold, keep, val_ids_sha1, per_patch)` for one finished run; see `sweep_run`."
    by_t, keep, sha, per_patch, _ = sweep_run(run_dir, thresholds, report_ids, report_threshold, device)
    return by_t, keep, sha, per_patch


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--out', required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=list(DEFAULT_THRESHOLDS))
    p.add_argument('--ids', default=None, help='sample_ids.json written by rfdetr_benchmark.py eval --report: also build a review report on those patches')
    p.add_argument('--report-dir', default=None)
    p.add_argument('--report-id', default='semantic-baseline')
    p.add_argument('--report-threshold', type=float, default=0.5)
    a = p.parse_args()
    ids = set(json.loads(Path(a.ids).read_text())) if a.ids else None
    by_t, keep, val_sha1, per_patch, info = sweep_run(a.run_dir, a.thresholds, ids, a.report_threshold)
    if ids is not None:
        assert len(keep) == len(ids), f'{len(ids) - len(keep)} of the {len(ids)} --ids are not in this run\'s val split'
        from segmentation_tools_pt.review_report import Sample, build_report
        samples = [Sample(s, keep[s][0], keep[s][1].astype(np.uint8), keep[s][2].astype(np.uint8), class_names=info['classes'])
                   for s in json.loads(Path(a.ids).read_text()) if s in keep]
        print('report:', build_report(a.report_id, a.report_dir, samples), f'({len(samples)} of {len(ids)} samples)')
    Path(a.out).with_name(Path(a.out).stem + '_per_patch.json').write_text(json.dumps(per_patch))
    Path(a.out).write_text(json.dumps({'run_dir': a.run_dir, 'val_ids_sha1': val_sha1, 'thresholds': by_t}, indent=1))
    print(json.dumps(by_t, indent=1))
