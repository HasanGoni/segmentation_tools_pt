"""Held-out threshold comparison of RF-DETR vs semantic arms and the go/no-go rule (issue #106).

Thin wrapper around `segmentation_tools_pt.eval` (issue #109; `segpt-eval --heldout` is the general command for semantic runs).
The rule and the procedure: `docs/superpowers/specs/2026-11-22-rfdetr-confirmation-plan.md` (written before the runs finished).
Its self-test now lives as `#| test` cells in `nbs/17_eval.ipynb` (`nbdev-test --path nbs/17_eval.ipynb`).
Inputs are `per_patch.json` (written by `rfdetr_benchmark.py eval`) and `<out>_per_patch.json` (written next to `--out` by
`semantic_threshold_sweep.py` or by `segpt-eval` as `<run>_per_patch.json`).

    python scripts/heldout_compare.py --metadata outputs/hierarchy_ab/patches/metadata.json --candidate rfdetr \\
        --arm rfdetr=a/per_patch.json,b/per_patch.json --arm unet_r18_e15=c/per_patch.json,d/per_patch.json --out heldout.json
"""
import argparse, json
from pathlib import Path

from segmentation_tools_pt.eval import arm_summary, decide, held_out, run_summary, split_halves



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--metadata', required=True, help='patch cache metadata.json (val entries carry source_id)')
    p.add_argument('--arm', action='append', default=[], help='NAME=per_patch1.json,per_patch2.json (one file per seed)')
    p.add_argument('--candidate', required=True, help='arm name of the instance model')
    p.add_argument('--classes', default='chip,lifted', help='class names of the arms, comma separated (#106 data: chip,lifted)')
    p.add_argument('--objects-class', default='lifted', help='class whose object F1 is selected and compared')
    p.add_argument('--iou-class', default='chip', help='class whose IoU must not be worse')
    p.add_argument('--out')
    a = p.parse_args(argv)
    classes = tuple(a.classes.split(','))
    meta = json.loads(Path(a.metadata).read_text())['val']
    stem_source = {e['sample_id']: e['source_id'] for e in meta}
    halves = split_halves(stem_source.values())
    arms, detail = {}, {}
    for spec in a.arm:
        name, files = spec.split('=', 1)
        runs = []
        for f in files.split(','):
            by_t = json.loads(Path(f).read_text())
            assert set(next(iter(by_t.values()))) == set(stem_source), f'{f}: patch ids differ from the val split in {a.metadata}'
            runs.append(run_summary(held_out(by_t, stem_source, halves, classes, a.objects_class)))
        arms[name] = arm_summary(runs, a.objects_class)
        detail[name] = runs
    verdict = decide(arms, a.candidate, a.objects_class, a.iou_class)
    res = {'n_sources': [len(halves[0]), len(halves[1])], 'arms': arms, 'runs': detail, 'verdict': verdict}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({'arms': arms, 'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
