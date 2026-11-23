"""Held-out threshold comparison of RF-DETR vs semantic arms and the go/no-go rule (issue #106).

Rule and procedure: `docs/superpowers/specs/2026-11-22-rfdetr-confirmation-plan.md` (written before the runs finished).
Inputs are `per_patch.json` (written by `rfdetr_benchmark.py eval`) and `<out>_per_patch.json` (written next to `--out` by `semantic_threshold_sweep.py`).

    python scripts/heldout_compare.py --metadata outputs/hierarchy_ab/patches/metadata.json --candidate rfdetr \\
        --arm rfdetr=a/per_patch.json,b/per_patch.json --arm unet_r18_e15=c/per_patch.json,d/per_patch.json --out heldout.json
    python scripts/heldout_compare.py --selftest
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np

from _bench_common import aggregate, object_f1

MIN_GAIN = 0.01        # F1 gain RF-DETR needs over the best semantic arm, on top of the seed noise
NOISE_FACTOR = 2       # ... which must also exceed this many seed spreads
CHIP_MARGIN = 0.01     # RF-DETR's chip IoU may not trail the best semantic arm by more than this


def split_halves(source_ids):
    "Two disjoint, deterministic halves of the source images: sort by sha1, alternate."
    ordered = sorted(set(source_ids), key=lambda s: hashlib.sha1(str(s).encode()).hexdigest())
    return set(ordered[0::2]), set(ordered[1::2])


def _thresholds(by_threshold):
    return sorted(by_threshold, key=float)


def held_out(by_threshold, stem_source, halves, cls='lifted'):
    """Select the threshold on one half (best `cls` object F1, ties to the higher threshold), score the other, both ways.

    `by_threshold` is `{threshold: {patch_id: row}}` as written by the eval scripts. Returns one dict per direction.
    """
    out = []
    for sel, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
        best_t, best_f1 = None, -1.0
        for t in _thresholds(by_threshold):
            rows = [r for s, r in by_threshold[t].items() if stem_source[s] in sel]
            a = aggregate(rows)
            f1 = object_f1(a[f'{cls}_detected'], a[f'{cls}_total'], a[f'{cls}_false_alarms'])
            if f1 >= best_f1:
                best_t, best_f1 = t, f1
        a = aggregate([r for s, r in by_threshold[best_t].items() if stem_source[s] in ev])
        a['f1_lifted'] = object_f1(a['lifted_detected'], a['lifted_total'], a['lifted_false_alarms'])
        a['threshold'] = float(best_t); a['selection_f1'] = best_f1
        out.append(a)
    return out


def run_summary(directions):
    "Mean of the two directions for one run."
    keys = [k for k in directions[0] if k not in ('threshold',)]
    s = {k: float(np.mean([d[k] for d in directions])) for k in keys}
    s['thresholds'] = [d['threshold'] for d in directions]
    return s


def arm_summary(runs, keys=('f1_lifted', 'lifted_recall', 'lifted_false_alarms', 'iou_lifted', 'iou_chip')):
    "Mean over an arm's runs and the seed spread: half the range for 2 runs, the standard deviation for 3 or more, 0 for 1."
    out = {'n_runs': len(runs)}
    for k in keys:
        v = np.array([r[k] for r in runs], float)
        out[k] = float(v.mean())
        out[k + '_spread'] = 0.0 if len(v) < 2 else float((v.max() - v.min()) / 2 if len(v) == 2 else v.std())
    return out


def decide(arms, candidate):
    "Apply the rule: go only if the candidate's F1 beats the best other arm by more than the noise and chip IoU is not worse."
    others = {n: a for n, a in arms.items() if n != candidate}
    best = max(others, key=lambda n: others[n]['f1_lifted'])
    c, b = arms[candidate], others[best]
    gain = c['f1_lifted'] - b['f1_lifted']
    need = max(MIN_GAIN, NOISE_FACTOR * max(c['f1_lifted_spread'], b['f1_lifted_spread']))
    chip_ok = c['iou_chip'] >= b['iou_chip'] - CHIP_MARGIN
    return {'best_semantic_arm': best, 'f1_gain': gain, 'required_gain': need, 'chip_iou_ok': bool(chip_ok),
            'go': bool(gain > need and chip_ok)}


def _selftest():
    ids = [f'src{i}' for i in range(10)]
    a, b = split_halves(ids)
    assert a.isdisjoint(b) and a | b == set(ids) and abs(len(a) - len(b)) <= 1
    assert split_halves(ids) == (a, b) and split_halves(reversed(ids)) == (a, b)
    stem_source = {f'{s}_{k}': s for s in ids for k in 'ab'}

    def row(det, tot, fa):
        return dict(iou_chip=0.9, iou_lifted=0.8, chip_total=1, chip_detected=1, chip_false_alarms=0,
                    lifted_total=tot, lifted_detected=det, lifted_false_alarms=fa)
    by_t = {'0.1': {s: row(2, 2, 3) for s in stem_source}, '0.5': {s: row(1, 2, 0) for s in stem_source}}
    d = held_out(by_t, stem_source, (a, b))
    assert [x['threshold'] for x in d] == [0.5, 0.5] and abs(d[0]['f1_lifted'] - 2 / 3) < 1e-9, d
    assert d[0]['lifted_total'] == 2 * len([s for s in stem_source if stem_source[s] in b]) and d[1]['lifted_false_alarms'] == 0
    tie = {'0.1': {s: row(1, 2, 0) for s in stem_source}, '0.5': {s: row(1, 2, 0) for s in stem_source}}
    assert [x['threshold'] for x in held_out(tie, stem_source, (a, b))] == [0.5, 0.5]
    # selection uses only the selection half: make threshold 0.1 great on half A's patches, awful on B's
    by_t2 = {'0.1': {s: row(2, 2, 0) if stem_source[s] in a else row(0, 2, 9) for s in stem_source},
             '0.5': {s: row(1, 2, 0) for s in stem_source}}
    d2 = held_out(by_t2, stem_source, (a, b))
    assert d2[0]['threshold'] == 0.1 and d2[1]['threshold'] == 0.5      # A->B picks 0.1 on A; B->A picks 0.5 on B
    assert d2[0]['lifted_detected'] == 0                                # ... and then scores 0.1 on B: it does not transfer
    assert abs(arm_summary([{'f1_lifted': .5, 'lifted_recall': 0, 'lifted_false_alarms': 0, 'iou_lifted': 0, 'iou_chip': 0},
                            {'f1_lifted': .6, 'lifted_recall': 0, 'lifted_false_alarms': 0, 'iou_lifted': 0, 'iou_chip': 0}])['f1_lifted_spread'] - .05) < 1e-9

    def arm(f1, spread, chip=.9):
        return dict(f1_lifted=f1, f1_lifted_spread=spread, iou_chip=chip)
    assert decide({'rf': arm(.80, .005), 'u': arm(.70, .005)}, 'rf')['go']
    assert not decide({'rf': arm(.705, .001), 'u': arm(.70, .001)}, 'rf')['go']            # gain below MIN_GAIN
    assert not decide({'rf': arm(.74, .03), 'u': arm(.70, .02)}, 'rf')['go']               # gain below 2 * spread
    assert not decide({'rf': arm(.80, .005, chip=.85), 'u': arm(.70, .005)}, 'rf')['go']   # chip IoU worse by > 0.01
    assert decide({'rf': arm(.80, .0), 'u': arm(.70, .0), 'u2': arm(.79, .0)}, 'rf')['best_semantic_arm'] == 'u2'
    print('selftest ok')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--metadata', help='patch cache metadata.json (val entries carry source_id)')
    p.add_argument('--arm', action='append', default=[], help='NAME=per_patch1.json,per_patch2.json (one file per seed)')
    p.add_argument('--candidate', help='arm name of the instance model')
    p.add_argument('--out')
    a = p.parse_args(argv)
    if a.selftest:
        return _selftest()
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
            runs.append(run_summary(held_out(by_t, stem_source, halves)))
        arms[name] = arm_summary(runs)
        detail[name] = runs
    verdict = decide(arms, a.candidate)
    res = {'n_sources': [len(halves[0]), len(halves[1])], 'arms': arms, 'runs': detail, 'verdict': verdict}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({'arms': arms, 'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
