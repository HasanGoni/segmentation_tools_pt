"""RF-DETR seg vs the semantic baseline on a multilabel patch cache (issue #102).

Run it in a SEPARATE environment that has `rfdetr[train]` (the library does not depend on rfdetr):

    PYTHONPATH=src python scripts/rfdetr_benchmark.py train --cache PATCHES --out RUN --model small --epochs 15
    PYTHONPATH=src python scripts/rfdetr_benchmark.py eval  --cache PATCHES --out RUN

`train` converts the cache with `write_coco_folder` (per-class layout) and calls `RFDETRSeg*.train`.
`eval` predicts every val patch, rasterises with `detections_to_semantic`, and scores the result with the same
`object_metrics` / IoU code the semantic A/B notebook (`tutorials/07_hierarchy_ab_real_data.ipynb`) uses.
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

CLASSES = ['chip', 'lifted']
CHIP, LIFTED = 0, 1


def _iou(p, g):
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 1.0


def cmd_train(a):
    from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium
    from segmentation_tools_pt.masks import write_coco_folder
    out = Path(a.out)
    coco = out / 'coco'
    if not (coco / 'valid' / '_annotations.coco.json').exists():
        print('coco folder:', write_coco_folder(a.cache, str(coco), CLASSES, segmentation=a.segmentation))
    cls = {'nano': RFDETRSegNano, 'small': RFDETRSegSmall, 'medium': RFDETRSegMedium}[a.model]
    model = cls(num_channels=1)
    t = time.time()
    model.train(dataset_dir=str(coco), epochs=a.epochs, batch_size=a.batch_size, device='cuda',
                num_workers=a.workers, output_dir=str(out / 'train'), lr=a.lr, early_stopping=False)
    print('TRAIN_SECONDS', round(time.time() - t))


def cmd_eval(a):
    from rfdetr import RFDETR
    from segmentation_tools_pt.inference import detections_to_semantic
    from segmentation_tools_pt.metrics import object_metrics, subset_violation_rate
    from PIL import Image
    out = Path(a.out)
    ckpt = Path(a.ckpt) if a.ckpt else out / 'train' / 'last_ema.pth'
    model = RFDETR.from_checkpoint(str(ckpt))
    root = Path(a.cache) / 'val'
    stems = sorted(p.stem for p in (root / 'images').glob('*.png'))
    floor = min(a.thresholds)
    acc = {t: dict(iou=[[], []], viol=[], obj=[dict(total=0, detected=0, fa=0) for _ in CLASSES],
                   n_inst=[0, 0]) for t in a.thresholds}
    keep = {}
    t0 = time.time()
    for n, s in enumerate(stems):
        img = np.array(Image.open(root / 'images' / f'{s}.png'))
        gt = np.stack([cv2.imread(str(root / 'masks' / f'{s}_{c}.png'), -1) > 0 for c in CLASSES])
        dets = model.predict(Image.fromarray(img), threshold=floor)
        for t in a.thresholds:
            stack = detections_to_semantic(dets, shape=gt.shape[1:], class_ids=[0, 1], first_class_id=0, output='stack', threshold=t)
            pred = stack > 0
            r = acc[t]
            for k in range(len(CLASSES)):
                r['iou'][k].append(_iou(pred[k], gt[k]))
                o = object_metrics(pred[k].astype(np.uint8), gt[k].astype(np.uint8))
                r['obj'][k]['total'] += o['total_gt']; r['obj'][k]['detected'] += o['detected']; r['obj'][k]['fa'] += o['false_alarms']
            r['viol'].append(subset_violation_rate(pred[LIFTED], pred[CHIP]))
            if t == a.primary:
                keep[s] = (img, gt, pred)
        if n % 100 == 0:
            print(f'{n}/{len(stems)} patches, {time.time() - t0:.0f}s', flush=True)
    rows = {}
    for t, r in acc.items():
        rows[str(t)] = {
            'iou_chip': float(np.mean(r['iou'][CHIP])), 'iou_lifted': float(np.mean(r['iou'][LIFTED])),
            'lifted_total': r['obj'][LIFTED]['total'], 'lifted_detected': r['obj'][LIFTED]['detected'],
            'lifted_recall': r['obj'][LIFTED]['detected'] / max(r['obj'][LIFTED]['total'], 1),
            'lifted_false_alarms': r['obj'][LIFTED]['fa'],
            'chip_recall': r['obj'][CHIP]['detected'] / max(r['obj'][CHIP]['total'], 1),
            'chip_false_alarms': r['obj'][CHIP]['fa'],
            'violation': float(np.mean(r['viol'])),
        }
    res = {'checkpoint': str(ckpt), 'n_val': len(stems), 'primary_threshold': a.primary, 'thresholds': rows}
    (out / 'eval.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if a.report:
        from segmentation_tools_pt.review_report import Sample, build_report
        lifted_iou = {s: _iou(p[LIFTED], g[LIFTED]) for s, (_, g, p) in keep.items()}
        worst = sorted(keep, key=lambda s: lifted_iou[s])[:40]
        rest = [s for s in stems if s not in set(worst)]
        pick = worst + list(np.random.default_rng(0).choice(rest, 40, replace=False))
        (out / 'sample_ids.json').write_text(json.dumps([str(x) for x in pick]))   # the baseline report reuses these
        samples = [Sample(s, keep[s][0], keep[s][1].astype(np.uint8), keep[s][2].astype(np.uint8),
                          class_names=CLASSES, score=lifted_iou[s]) for s in pick]
        print('report:', build_report(f'rfdetr-{a.model_tag}', out / 'report', samples))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); e = sub.add_parser('eval')
    for q in (t, e):
        q.add_argument('--cache', required=True, help='multilabel patch cache (train/ val/ with images/ and masks/{id}_{class}.png)')
        q.add_argument('--out', required=True)
    t.add_argument('--model', choices=['nano', 'small', 'medium'], default='small')
    t.add_argument('--epochs', type=int, default=15)
    t.add_argument('--batch-size', type=int, default=8)
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--lr', type=float, default=1e-4)
    t.add_argument('--segmentation', default='auto', choices=['auto', 'polygon', 'rle'])
    e.add_argument('--ckpt', default=None, help='default: OUT/train/last_ema.pth (best_total can pick epoch 1)')
    e.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7])
    e.add_argument('--primary', type=float, default=0.5, help='fixed before looking at results; the rest is a sweep')
    e.add_argument('--report', action='store_true')
    e.add_argument('--model-tag', default='small')
    a = p.parse_args(argv)
    {'train': cmd_train, 'eval': cmd_eval}[a.cmd](a)


if __name__ == '__main__':
    main()
