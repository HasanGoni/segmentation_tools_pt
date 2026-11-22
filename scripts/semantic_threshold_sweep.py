"""Score a finished semantic run (`best.pth`) on the val split at several probability thresholds (issue #102).

Same evaluation as `tutorials/07_hierarchy_ab_real_data.ipynb` (`SegmentationDataset`, `get_val_transforms`, per-class IoU,
`object_metrics` on `lifted`), but with one shared threshold per sweep point instead of the run's `metrics.threshold`,
so a semantic model gets the same threshold tuning an instance model does in `scripts/rfdetr_benchmark.py`.

    python scripts/semantic_threshold_sweep.py outputs/hierarchy_ab/w0.0_s42 --out sweep_w0.0_s42.json
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from segmentation_tools_pt.dataset import SegmentationDataset
from segmentation_tools_pt.hydra_pipeline import resolve_channel_preset, resolve_crop_size, resolve_mask_cfg, run_patch_pipeline
from segmentation_tools_pt.metrics import object_metrics
from segmentation_tools_pt.model import get_model_from_cfg
from segmentation_tools_pt.transforms import get_val_transforms

CHIP, LIFTED = 0, 1


def _iou(p, g):
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 1.0


@torch.no_grad()
def sweep(run_dir, thresholds, report_ids=None, report_threshold=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    cfg = OmegaConf.load(Path(run_dir) / '.hydra' / 'config.yaml')
    preset = resolve_channel_preset(cfg.data.data_mode)
    mask_cfg = resolve_mask_cfg(cfg)
    mean, std, root = run_patch_pipeline(cfg, Path(cfg.data.raw_dir), Path(cfg.data.patch_dir), preset['n_source_channels'],
                                         preset['channel_names'], cfg.data.mask_suffix or '')
    ds = SegmentationDataset(root, 'val', transform=get_val_transforms(cfg.train.crop_size, tuple(mean), tuple(std)),
                             num_classes=cfg.train.num_classes, n_source_channels=preset['n_source_channels'],
                             channel_names=preset['channel_names'], **mask_cfg)
    model = get_model_from_cfg(cfg, preset['in_channels'], crop_size=resolve_crop_size(cfg), pretrained=False)
    model.load_state_dict(torch.load(Path(run_dir) / 'checkpoints' / 'best.pth', map_location='cpu', weights_only=True)['model_state'])
    model.to(device).eval()
    keep = {}
    acc = {t: dict(iou=[[], []], obj=[dict(total=0, detected=0, fa=0) for _ in range(2)]) for t in thresholds}
    for b in DataLoader(ds, batch_size=16):
        prob = torch.sigmoid(model(b['image'].to(device))).cpu()
        gt = b['mask'].bool()
        if report_ids is not None:
            for k, sid in enumerate(b['sample_id'] if 'sample_id' in b else []):
                if sid in report_ids:
                    keep[sid] = (cv2.imread(str(Path(root) / 'val' / 'images' / f'{sid}.png'), cv2.IMREAD_UNCHANGED), gt[k].numpy(), (prob[k] > report_threshold).numpy())
        for t in thresholds:
            pred = prob > t
            for k in range(len(pred)):
                for c in (CHIP, LIFTED):
                    p, g = pred[k, c], gt[k, c]
                    acc[t]['iou'][c].append(_iou(p, g))
                    o = object_metrics(p.numpy().astype(np.uint8), g.numpy().astype(np.uint8))
                    acc[t]['obj'][c]['total'] += o['total_gt']; acc[t]['obj'][c]['detected'] += o['detected']
                    acc[t]['obj'][c]['fa'] += o['false_alarms']
    out = {}
    for t, r in acc.items():
        lo, ch = r['obj'][LIFTED], r['obj'][CHIP]
        out[str(t)] = {'iou_chip': float(np.mean(r['iou'][CHIP])), 'iou_lifted': float(np.mean(r['iou'][LIFTED])),
                       'lifted_total': lo['total'], 'lifted_detected': lo['detected'],
                       'lifted_recall': lo['detected'] / max(lo['total'], 1), 'lifted_false_alarms': lo['fa'],
                       'chip_recall': ch['detected'] / max(ch['total'], 1), 'chip_false_alarms': ch['fa']}
    return out, keep


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--out', required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    p.add_argument('--ids', default=None, help='sample_ids.json written by rfdetr_benchmark.py eval --report: also build a review report on those patches')
    p.add_argument('--report-dir', default=None)
    p.add_argument('--report-id', default='semantic-baseline')
    p.add_argument('--report-threshold', type=float, default=0.5)
    a = p.parse_args()
    ids = set(json.loads(Path(a.ids).read_text())) if a.ids else None
    res, keep = sweep(a.run_dir, a.thresholds, ids, a.report_threshold)
    if ids is not None:
        from segmentation_tools_pt.review_report import Sample, build_report
        samples = [Sample(s, keep[s][0], keep[s][1].astype(np.uint8), keep[s][2].astype(np.uint8), class_names=['chip', 'lifted'])
                   for s in json.loads(Path(a.ids).read_text()) if s in keep]
        print('report:', build_report(a.report_id, a.report_dir, samples), f'({len(samples)} of {len(ids)} samples)')
    Path(a.out).write_text(json.dumps({'run_dir': a.run_dir, 'thresholds': res}, indent=1))
    print(json.dumps(res, indent=1))
