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
from segmentation_tools_pt.model import get_model_from_cfg
from segmentation_tools_pt.transforms import get_val_transforms

from _bench_common import aggregate, ids_sha1, patch_scores


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
    rows = {t: {} for t in thresholds}
    for b in DataLoader(ds, batch_size=16):
        prob = torch.sigmoid(model(b['image'].to(device))).cpu()
        gt = b['mask'].bool()
        if report_ids is not None:
            for k, sid in enumerate(b['sample_id']):
                if sid in report_ids:
                    keep[sid] = (cv2.imread(str(Path(root) / 'val' / 'images' / f'{sid}.png'), cv2.IMREAD_UNCHANGED),
                                 gt[k].numpy(), (prob[k] > report_threshold).numpy())
        for t in thresholds:
            pred = (prob > t).numpy()
            for k, sid in enumerate(b['sample_id']):
                rows[t][sid] = patch_scores(pred[k], gt[k].numpy())
    out = {str(t): aggregate(r.values()) for t, r in rows.items()}
    return out, keep, ids_sha1(ds.sample_ids), {str(t): r for t, r in rows.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--out', required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999])
    p.add_argument('--ids', default=None, help='sample_ids.json written by rfdetr_benchmark.py eval --report: also build a review report on those patches')
    p.add_argument('--report-dir', default=None)
    p.add_argument('--report-id', default='semantic-baseline')
    p.add_argument('--report-threshold', type=float, default=0.5)
    a = p.parse_args()
    ids = set(json.loads(Path(a.ids).read_text())) if a.ids else None
    res, keep, val_sha1, per_patch = sweep(a.run_dir, a.thresholds, ids, a.report_threshold)
    if ids is not None:
        assert len(keep) == len(ids), f'{len(ids) - len(keep)} of the {len(ids)} --ids are not in this run\'s val split'
        from segmentation_tools_pt.review_report import Sample, build_report
        samples = [Sample(s, keep[s][0], keep[s][1].astype(np.uint8), keep[s][2].astype(np.uint8), class_names=['chip', 'lifted'])
                   for s in json.loads(Path(a.ids).read_text()) if s in keep]
        print('report:', build_report(a.report_id, a.report_dir, samples), f'({len(samples)} of {len(ids)} samples)')
    Path(a.out).with_name(Path(a.out).stem + '_per_patch.json').write_text(json.dumps(per_patch))
    Path(a.out).write_text(json.dumps({'run_dir': a.run_dir, 'val_ids_sha1': val_sha1, 'thresholds': res}, indent=1))
    print(json.dumps(res, indent=1))
