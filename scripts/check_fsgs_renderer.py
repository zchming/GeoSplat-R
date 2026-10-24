"""Diagnose native/gsplat FSGS visibility differences after completed evaluation.

This is an explicit secondary compatibility diagnostic, not a replacement for
previous frozen evaluation scores. It never updates checkpoint parameters.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geosplat_r.baseline import load_views, sha256
from geosplat_r.fsgs_render import NATIVE_NEAR_PLANE, render_native_compatible


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint', 'test-data', 'native-metrics', 'weights', 'output']:
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8)
    os.environ['TORCH_HOME'] = str(args.weights)
    if sha256(args.weights / 'hub/checkpoints/alexnet-owt-7be5be79.pth') != '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02':
        raise ValueError('Explicit verified AlexNet weights required')
    checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
    native = json.loads(args.native_metrics.read_text())
    if checkpoint['source_checkpoint_sha256'] != native['checkpoint_sha256']:
        raise ValueError('Native scores must belong to the same source model')
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda().eval()
    records = []
    for view in load_views(args.test_data):
        output = render_native_compatible(checkpoint, view)[0][0].clamp(0, 1)
        prediction = output.permute(2, 0, 1)[None]
        target = view['pixels'].cuda().permute(2, 0, 1)[None]
        z = (torch.cat([checkpoint['splats']['means'], torch.ones_like(checkpoint['splats']['means'][:, :1])], dim=1)
             @ view['view'][0].T)[:, 2]
        row = dict(image=view['name'], psnr=float(-10 * torch.log10((prediction - target).square().mean())),
                   ssim=float(structural_similarity_index_measure(prediction, target, gaussian_kernel=True,
                               sigma=1.5, kernel_size=11, data_range=1., k1=.01, k2=.03)),
                   lpips_alex=float(lpips(prediction, target)),
                   centers_between_standard_and_native_clip=int(((z >= .01) & (z <= .2)).sum()))
        lpips.reset()
        source = next(p for p in native['per_image'] if p['image'] == view['name'])
        row['absolute_metric_difference_vs_native'] = {k:abs(row[k] - source[k]) for k in ['psnr','ssim','lpips_alex']}
        records.append(row)
        Image.fromarray((output.cpu().numpy() * 255).round().astype(np.uint8)).save(args.output / view['name'])
    report = dict(scope='post-evaluation compatibility diagnostic; original scores retained',
                  native_rule='CUDA in_frustum: reject center Z <= 0.2', gsplat_near_plane=NATIVE_NEAR_PLANE,
                  checkpoint_sha256=sha256(args.checkpoint), per_image=records,
                  mean={k:float(np.mean([p[k] for p in records])) for k in ['psnr','ssim','lpips_alex']},
                  max_absolute_metric_difference_vs_native={k:max(p['absolute_metric_difference_vs_native'][k] for p in records)
                                                          for k in ['psnr','ssim','lpips_alex']})
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
