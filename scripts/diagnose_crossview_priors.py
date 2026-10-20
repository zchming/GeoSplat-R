"""One frozen CPU-only cross-view diagnostic on fourteen leaves training files."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '4'

import argparse
import io
import json
from pathlib import Path
import signal
import sys
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from diagnose_train_support import Budget, color, save_csv, save_json
from geosplat_r.train_diagnostics import (AllowlistedInputs, NAMES, PLAN_SHA256 as D1_SHA,
                                         fold_for_id, sha, verify_full_fit)
from geosplat_r.crossview_diagnostics import (PLAN_SHA256, evaluate_views, summarize,
                                             validate_camera)


def config_file(path):
    path = path.resolve(strict=True)
    if (path.suffix != '.json' or path.name == 'metrics.json'
            or any(p.lower() == 'test' or 'evaluation' in p.lower() for p in path.parts)):
        raise ValueError('Forbidden configuration location')
    return path


def prevalidate_output(output, roots):
    if output.exists():
        raise FileExistsError(output)
    target = output.resolve()
    if target == REPO or REPO in target.parents or target in REPO.parents:
        raise ValueError('Output overlaps repository')
    for root in roots.values():
        root = root.resolve(strict=True)
        if target == root or root in target.parents or target in root.parents:
            raise ValueError('Output overlaps asset roots')
    return target


def load_assets(loader, plan):
    for row in plan['allowlisted_files']:
        loader.read(row['root'], row['path'])
    train, cameras = loader.json('train', 'input-manifest.json'), loader.json('train', 'cameras.json')
    anchors_meta, priors_meta = loader.json('anchors', 'manifest.json'), loader.json('priors', 'manifest.json')
    if train['scene'] != 'leaves' or train['split'] != 'train' or train['resolution'] != [504, 378]:
        raise ValueError('Unexpected train manifest')
    for meta in [train, anchors_meta, priors_meta]:
        if meta['protocol'] != 'leaves-undistorted-v1':
            raise ValueError('Frozen protocol mismatch')
        if (meta['cameras_sha256'] != sha(loader.read('train', 'cameras.json'))
                or meta['points_sha256'] != sha(loader.read('train', 'points.npz'))):
            raise ValueError('Train geometry provenance mismatch')
    if priors_meta['anchors_manifest_sha256'] != sha(loader.read('anchors', 'manifest.json')):
        raise ValueError('Anchor/prior provenance mismatch')
    if ([v['name'] for v in cameras] != NAMES or [r['image'] for r in anchors_meta['records']] != NAMES
            or [r['image'] for r in priors_meta['records']] != NAMES):
        raise ValueError('Unexpected training views')
    if train['train_images'] != [{'name': v['name'], 'sha256': v['image_sha256']} for v in cameras]:
        raise ValueError('RGB manifest mismatch')
    assets = []
    for i, name in enumerate(NAMES):
        stem = Path(name).stem
        a, p = loader.arrays('anchors', stem + '.npz'), loader.arrays('priors', stem + '.npz')
        if set(a) != {'xy', 'inverse_z', 'point_ids'} or set(p) != {'raw_inverse_depth', 'inverse_z', 'valid'}:
            raise ValueError('Unexpected cached schema')
        if len(a['xy']) != plan['known_anchor_counts'][i] or anchors_meta['records'][i]['anchors'] != len(a['xy']):
            raise ValueError('Frozen anchor count changed')
        for root, meta in [('anchors', anchors_meta['records'][i]), ('priors', priors_meta['records'][i])]:
            if meta['file'] != stem + '.npz' or meta['sha256'] != sha(loader.read(root, stem + '.npz')):
                raise ValueError('Array manifest mismatch')
        if a['point_ids'].dtype != np.int64 or len(np.unique(a['point_ids'])) != len(a['point_ids']):
            raise ValueError('Invalid global point IDs')
        if (p['inverse_z'].shape != (378, 504) or not np.isfinite(p['inverse_z']).all()
                or p['valid'].shape != (378, 504) or p['valid'].dtype != np.bool_):
            raise ValueError('Invalid cached target/mask schema')
        rgb_bytes = loader.read('train', 'images/' + name)
        if sha(rgb_bytes) != cameras[i]['image_sha256'] or priors_meta['records'][i]['image_sha256'] != sha(rgb_bytes):
            raise ValueError('Training RGB hash mismatch')
        with Image.open(io.BytesIO(rgb_bytes)) as rgb:
            if rgb.size != (504, 378) or rgb.mode != 'RGB':
                raise ValueError('Frozen RGB dimensions changed')
            rgb = rgb.copy()
        validate_camera(cameras[i])
        assets.append(dict(name=name, camera=cameras[i], prior=p, anchors=a, rgb=rgb))
    # This guard is separate from the feature kernel; no full-fit values are passed to it.
    for i, asset in enumerate(assets):
        verify_full_fit(asset['prior'], asset['anchors'], priors_meta['records'][i]['alignment'])
    return assets


def view_panel(asset, targets, output):
    canvas = Image.new('RGB', (2016, 474), 'white')
    draw = ImageDraw.Draw(canvas)
    panels = [asset['rgb'].copy()]
    panels += [Image.blend(asset['rgb'], Image.new('RGB', (504, 378), '#cccccc'), .5) for _ in range(3)]
    brushes = [ImageDraw.Draw(p) for p in panels]
    status_colors = ['#777777', '#ff7f00', '#377eb8']
    for row in targets:
        if row['image'] != asset['name']:
            continue
        x, y = row['x'], row['y']
        for i, key, cap in [(1, 'score', 2.), (2, 'reference_error', .5)]:
            if row[key] is not None:
                brushes[i].ellipse((x - 3, y - 3, x + 3, y + 3), fill=tuple(color(row[key], cap).tolist()))
        brushes[3].ellipse((x - 3, y - 3, x + 3, y + 3), fill=status_colors[row['source_count']])
    labels = ['Train RGB', 'Sparse two-source disagreement [0,2]',
              'Sparse OOF reference residual [0,0.5]', 'Available sources: 0 gray / 1 orange / 2 blue']
    draw.text((5, 4), asset['name'] + ' | UNKNOWN unmarked pixels; no dense truth, confidence or reliability weights', fill='black')
    for i, panel in enumerate(panels):
        canvas.paste(panel, (504 * i, 48))
        draw.text((504 * i + 5, 24), labels[i], fill='black')
    draw.text((5, 433), 'Score/error: blue=0, orange=display cap; all original numeric values and missing rows retained.', fill='black')
    draw.text((5, 451), 'Two sources means availability only. Agreement can be wrong; source-prior-closer sign never filters observations.', fill='black')
    canvas.save(output)


def run(args):
    roots = {'train': args.train_data, 'anchors': args.anchors, 'priors': args.priors}
    target = prevalidate_output(args.output, roots)
    target.mkdir(parents=True, exist_ok=False)
    budget, loader = Budget(), None
    status, error = 'failed', None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def alarm_handler(*_):
        raise TimeoutError('Frozen ten-minute diagnostic alarm')
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(600)
    try:
        plan_bytes = config_file(args.plan).read_bytes()
        if sha(plan_bytes) != PLAN_SHA256:
            raise ValueError('D2 plan differs from frozen proposal')
        plan = json.loads(plan_bytes)
        d1_bytes = (REPO / plan['input_contract']['path']).read_bytes()
        if sha(d1_bytes) != D1_SHA or sha(d1_bytes) != plan['input_contract']['sha256']:
            raise ValueError('D1 input contract changed')
        d1_plan = json.loads(d1_bytes)
        manifest_path = config_file(args.input_allowlist)
        if any(root.resolve() == manifest_path or root.resolve() in manifest_path.parents for root in roots.values()):
            raise ValueError('Independent allowlist must be outside assets')
        manifest_bytes = manifest_path.read_bytes()
        loader = AllowlistedInputs(roots, d1_plan, json.loads(manifest_bytes))
        cv2.setNumThreads(4)
        assets = load_assets(loader, d1_plan)
        budget.check()
        groups = {}
        for asset in assets:
            for pid in asset['anchors']['point_ids']:
                groups.setdefault(int(pid), []).append(asset['name'])
        save_json(target / 'point-folds.json', [{'point_id': p, 'fold': fold_for_id(p), 'observed_views': v}
                                               for p, v in sorted(groups.items())])
        source_names = ['scripts/diagnose_crossview_priors.py', 'src/geosplat_r/crossview_diagnostics.py',
            'tests/test_crossview_diagnostics.py', 'scripts/diagnose_train_support.py',
            'src/geosplat_r/train_diagnostics.py', 'src/geosplat_r/depth.py', 'scripts/export_train_anchors.py']
        sources = {name: sha((REPO / name).read_bytes()) for name in source_names}
        save_json(target / 'provenance.json', {'protocol': plan['protocol'], 'plan_sha256': PLAN_SHA256,
            'd1_contract_sha256': D1_SHA, 'allowlist_sha256': sha(manifest_bytes), 'source_hashes': sources,
            'versions': {'numpy': np.__version__, 'opencv': cv2.__version__},
            'authorization_basis': 'User approved D2 implementation, checks and one frozen offline diagnostic; proposal status is historical.'})
        fits, targets, directions = evaluate_views(assets, budget.check)
        summary = summarize(assets, fits, targets, directions)
        if (len(fits), len(targets), len(directions)) != (15, 610, 1220):
            raise ValueError('Frozen record counts changed')
        if len(loader.accesses) != 14 or loader.denials or any(x in sys.modules for x in ['torch', 'gsplat', 'depth_anything_v2']):
            raise RuntimeError('Unexpected asset or model/renderer import')
        summary.update(protocol=plan['protocol'], scene='leaves', plan_sha256=PLAN_SHA256,
                       source_hashes=sources, input_asset_reads=len(loader.accesses))
        save_json(target / 'fits.json', fits)
        save_csv(target / 'targets.csv', targets)
        save_csv(target / 'directions.csv', directions)
        save_json(target / 'summary.json', summary)
        for asset in assets:
            budget.check()
            view_panel(asset, targets, target / (Path(asset['name']).stem + '-diagnostic.png'))
        budget.check()
        if sum(p.stat().st_size for p in target.rglob('*') if p.is_file()) > 200 * 2**20:
            raise RuntimeError('Frozen 200 MiB output budget exceeded')
        status = 'passed'
    except BaseException as exc:
        error = repr(exc)
        save_json(target / 'failure.json', {'error': error, 'traceback': traceback.format_exc()})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        save_json(target / 'resources.json', {'status': status, 'error': error, **budget.resources()})
        save_json(target / 'input-access.json', {'assets': loader.accesses if loader else [],
                                               'denials': loader.denials if loader else []})
    print(json.dumps({'status': status, 'screening_pass': summary['screening_pass'], **budget.resources()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['train-data', 'anchors', 'priors', 'input-allowlist', 'plan', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
