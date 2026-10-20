"""One frozen CPU-only leaves train-proxy diagnostic; no training or rendering."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '4'

import argparse
import csv
import io
import json
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from geosplat_r.train_diagnostics import (PLAN_SHA256, NAMES, AllowlistedInputs,
    analyze_arrays, fold_for_id, sha, verify_full_fit)


class Budget:
    def __init__(self):
        self.wall, self.perf = time.time(), time.perf_counter()

    def resources(self):
        return {'elapsed_wall_seconds': time.time() - self.wall,
                'elapsed_perf_counter_seconds': time.perf_counter() - self.perf,
                'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                'rss_method': 'Linux ru_maxrss KiB / 1024; process high-water RSS',
                'device': 'CPU', 'threads_max': 4, 'optimizer_updates': 0,
                'model_inference': False, 'rendering': False, 'cuda_initialized': False}

    def check(self):
        r = self.resources()
        if max(r['elapsed_wall_seconds'], r['elapsed_perf_counter_seconds']) > 600:
            raise TimeoutError('Frozen ten-minute diagnostic budget exceeded')
        if r['peak_rss_mib'] > 2048:
            raise MemoryError('Frozen 2 GiB RSS target exceeded; stop for investigation')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def config_path(path):
    path = path.resolve(strict=True)
    if path.suffix != '.json' or any(v.lower() == 'test' or 'evaluation' in v.lower() for v in path.parts):
        raise ValueError('Forbidden test/evaluation configuration location')
    return path


def color(value, cap):
    t = np.clip(np.asarray(value) / cap, 0, 1)
    return np.stack([40 + 215 * t, 90 + 55 * t, 210 - 160 * t], axis=-1).astype(np.uint8)


def view_panel(rgb, xy, records, distances, summary, output):
    canvas = Image.new('RGB', (2016, 474), 'white')
    draw = ImageDraw.Draw(canvas)
    panels = [rgb.copy(), rgb.copy(), Image.fromarray(color(distances, .1)),
              Image.blend(rgb, Image.new('RGB', rgb.size, '#cccccc'), .5)]
    anchors, errors = ImageDraw.Draw(panels[1]), ImageDraw.Draw(panels[3])
    palette = ['#377eb8', '#ff7f00', '#984ea3', '#4daf4a', '#e41a1c']
    for x in range(0, 504, 42):
        anchors.line((x, 0, x, 377), fill='#aaaaaa')
        errors.line((x, 0, x, 377), fill='#aaaaaa')
    for y in range(0, 378, 42):
        anchors.line((0, y, 503, y), fill='#aaaaaa')
        errors.line((0, y, 503, y), fill='#aaaaaa')
    for row in records:
        x, y = row['x'], row['y']
        anchors.ellipse((x - 2, y - 2, x + 2, y + 2), fill=palette[row['fold']])
        c = tuple(color(row['error'], .5).tolist()) if row['error'] is not None else '#777777'
        errors.ellipse((x - 3, y - 3, x + 3, y + 3), fill=c)
    labels = ['Train RGB (no test data)', 'Observed anchors / fixed global-point folds',
              'Nearest anchor distance /630; display [0,0.1]', 'Sparse OOF residual /fit median [0,0.5]']
    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (504 * i, 48))
        draw.text((504 * i + 5, 23), label, fill='black')
    draw.text((5, 4), summary['image'] + ' | unmarked residual pixels are UNKNOWN; no interpolation or reliability labels', fill='black')
    draw.text((5, 433), 'Distance/error colors: blue=0, orange=cap; display clipping only, all numerical values retained.', fill='black')
    draw.text((5, 451), f"Anchors {summary['anchors']}; zero-anchor tiles {summary['zero_anchor_tiles']}/108; tiles with >=5 OOF observations {summary['tiles_with_observed_error']}/108", fill='black')
    canvas.save(output)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    target = args.output.resolve()
    if target == REPO or REPO in target.parents:
        raise ValueError('Raw diagnostic output must be outside the repository')
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget()
    loader = None
    status, error = 'failed', None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def alarm_handler(*_):
        raise TimeoutError('Frozen ten-minute diagnostic alarm')
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(600)
    try:
        plan_data = config_path(args.plan).read_bytes()
        if sha(plan_data) != PLAN_SHA256:
            raise ValueError('Plan differs from reviewed D1 proposal')
        plan = json.loads(plan_data)
        roots = {'train': args.train_data, 'anchors': args.anchors, 'priors': args.priors}
        for root in roots.values():
            resolved = root.resolve(strict=True)
            if target == resolved or resolved in target.parents or target in resolved.parents:
                raise ValueError('Diagnostic output overlaps asset roots')
        allow_path = config_path(args.input_allowlist)
        if any(root.resolve() in allow_path.parents for root in roots.values()):
            raise ValueError('Independent allowlist must be outside asset roots')
        loader = AllowlistedInputs(roots, plan, json.loads(allow_path.read_text()))
        # Read exactly the approved asset bytes, never walk an asset directory.
        for item in plan['allowlisted_files']:
            loader.read(item['root'], item['path'])
        train = loader.json('train', 'input-manifest.json')
        cameras = loader.json('train', 'cameras.json')
        anchors_meta, priors_meta = loader.json('anchors', 'manifest.json'), loader.json('priors', 'manifest.json')
        if train['scene'] != 'leaves' or train['split'] != 'train' or train['resolution'] != [504, 378]:
            raise ValueError('Unexpected train manifest')
        for m in [train, anchors_meta, priors_meta]:
            if m['protocol'] != 'leaves-undistorted-v1':
                raise ValueError('Frozen scene protocol mismatch')
            if m['cameras_sha256'] != sha(loader.read('train', 'cameras.json')) or m['points_sha256'] != sha(loader.read('train', 'points.npz')):
                raise ValueError('Train geometry provenance mismatch')
        if priors_meta['anchors_manifest_sha256'] != sha(loader.read('anchors', 'manifest.json')):
            raise ValueError('Anchor/priors provenance mismatch')
        if [v['name'] for v in cameras] != NAMES or [v['image'] for v in anchors_meta['records']] != NAMES or [v['image'] for v in priors_meta['records']] != NAMES:
            raise ValueError('Unexpected training views')
        if train['train_images'] != [{'name': v['name'], 'sha256': v['image_sha256']} for v in cameras]:
            raise ValueError('RGB manifest mismatch')
        cv2.setNumThreads(4)
        assets, groups = [], {}
        for i, name in enumerate(NAMES):
            stem = Path(name).stem
            a = loader.arrays('anchors', stem + '.npz')
            p = loader.arrays('priors', stem + '.npz')
            if set(a) != {'xy', 'inverse_z', 'point_ids'} or set(p) != {'raw_inverse_depth', 'inverse_z', 'valid'}:
                raise ValueError('Unexpected cached array schema')
            if len(a['xy']) != plan['known_anchor_counts'][i] or anchors_meta['records'][i]['anchors'] != len(a['xy']):
                raise ValueError('Frozen anchor count changed')
            for root, meta in [('anchors', anchors_meta['records'][i]), ('priors', priors_meta['records'][i])]:
                if meta['file'] != stem + '.npz' or meta['sha256'] != sha(loader.read(root, stem + '.npz')):
                    raise ValueError('Array manifest mismatch')
            if a['point_ids'].dtype != np.int64 or len(np.unique(a['point_ids'])) != len(a['point_ids']):
                raise ValueError('Invalid cached global point IDs')
            if not np.isfinite(p['inverse_z']).all() or p['inverse_z'].shape != (378, 504):
                raise ValueError('Invalid cached aligned target')
            for pid in a['point_ids']:
                groups.setdefault(int(pid), []).append(name)
            rgb_bytes = loader.read('train', 'images/' + name)
            if sha(rgb_bytes) != cameras[i]['image_sha256'] or priors_meta['records'][i]['image_sha256'] != sha(rgb_bytes):
                raise ValueError('Training RGB hash mismatch')
            with Image.open(io.BytesIO(rgb_bytes)) as image:
                if image.size != (504, 378) or image.mode != 'RGB':
                    raise ValueError('Frozen native RGB dimensions changed')
                rgb = image.copy()
            assets.append((name, a, p, rgb))
        # Check the frozen reference formula before any dependent CV/coverage work.
        for i, (_, anchors, prior, _) in enumerate(assets):
            verify_full_fit(prior, anchors, priors_meta['records'][i]['alignment'])
        assignments = [{'point_id': pid, 'fold': fold_for_id(pid), 'observed_views': views}
                       for pid, views in sorted(groups.items())]
        save_json(args.output / 'point-folds.json', assignments)
        source_names = ['scripts/diagnose_train_support.py', 'src/geosplat_r/train_diagnostics.py',
                        'src/geosplat_r/depth.py', 'scripts/export_train_anchors.py']
        sources = {n: sha((REPO / n).read_bytes()) for n in source_names}
        save_json(args.output / 'provenance.json', {'plan_sha256': PLAN_SHA256,
            'allowlist_sha256': sha(allow_path.read_bytes()), 'source_hashes': sources,
            'versions': {'numpy': np.__version__, 'opencv': cv2.__version__},
            'authorization_basis': 'User approved implementing D1 and one CPU offline diagnostic; frozen proposal status is historical.'})
        summaries, fold_rows, anchor_rows, tile_rows = [], [], [], []
        for i, (name, anchors, prior, rgb) in enumerate(assets):
            budget.check()
            summary, rows, tiles, distance = analyze_arrays(prior, anchors, budget.check)
            summary['image'] = name
            summaries.append(summary)
            fold_rows.extend({'image': name, **x} for x in summary['folds'])
            anchor_rows.extend({'image': name, **x} for x in rows)
            tile_rows.extend({'image': name, **x} for x in tiles)
            view_panel(rgb, anchors['xy'], rows, distance, summary, args.output / (Path(name).stem + '-diagnostic.png'))
            budget.check()
        if len(loader.accesses) != 14 or loader.denials or any(x in sys.modules for x in ['torch', 'gsplat', 'depth_anything_v2']):
            raise RuntimeError('Unexpected assets or model/renderer dependency loaded')
        summary = {'protocol': plan['protocol'], 'scene': 'leaves', 'images': summaries,
            'unique_observed_points': len(groups), 'anchor_observations': len(anchor_rows),
            'fold_records': len(fold_rows), 'tile_records': len(tile_rows),
            'all_cv_complete': all(x['cv_complete'] for x in summaries),
            'limits': 'Train-only sparse SfM proxy, correlated views/points; no dense truth, confidence map, causal LPIPS explanation, significance or reliability weights.',
            'plan_sha256': PLAN_SHA256, 'source_hashes': sources}
        save_json(args.output / 'summary.json', summary)
        save_json(args.output / 'folds.json', fold_rows)
        save_csv(args.output / 'anchor-residuals.csv', anchor_rows)
        save_json(args.output / 'tiles.json', tile_rows)
        flat_tiles = [{k: x[k] for k in ['image', 'tile', 'row', 'column', 'anchor_count', 'pixel_count', 'prior_valid_pixels', 'oof_count', 'error_status']} |
                      {'distance_mean': x['full_distance']['mean'], 'valid_distance_mean': x['prior_valid_distance']['mean'],
                       'oof_error_mean': x['oof_error']['mean'] if x['oof_error'] else None} for x in tile_rows]
        save_csv(args.output / 'tiles.csv', flat_tiles)
        budget.check()
        if sum(x.stat().st_size for x in args.output.rglob('*') if x.is_file()) > 200 * 2**20:
            raise RuntimeError('Frozen output budget exceeded')
        status = 'passed'
    except BaseException as exc:
        error = repr(exc)
        save_json(args.output / 'failure.json', {'error': error, 'traceback': traceback.format_exc()})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        save_json(args.output / 'resources.json', {'status': status, 'error': error, **budget.resources()})
        save_json(args.output / 'input-access.json', {'assets': loader.accesses if loader else [],
                                                   'denials': loader.denials if loader else []})
    print(json.dumps({'status': status, 'summary': str(args.output / 'summary.json'), **budget.resources()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['train-data', 'anchors', 'priors', 'input-allowlist', 'plan', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
