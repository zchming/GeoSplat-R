"""One frozen leaves raw-track diagnostic; no models, images, priors or training."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import importlib.abc
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from geosplat_r.raw_track_diagnostics import (PLAN_SHA256, PROPOSAL_SHA256,
    accept_model, evaluate_tracks, summarize)


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'gsplat', 'pycolmap', 'depth_anything_v2', 'fused_ssim'}:
            raise RuntimeError('Model/training imports forbidden: ' + fullname)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def config_location(path):
    path = path.absolute()
    if (path.resolve(strict=True) != path or path.suffix != '.json'
            or any(p.lower() == 'test' or 'evaluation' in p.lower() for p in path.parts)):
        raise ValueError('Invalid config location')
    return path


def output_location(path, train, preparation):
    if path.exists():
        raise FileExistsError(path)
    path = path.absolute()
    if path.resolve() != path:
        raise ValueError('Noncanonical output path')
    for root in [REPO, train.parent]:
        if path == root or root in path.parents or path in root.parents:
            raise ValueError('Output overlaps source or prepared asset root')
    if path == preparation or preparation in path.parents or path in preparation.parents:
        raise ValueError('Output overlaps preparation metadata')
    return path


class Inputs:
    def __init__(self, train, preparation, plan, allowlist):
        train, preparation = train.absolute(), preparation.absolute()
        for path in [train, preparation]:
            if path.resolve(strict=True) != path or any(p.lower() == 'test' or 'evaluation' in p.lower() for p in path.parts):
                raise ValueError('Noncanonical/non-training input')
        if train.name != 'train' or not train.is_dir() or not preparation.is_file():
            raise ValueError('Explicit train leaf and preparation file required')
        if (set(allowlist) != {'protocol', 'plan_sha256', 'files'}
                or allowlist['protocol'] != plan['protocol'] or allowlist['plan_sha256'] != PLAN_SHA256):
            raise ValueError('Allowlist contract mismatch')
        expected = {(row['root'], row['path']): row for row in plan['inputs']}
        if len(expected) != 6 or len(allowlist['files']) != 6:
            raise ValueError('Exactly six input assets required')
        self.files, self.records = {}, {}
        self.cache, self.accesses = {}, []
        for row in allowlist['files']:
            if set(row) != {'root', 'path', 'absolute_path', 'bytes', 'sha256'}:
                raise ValueError('Unexpected allowlist fields')
            key = row['root'], row['path']
            if (key in self.files or key not in expected or Path(row['path']).is_absolute()
                    or '..' in Path(row['path']).parts or {k: row[k] for k in expected[key]} != expected[key]):
                raise ValueError('Unapproved/duplicate input metadata')
            candidate = train / row['path'] if row['root'] == 'train' else preparation
            if (candidate.resolve(strict=True) != candidate or str(candidate) != row['absolute_path']
                    or not candidate.is_file()):
                raise ValueError('Input path mismatch or symlink escape')
            self.files[key], self.records[key] = candidate, row

    def read_all(self):
        for key, path in self.files.items():
            data = path.read_bytes()
            row = self.records[key]
            if len(data) != row['bytes'] or digest(data) != row['sha256']:
                raise ValueError('Frozen asset hash/size drift: ' + str(key))
            self.cache[key] = data
            self.accesses.append({k: row[k] for k in ['root', 'path', 'bytes', 'sha256']})
        cam_bytes = self.cache['train', 'cameras.json']
        manifest = json.loads(self.cache['train', 'input-manifest.json'])
        if manifest['cameras_sha256'] != digest(cam_bytes):
            raise ValueError('Camera provenance mismatch')
        return dict(cameras=json.loads(cam_bytes), manifest=manifest,
                    preparation=json.loads(self.cache['preparation', 'preparation.json']),
                    cameras_bin=self.cache['train', 'triangulated/cameras.bin'],
                    images_bin=self.cache['train', 'triangulated/images.bin'],
                    tracks_bin=self.cache['train', 'triangulated/points3D.bin'])


class ReadBoundary:
    """Read audit for trusted diagnostic code, not an untrusted-code sandbox."""
    def __init__(self, files, output):
        self.files, self.output = set(files), output
        self.code = {Path(__file__).resolve(), REPO / 'src/geosplat_r/raw_track_diagnostics.py',
                     REPO / 'src/geosplat_r/train_diagnostics.py'}
        self.libraries = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        self.reads, self.denials = set(), []

    def __call__(self, event, arguments):
        if event == 'socket.connect':
            self.denials.append({'event': event, 'reason': 'network forbidden'})
            raise RuntimeError('Network forbidden')
        if event != 'open' or not isinstance(arguments[0], (str, bytes)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        mode, flags = arguments[1:3]
        write = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR)
        if path == self.output or self.output in path.parents:
            return
        if not write and path in self.files:
            self.reads.add(str(path))
            return
        if not write and (path in self.code or any(root in path.parents for root in self.libraries)):
            return
        self.denials.append({'event': event, 'path': str(path), 'write': bool(write)})
        raise RuntimeError('Access outside frozen input/output/code allowlist: ' + str(path))


class Budget:
    def __init__(self):
        self.wall, self.perf = time.time(), time.perf_counter()

    def resources(self):
        return {'elapsed_wall_seconds': time.time() - self.wall,
                'elapsed_perf_counter_seconds': time.perf_counter() - self.perf,
                'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                'rss_method': 'Linux ru_maxrss KiB/1024, process high-water RSS',
                'device': 'CPU', 'threads_max': 4, 'model_inference': False,
                'rendering': False, 'optimizer_updates': 0, 'cuda_initialized': False}

    def check(self):
        if self.resources()['peak_rss_mib'] > 2048:
            raise MemoryError('2 GiB RSS budget exceeded')
        if time.perf_counter() - self.perf > 600:
            raise TimeoutError('Ten-minute budget exceeded')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train-data', 'preparation', 'plan', 'input-allowlist', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='G0 input acceptance only; no scores/residuals')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    train = args.train_data.absolute()
    prep = args.preparation.absolute()
    output = output_location(args.output, train, prep)
    plan_bytes = config_location(args.plan).read_bytes()
    if digest(plan_bytes) != PLAN_SHA256:
        raise ValueError('Frozen plan hash mismatch')
    plan = json.loads(plan_bytes)
    if (plan['plan_sha256'] != PROPOSAL_SHA256 or plan['finite_difference_step_px'] != .001
            or plan['source_noise_std_px'] != 1.):
        raise ValueError('Unexpected proposal/numeric settings')
    allow = json.loads(config_location(args.input_allowlist).read_bytes())
    loader = Inputs(train, prep, plan, allow)
    if any(name.split('.')[0] in {'torch', 'gsplat', 'pycolmap', 'depth_anything_v2'} for name in sys.modules):
        raise RuntimeError('Heavy model import detected')
    sys.meta_path.insert(0, NoModels())
    output.mkdir(parents=True, exist_ok=False)
    boundary = ReadBoundary(loader.files.values(), output)
    sys.addaudithook(boundary)
    budget = Budget()
    def timeout(signum, frame):
        raise TimeoutError('Frozen execution timeout')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    status = 'failed'
    try:
        blobs = loader.read_all()
        cameras, observations, inventory = accept_model(blobs)
        budget.check()
        save_json(output / 'input-acceptance.json', inventory)
        if not args.validate_only:
            rows = evaluate_tracks(cameras, observations, budget.check)
            summary = summarize(rows, len(observations))
            save_json(output / 'records.json', rows)
            with (output / 'records.csv').open('x', newline='') as stream:
                keys = ['point_id', 'fold', 'image', 'track_length', 'score', 'angle_control', 'residual_px', 'status', 'angle_status']
                writer = csv.DictWriter(stream, keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            save_json(output / 'summary.json', summary)
        budget.check()
        status = 'complete'
    except Exception as error:
        save_json(output / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                                          'traceback': traceback.format_exc()})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        save_json(output / 'access.json', {'read_files': sorted(boundary.reads),
                                        'verified_inputs': loader.accesses, 'denials': boundary.denials})
        save_json(output / 'resources.json', budget.resources())
        save_json(output / 'status.json', {'status': status, 'validate_only': args.validate_only,
                   'plan_sha256': PLAN_SHA256, 'proposal_sha256': PROPOSAL_SHA256,
                   'torch_imported': 'torch' in sys.modules, 'models_loaded': False})
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 50 * 1024**2:
            raise RuntimeError('Output budget exceeded; preserve output and investigate')
    return 0


if __name__ == '__main__':
    sys.exit(main())
