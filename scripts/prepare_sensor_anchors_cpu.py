"""Prepare one frozen three-RGB anchor set, only after synthetic isolation gates.

Explicit source, data, plan, interpreters and fresh output; no models or labels.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

PLAN_SHA = '63179fd0127e668a1586038afe57f1994a6a0e49ceeed0e6da519cef204e6352'


def geometry():
    path = Path(__file__).with_name('sensor_geometry.py')
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / 'src/geosplat_r/sensor_geometry.py'
    spec = importlib.util.spec_from_file_location('sensor_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def worker(args):
    g = geometry()
    import numpy as np
    start = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    try:
        if args.mode in ('probe', 'probe-camera'):
            denied = []
            for path in config['forbidden']:
                for method in ('python', 'os', 'numpy'):
                    try:
                        if method == 'python':
                            Path(path).read_bytes()
                        elif method == 'os':
                            fd = os.open(path, os.O_RDONLY)
                            os.close(fd)
                        else:
                            np.load(path, allow_pickle=False)
                    except OSError:
                        denied.append([path, method])
                    else:
                        raise RuntimeError('Forbidden path readable: ' + path)
            if args.mode == 'probe':
                import cv2
                for path in config['forbidden']:
                    if cv2.imread(path) is not None:
                        raise RuntimeError('OpenCV accessed forbidden input')
                    denied.append([path, 'opencv'])
                image = cv2.imread('/allowed/image.png')
                assert image is not None and image.shape == (3, 3, 3)
                allowed_sum = int(image.sum())
            else:
                import pycolmap
                assert pycolmap.__version__ == '4.2.1'
                c = pycolmap.Camera(model='THIN_PRISM_FISHEYE', width=1008, height=672,
                    params=[600, 600, 504, 336, .2, .1, .001, -.001, -.1, .3, .001, .001])
                xy = np.array([[.5, .5], [504.5, 336.5], [1007.5, 671.5]])
                rays = c.cam_from_img(xy)
                points = np.column_stack([rays, np.ones(3)])
                assert np.max(np.abs(g.forward(c.params, points) - xy)) < 1e-5
                assert np.max(np.abs(c.img_from_cam(points) - xy)) < 1e-5
                allowed_sum = int(np.load('/allowed/array.npy', allow_pickle=False).sum())
            save(out/'probe.json', dict(status='passed', denied=denied, allowed_sum=allowed_sum))
            return
        views, accepted = g.validate_predictor(args.data, config)
        if args.mode == 'features':
            import cv2
            assert cv2.__version__ == '4.11.0' and np.__version__ == '1.26.4'
            cv2.setNumThreads(4)
            cv2.setRNGSeed(42)
            sift = cv2.SIFT_create(**config['sift'])
            features, images = [], []
            for view in views:
                rgb = cv2.imread(str(args.data/'rgb'/view['name']), cv2.IMREAD_COLOR)
                if rgb is None or rgb.shape != (4032, 6048, 3):
                    raise ValueError('Native RGB dimensions mismatch')
                image = cv2.resize(rgb, (1008, 672), interpolation=cv2.INTER_AREA)
                kp, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
                xy = np.array([x.pt for x in kp], dtype=np.float64).reshape(-1, 2)
                if desc is None:
                    desc = np.empty((0, 128), dtype=np.float32)
                filename = Path(view['name']).stem
                np.savez(out/(filename+'.npz'), xy=xy, descriptors=desc)
                if not cv2.imwrite(str(out/(filename+'.png')), image):
                    raise OSError('Cannot write work preview')
                features.append((xy, desc))
                images.append(dict(name=view['name'], features=len(xy)))
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            edges, pair_counts = [], []
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = features[i][1], features[j][1]
                    def ratio_map(source, target):
                        if len(source) == 0 or len(target) < 2:
                            return {}
                        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(source, target, k=2)
                                if len(pair) == 2 for m, n in [pair] if m.distance < .8*n.distance}
                    ab, ba = ratio_map(a, b), ratio_map(b, a)
                    pair = [((i, x), (j, y)) for x, y in sorted(ab.items()) if ba.get(y) == x]
                    edges.extend(pair)
                    pair_counts.append(dict(images=[views[i]['name'], views[j]['name']],
                                            forward_ratio=len(ab), reverse_ratio=len(ba), reciprocal=len(pair)))
            save(out/'features.json', dict(status='completed_features', images=images, pairs=pair_counts,
                 edges=edges, accepted_predictor_assets=accepted,
                 files={p.name: g.sha256(p) for p in sorted(out.iterdir()) if p.is_file()}))
        elif args.mode == 'geometry':
            import pycolmap
            assert pycolmap.__version__ == '4.2.1'
            feature_summary = json.loads((args.features/'features.json').read_text())
            for name, digest in feature_summary['files'].items():
                if Path(name).name != name or g.sha256(args.features/name) != digest:
                    raise ValueError('Feature provenance mismatch')
            rays, camera_checks = [], []
            for view in views:
                with np.load(args.features/(Path(view['name']).stem+'.npz'), allow_pickle=False) as arrays:
                    view['xy'] = arrays['xy'].copy()
                params, _, _ = g.camera_arrays(view)
                camera = pycolmap.Camera(model='THIN_PRISM_FISHEYE', width=1008, height=672, params=params)
                xy = view['xy']
                rays.append(camera.cam_from_img(xy + .5))
                grid = np.array([[x+.5, y+.5] for y in np.linspace(0, 671, 9) for x in np.linspace(0, 1007, 12)])
                lifted = camera.cam_from_img(grid)
                points = np.column_stack([lifted, np.ones(len(lifted))])
                independent = float(np.max(np.abs(g.forward(params, points) - grid)))
                api = float(np.max(np.abs(camera.img_from_cam(points) - grid)))
                if not np.isfinite(lifted).all() or max(independent, api) >= 1e-5:
                    raise ValueError('Independent camera roundtrip gate failed')
                camera_checks.append(dict(name=view['name'], grid_points=108,
                    independent_forward_pixel_error=independent, API_pixel_error=api))
            groups, conflicts = g.components(feature_summary['edges'])
            tracks, rejected = [], {}
            for group in groups:
                observations = [(int(i), int(k)) for i, k in group]
                lifted = [rays[i][k] for i, k in observations]
                if not np.isfinite(lifted).all():
                    reason = 'unprojection_failed'
                    track = None
                else:
                    track, reason = g.triangulate(observations, views, lifted)
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                else:
                    tracks.append(track)
            anchors = g.select_anchors(tracks, views)
            for record in anchors:
                save(out/(Path(record['name']).stem+'-anchors.json'), record)
            save(out/'tracks.json', dict(accepted=tracks, conflicted_components=conflicts))
            save(out/'anchors.json', dict(protocol='terrace-sensor-diagnostic-plan-v1',
                status='passed_anchor_support' if all(a['support_passed'] for a in anchors) else 'blocked_anchor_support',
                camera_checks=camera_checks, candidate_components=len(groups), conflict_components=len(conflicts),
                accepted_tracks=len(tracks), rejected_tracks=rejected,
                per_image=[{k: v for k, v in a.items() if k != 'anchors'} for a in anchors],
                accepted_predictor_assets=accepted, poses_intrinsics_fixed=True,
                coordinates='actual measured SIFT indices, never reprojected coordinates',
                GT_read=False, model_loaded=False))
        else:
            raise ValueError('Unknown worker mode')
    finally:
        if 'torch' in sys.modules or 'depth_anything_v2' in sys.modules:
            raise RuntimeError('Model runtime unexpectedly imported during CPU preparation')
        save(out/'resources.json', dict(model_runtime_imported=False, wall_seconds=time.monotonic()-start,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            CPU_threads_max=4, GPU_used=False))


def runtime_mounts(python):
    info = json.loads(subprocess.check_output([str(python), '-I', '-c',
        'import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))'], text=True))
    paths = set()
    for p in info:
        paths.add(str(Path(p)))
        paths.add(str(Path(p).resolve()))
    mounts = []
    for p in sorted(paths):
        if p != '/usr':
            mounts += ['--ro-bind', p, p]
    return mounts


def sandbox(python, mounts, args):
    base = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
            '--clearenv', '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
            '--ro-bind', '/lib64', '/lib64', '--symlink', 'usr/bin', '/bin',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp']
    if Path('/etc/ld.so.cache').exists():
        base += ['--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache']
    for key, value in dict(PATH='/usr/bin:/bin', HOME='/tmp', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4').items():
        base += ['--setenv', key, value]
    return base + runtime_mounts(python) + mounts + [str(python), '-I', '/code/prepare_sensor_anchors_cpu.py'] + args


def rss_tree(pid):
    total = 0
    try:
        status = Path('/proc')/str(pid)/'status'
        total += next((int(l.split()[1]) for l in status.read_text().splitlines() if l.startswith('VmRSS:')), 0)
        children = Path('/proc')/str(pid)/'task'/str(pid)/'children'
        total += sum(rss_tree(int(x)) for x in children.read_text().split())
    except (OSError, StopIteration):
        pass
    return total


def execute(command, root, name, timeout=600, expected_failure=False):
    start = time.monotonic()
    peak = 0
    with (root/(name+'.log')).open('x') as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        reason = None
        while p.poll() is None:
            peak = max(peak, rss_tree(p.pid))
            if peak > 2*1024**2:
                reason = 'RSS_budget_exceeded'
            if time.monotonic()-start > timeout:
                reason = 'timeout'
            if reason:
                os.killpg(p.pid, signal.SIGKILL)
                break
            time.sleep(.05)
        code = p.wait()
    result = dict(returncode=code, wall_seconds=time.monotonic()-start, sampled_process_tree_RSS_KiB=peak,
                  stop_reason=reason, expected_failure=expected_failure)
    save(root/(name+'-process.json'), result)
    if reason or ((code == 0) == expected_failure):
        raise RuntimeError(name + ' process failed: ' + str(result))
    return result


def run(args):
    g = geometry()
    if g.sha256(args.plan) != PLAN_SHA:
        raise ValueError('Frozen protocol digest mismatch')
    plan = json.loads(args.plan.read_text())
    root = args.output.absolute()
    data = args.data.absolute()
    if (root.exists() or root == data or data in root.parents or root in data.parents
            or args.source.resolve() == root or args.source.resolve() in root.parents):
        raise ValueError('Fresh external output, disjoint from data, required')
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    final_status = 'failed_input_acceptance'
    try:
        config = dict(names=plan['dataset']['selected_names'], sift=plan['anchors']['sift'], inputs={})
        for row in plan['dataset']['inputs']:
            if row['role'] == 'predictor_rgb':
                rel = 'rgb/' + row['image']
            elif row['role'] == 'predictor_cameras':
                rel = 'cameras.json'
            else:
                continue
            config['inputs'][rel] = {k: row[k] for k in ['bytes', 'sha256']}
        # These checks read exact predictor files only, without decoding RGB.
        g.validate_predictor(data, config)
        bundle = root/'code'
        bundle.mkdir()
        sources = {'prepare_sensor_anchors_cpu.py': args.source/'scripts/prepare_sensor_anchors_cpu.py',
                   'sensor_geometry.py': args.source/'src/geosplat_r/sensor_geometry.py'}
        for name, source in sources.items():
            shutil.copyfile(source, bundle/name)
        save(root/'source-manifest.json', {name: g.sha256(bundle/name) for name in sources})
        save(root/'predictor-config.json', config)
        final_status = 'blocked_isolation'
        fixtures = root/'synthetic'
        fixtures.mkdir()
        # Generate fixture files using only the existing feature environment, no dataset reads.
        fixture_code = ('import numpy as np,cv2,sys; from pathlib import Path; p=Path(sys.argv[1]); '
                        'np.save(p/"array.npy",np.array([1,2,3])); '
                        'assert cv2.imwrite(str(p/"image.png"),np.full((3,3,3),7,dtype=np.uint8))')
        subprocess.run([str(args.python), '-I', '-c', fixture_code, str(fixtures)], check=True)
        hidden = root/'unmounted-label.npy'
        hidden.write_bytes((fixtures/'array.npy').read_bytes())
        hidden_image = root/'unmounted-label.png'
        shutil.copyfile(fixtures/'image.png', hidden_image)
        forbidden = [str(hidden), str(hidden_image), '/evaluator/label.npy', '/archive/input.7z', '/metadata/images.txt',
                     '/inputs/rgb/other.JPG']
        save(root/'probe-config.json', dict(forbidden=forbidden))
        common = ['--ro-bind', str(bundle), '/code']
        for index, (python, mode) in enumerate([(args.python, 'probe'), (args.camera_python, 'probe-camera'),
                                                (args.python, 'probe')]):
            if index == 2:
                original = hidden.read_bytes()
                hidden.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                hidden_image.write_bytes(hidden_image.read_bytes() + b'changed evaluator-only metadata')
            out = root/('preflight-'+str(index))
            out.mkdir()
            mounts = common + ['--ro-bind', str(fixtures), '/allowed', '--ro-bind', str(root/'probe-config.json'),
                              '/config.json', '--bind', str(out), '/output']
            command = sandbox(python, mounts, ['--mode', mode, '--config', '/config.json', '--output', '/output/result'])
            execute(command, root, 'preflight-'+str(index), timeout=60)
            if index == 0:
                before = g.sha256(out/'result/probe.json')
                execute(command, root, 'preflight-overwrite-rejected', timeout=60, expected_failure=True)
                if g.sha256(out/'result/probe.json') != before:
                    raise RuntimeError('Prior output changed')
        if (root/'preflight-0/result/probe.json').read_bytes() != (root/'preflight-2/result/probe.json').read_bytes():
            raise RuntimeError('Synthetic output changed with evaluator-only label')
        save(root/'preflight.json', dict(status='passed', namespace=True, native_OpenCV_denial=True,
             pycolmap_full_model=True, unmounted_label_change_invariant=True, overwrite_refused=True))
        final_status = 'failed_CPU_preparation'
        mounts = common + ['--ro-bind', str(root/'predictor-config.json'), '/config.json']
        for rel in sorted(config['inputs']):
            mounts += ['--ro-bind', str(data/rel), '/inputs/'+rel]
        remaining = lambda: max(0, 600-(time.monotonic()-start))
        for mode, python in [('features', args.python), ('geometry', args.camera_python)]:
            stage = root/(mode+'-stage')
            stage.mkdir()
            stage_mounts = mounts + ['--bind', str(stage), '/output']
            worker_args = ['--mode', mode, '--config', '/config.json', '--data', '/inputs', '--output', '/output/result']
            if mode == 'geometry':
                stage_mounts += ['--ro-bind', str(root/'features-stage/result'), '/features']
                worker_args += ['--features', '/features']
            execute(sandbox(python, stage_mounts, worker_args), root, mode, timeout=remaining())
            used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            if used > 2*1024**3:
                raise RuntimeError('Output budget exceeded')
        final_status = json.loads((root/'geometry-stage/result/anchors.json').read_text())['status']
        for name, digest in json.loads((root/'source-manifest.json').read_text()).items():
            if g.sha256(bundle/name) != digest:
                raise RuntimeError('Executed source changed')
        g.validate_predictor(data, config)
    finally:
        save(root/'completion.json', dict(status=final_status, wall_seconds=time.monotonic()-start,
             plan_sha256=PLAN_SHA, GPU_used=False, weights_loaded=False, GT_read=False,
             error=None if sys.exc_info()[0] is None else str(sys.exc_info()[1])))
    print((root/'completion.json').read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['run', 'features', 'geometry', 'probe', 'probe-camera'], default='run')
    for key in ['source', 'data', 'plan', 'output', 'python', 'camera-python', 'config', 'features']:
        ap.add_argument('--'+key, type=Path)
    args = ap.parse_args()
    required = ['source', 'data', 'plan', 'output', 'python', 'camera_python'] if args.mode == 'run' else ['config', 'output']
    for name in required:
        if getattr(args, name) is None:
            ap.error('Missing explicit location: '+name)
    if args.mode == 'run':
        run(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
