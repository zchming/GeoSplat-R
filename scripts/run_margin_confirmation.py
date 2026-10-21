"""One frozen, previously unopened courtyard ambiguity confirmation.

Reuse the historical predictor and ordinal numerical core without modifying them.
Prediction, descriptor and evaluation stages have separate process mounts.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time

PLAN_SHA = '1cc7e92c2dad2bdda859ca4f6c7d9a216116f56f8ad21f437a8ea657387f41f2'
FILES = {'run_margin_confirmation.py': 'scripts/run_margin_confirmation.py',
         'run_relative_diagnostic.py': 'scripts/run_relative_diagnostic.py',
         'prepare_sensor_anchors_cpu.py': 'scripts/prepare_sensor_anchors_cpu.py',
         'relative_io.py': 'src/geosplat_r/relative_io.py',
         'relative_diagnostics.py': 'src/geosplat_r/relative_diagnostics.py',
         'ordinal_margin.py': 'src/geosplat_r/ordinal_margin.py'}


def load(root, name):
    path = Path(root)/name
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def role_config(context, stage, sources):
    if (context['confirmation_plan_sha256'] != PLAN_SHA
            or len(context['names']) != 3 or len(set(context['names'])) != 3
            or context['native_wh'] != [6048, 4032] or context['work_wh'] != [1008, 672]):
        raise ValueError('Frozen scene identity/grid mismatch')
    for name in context['names']:
        if Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('dslr_images/'):
            raise ValueError('Unsafe image name')
    common = {k: context[k] for k in ['names', 'native_wh', 'work_wh', 'input_sha256']}
    common.update(source_sha256=sources, confirmation_plan_sha256=PLAN_SHA)
    if stage in ['preflight', 'predict']:
        common['model'] = context['model']
        if stage == 'predict':
            common['inputs'] = context['inputs']
    elif stage == 'evaluate':
        common['labels'] = context['labels']
    elif stage != 'describe':
        raise ValueError('Explicit role required')
    return common


def verify_descriptors(root, config, arrays, margin, prediction_root='/predictions'):
    import numpy as np
    root = Path(root)
    seal = json.loads((root/'seal.json').read_text())
    if (seal['confirmation_plan_sha256'] != PLAN_SHA or seal['names'] != config['names']
            or seal['source_sha256'] != config['source_sha256']
            or seal['input_sha256'] != config['input_sha256']
            or len(seal['records']) != 3
            or {p.name for p in root.iterdir()} != {'seal.json', '0.npy', '1.npy', '2.npy'}
            or root.is_symlink() or any(p.is_symlink() for p in root.iterdir())):
        raise ValueError('All three descriptor identities required')
    refs = []
    for i, (row, raw) in enumerate(zip(seal['records'], arrays)):
        path = root/('%d.npy'%i)
        if (row['file'] != path.name or margin.digest(path) != row['sha256']
                or row['raw_sha256'] != margin.digest(Path(prediction_root)/('%d.npy'%i))):
            raise ValueError('Descriptor/raw digest mismatch')
        values = np.load(path, allow_pickle=False)
        expected, counts = domain_reference(raw, margin)
        if (values.dtype != np.float64 or not np.array_equal(values, expected)
                or row['counts'] != counts):
            raise ValueError('Descriptor values/counts mismatch')
        refs.append(values)
    return refs, all(x['passed'] for x in seal['records'])


def domain_reference(raw, margin):
    values, invalid = margin.sorted_reference(raw[2:670, 2:1006])
    return values, dict(domain_pixels=670672, valid_pixels=len(values), invalid_pixels=invalid,
                        valid_fraction=len(values)/670672)


def assess(images):
    import numpy as np
    if len(images) != 3 or [x['image_index'] for x in images] != [0, 1, 2]:
        raise ValueError('Exactly three ordered confirmation images required')
    if not all(x['support']['passed'] for x in images):
        return dict(status='blocked_error_support')
    image_gates = []
    for row in images:
        risk = row['margin']['risk']
        image_gates.append(bool(risk['0.5'] <= .8*risk['1.0']
            and risk['0.5'] <= .9*row['shuffle_mean']['0.5']
            and all(s['risk']['0.5'] <= s['risk']['1.0']+1e-12
                    for s in row['margin']['strata'].values())))
    risks = {str(c): float(np.mean([x['margin']['risk'][str(c)] for x in images]))
             for c in [.25, .5, .75, 1.]}
    scene_gate = risks['0.5'] <= .75*risks['1.0']
    return dict(status='passed_unopened_scene_margin_screen' if all(image_gates) and scene_gate
                else 'failed_confirmation_margin_gate', image_gates=image_gates,
                scene_gate=bool(scene_gate), risk=risks)


def cpu_stage(stage, cfg, output, io, margin, access):
    import numpy as np
    arrays = io.sealed_predictions('/predictions', cfg)
    if stage == 'describe':
        dest = output/'descriptors'
        dest.mkdir()
        records = []
        for i, raw in enumerate(arrays):
            values, counts = domain_reference(raw, margin)
            filename = '%d.npy'%i
            np.save(dest/filename, values)
            records.append(dict(file=filename, sha256=margin.digest(dest/filename),
                raw_sha256=margin.digest(Path('/predictions')/filename), counts=counts,
                passed=counts['valid_fraction'] >= .99))
        seal = {k: cfg[k] for k in ['names', 'source_sha256', 'input_sha256', 'confirmation_plan_sha256']}
        io.rd.write_json(dest/'seal.json', dict(seal, records=records))
        return dict(status='passed_descriptors' if all(x['passed'] for x in records)
                    else 'blocked_prediction_descriptor_support')
    refs, supported = verify_descriptors('/descriptors', cfg, arrays, margin)
    if not supported:
        return dict(status='blocked_prediction_descriptor_support')
    base = output/'baseline'
    base.mkdir()
    io.evaluate_sealed('/predictions', '/labels', cfg, base, access)
    prepared, images = [], []
    for i, raw in enumerate(arrays):
        image = margin.prepare_image(raw, base/('image-%d-evaluator-only-pairs.npz'%i), base/('image-%d.json'%i))
        prepared.append(image)
        images.append(dict(image_index=i, name=cfg['names'][i], support=image['support']))
    if all(x['support']['passed'] for x in images):
        for i, (image, row, ref) in enumerate(zip(prepared, images, refs)):
            features = margin.empirical_cdf(ref, image['q'])
            row.update(margin=margin.risk_curves(image, features),
                       constant=margin.risk_curves(image, features, 'constant'),
                       oracle=margin.risk_curves(image, features, 'oracle'), shuffled=[])
            for seed in range(42, 62):
                values = margin.shuffled_features(features, image['valid'], seed, i)
                row['shuffled'].append(dict(seed=seed, **margin.risk_curves(image, values)))
            row['shuffle_mean'] = {str(c): float(np.mean([x['risk'][str(c)] for x in row['shuffled']]))
                                   for c in [.25, .5, .75, 1.]}
    for i, row in enumerate(images):
        io.rd.write_json(output/('image-%d.json'%i), row)
    return assess(images)


def worker(args):
    root = Path('/code')
    io = load(root, 'relative_io.py')
    gpu = load(root, 'run_relative_diagnostic.py')
    margin = load(root, 'ordinal_margin.py')
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(exist_ok=False)
    access = []
    started = time.monotonic()
    try:
        refused = gpu.denied_reads(cfg['forbidden'])
        if args.stage in ['preflight', 'predict']:
            gpu.gpu_worker(cfg, args.output, args.stage == 'preflight')
            result = dict(status='passed_'+args.stage)
        else:
            if Path('/dev/dxg').exists() or 'torch' in sys.modules:
                raise RuntimeError('CPU evaluator/descriptor exposed GPU')
            result = cpu_stage(args.stage, cfg, args.output, io, margin, access)
        io.rd.write_json(args.output/'result.json', dict(result, forbidden_reads_refused=refused))
    finally:
        io.rd.write_json(args.output/'reference-access.json', dict(attempts=access))
        io.rd.write_json(args.output/'resources.json', dict(wall_seconds=time.monotonic()-started,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def run(args):
    source = args.source.resolve()
    margin = load(source/'src/geosplat_r', 'ordinal_margin.py')
    if margin.digest(args.plan) != PLAN_SHA:
        raise ValueError('Frozen confirmation plan drift')
    plan = json.loads(args.plan.read_text())
    margin_plan_path = source/'configs/ordinal-error-margin-plan.json'
    if margin.digest(margin_plan_path) != plan['frozen_margin_plan_sha256']:
        raise ValueError('Inherited margin protocol drift')
    margin_plan = json.loads(margin_plan_path.read_text())
    base_path = source/'configs/terrace-relative-diagnostic-plan.json'
    if margin.digest(base_path) != margin_plan['inherited_protocol']['terrace_plan_sha256']:
        raise ValueError('Inherited model protocol drift')
    model_spec = json.loads(base_path.read_text())['prediction']['model']
    for name, key in [('relative_diagnostics.py', 'numeric_core_sha256'), ('ordinal_margin.py', 'margin_core_sha256')]:
        if margin.digest(source/FILES[name]) != plan[key]:
            raise ValueError('Frozen numerical source drift')
    required = {'preflight': ['model_source', 'weights'], 'predict': ['data', 'model_source', 'weights'],
                'describe': ['predictions'], 'evaluate': ['predictions', 'descriptors', 'labels']}[args.stage]
    all_inputs = ['data', 'model_source', 'weights', 'predictions', 'descriptors', 'labels']
    if any(getattr(args, k) is None for k in required) or any(getattr(args, k) is not None for k in all_inputs if k not in required):
        raise ValueError('Explicit stage inputs, no cross-role arguments required')
    output = args.output.resolve()
    protected = [source, args.config.resolve(), args.plan.resolve()]+[getattr(args, k).resolve() for k in required]
    if output.exists() or any(output == x or x in output.parents or output in x.parents for x in protected):
        raise ValueError('Fresh disjoint output required')
    hashes = {name: margin.digest(source/path) for name, path in FILES.items()}
    hashes['confirmation-plan.json'] = PLAN_SHA
    context = json.loads(args.config.read_text())
    if context['model'] != model_spec:
        raise ValueError('Frozen model specification drift')
    cfg = role_config(context, args.stage, hashes)
    if args.stage == 'predict':
        load(source/'src/geosplat_r', 'relative_io.py').admit_predictor(args.data, cfg)
    output.mkdir(exist_ok=False)
    bundle = output/'code'
    bundle.mkdir()
    for name, path in FILES.items():
        shutil.copyfile(source/path, bundle/name)
    sentinel = output/'host-only.txt'
    sentinel.write_text('unmounted')
    cfg['forbidden'] = [str(sentinel), '/raw/archive.7z']
    if args.stage in ['predict', 'preflight', 'describe']:
        cfg['forbidden'] += ['/labels/'+name for name in cfg['names']]
    if args.stage in ['describe', 'evaluate']:
        cfg['forbidden'] += ['/weights/small.pth', '/model-source/depth_anything_v2/dpt.py']
    margin.rd.write_json(output/'worker-config.json', cfg)
    margin.rd.write_json(output/'source-manifest.json', hashes)
    rt = load(bundle, 'prepare_sensor_anchors_cpu.py')
    gpu = load(bundle, 'run_relative_diagnostic.py')
    mounts = ['--ro-bind', str(bundle), '/code', '--ro-bind', str(output/'worker-config.json'), '/config.json',
              '--bind', str(output), '/output']
    destinations = dict(data='/inputs', predictions='/predictions', descriptors='/descriptors', labels='/labels',
                        model_source='/model-source', weights='/weights/small.pth')
    for key in required:
        mounts += ['--ro-bind', str(getattr(args, key).resolve()), destinations[key]]
    sampler = None
    if args.stage in ['preflight', 'predict']:
        mounts += gpu.device_mounts()
        sampler = gpu.BoardSampler(output/'board-memory.csv')
    command = rt.sandbox(args.python, mounts, ['--worker', '--stage', args.stage, '--config', '/config.json', '--output', '/output/result'])
    command[command.index('/code/prepare_sensor_anchors_cpu.py')] = '/code/run_margin_confirmation.py'
    try:
        if sampler:
            sampler.start()
        rt.execute(command, output, args.stage, timeout=600)
    finally:
        if sampler:
            board = sampler.stop()
            margin.rd.write_json(output/'board-memory.json', board)
            if board['sampled_whole_board_peak_MiB'] is None or board['sampled_whole_board_peak_MiB'] >= 12288:
                raise RuntimeError('GPU measurement absent or VRAM budget exceeded')
    if any(margin.digest(bundle/name) != value for name, value in hashes.items() if name in FILES):
        raise ValueError('Executed source drift')
    if sum(x.stat().st_size for x in output.rglob('*') if x.is_file()) > 256*1024**2:
        raise RuntimeError('Output budget exceeded; preserve and stop')
    print((output/'result/result.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['preflight', 'predict', 'describe', 'evaluate'], required=True)
    parser.add_argument('--worker', action='store_true')
    for key in ['source', 'plan', 'config', 'python', 'output', 'data', 'labels', 'predictions', 'descriptors', 'model-source', 'weights']:
        parser.add_argument('--'+key, type=Path)
    args = parser.parse_args()
    required = ['config', 'output'] if args.worker else ['source', 'plan', 'config', 'python', 'output']
    if any(getattr(args, k) is None for k in required):
        parser.error('Explicit locations required')
    if args.worker:
        worker(args)
    else:
        run(args)
