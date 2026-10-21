"""Synthetic-only CPU acceptance for the frozen anchor-free ordinal evaluator.

No real-data mode, raw ETH3D loader, model, weights or GPU entry is supplied.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time


def load_module(name, sibling, repo_relative):
    path = Path(__file__).with_name(sibling)
    if not path.exists():
        path = Path(__file__).resolve().parents[1]/repo_relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def core():
    return load_module('relative_core', 'relative_diagnostics.py', 'src/geosplat_r/relative_diagnostics.py')


def runtime():
    return load_module('cpu_runtime', 'prepare_sensor_anchors_cpu.py', 'scripts/prepare_sensor_anchors_cpu.py')


def synthetic_config(fixtures, sources):
    rd = core()
    fixtures = Path(fixtures)
    m = json.loads((fixtures/'manifest.json').read_text())
    if (fixtures.is_symlink() or m.get('kind') != 'synthetic-relative-fixture-v1'
            or m.get('names') != ['synthetic-0', 'synthetic-1', 'synthetic-2']
            or m.get('native_shape') != [576, 864] or m.get('work_shape') != [96, 144]
            or len(m.get('records', [])) != 3):
        raise ValueError('Only explicitly tagged, small synthetic fixtures accepted')
    if {p.name for p in (fixtures/'rgb').iterdir()} != {'0.npy','1.npy','2.npy'}:
        raise ValueError('Synthetic RGB inventory must contain exactly three inputs')
    if {p.name for p in (fixtures/'labels').iterdir()} != {'0.npy','1.npy','2.npy'}:
        raise ValueError('Synthetic label inventory must contain exactly three inputs')
    for i, row in enumerate(m['records']):
        if row['rgb_file'] != str(i)+'.npy' or row['label_file'] != str(i)+'.npy':
            raise ValueError('Unexpected synthetic path')
        p = fixtures/'rgb'/row['rgb_file']
        if p.is_symlink() or rd.digest(p) != row['rgb_sha256']:
            raise ValueError('Synthetic input content drift')
    if any(p.is_symlink() for p in fixtures.rglob('*')):
        raise ValueError('Linked fixture assets forbidden')
    return dict(**m, source_sha256=sources,
        input_sha256={r['rgb_file']: r['rgb_sha256'] for r in m['records']})


def worker(args):
    rd = core()
    import numpy as np
    if np.__version__ != '1.26.4':
        raise ValueError('Frozen RNG version required')
    start = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    reads = []
    try:
        config = json.loads(args.config.read_text())
        if (config.get('kind') != 'synthetic-relative-fixture-v1'
                or config.get('names') != ['synthetic-0','synthetic-1','synthetic-2']
                or config.get('native_shape') != [576,864] or config.get('work_shape') != [96,144]
                or len(config.get('records',[])) != 3
                or any(row.get('rgb_file') != str(i)+'.npy' or
                       (args.mode == 'evaluate' and row.get('label_file') != str(i)+'.npy')
                       for i,row in enumerate(config['records']))):
            raise ValueError('Worker also requires small synthetic-only schema')
        if args.mode == 'predict-probe':
            import cv2
            refused = []
            for path in config['forbidden']:
                for method in ['python', 'os', 'numpy', 'opencv']:
                    import os
                    try:
                        if method == 'python':
                            Path(path).read_bytes()
                        elif method == 'os':
                            fd = os.open(path, os.O_RDONLY)
                            os.close(fd)
                        elif method == 'numpy':
                            np.load(path, allow_pickle=False)
                        elif cv2.imread(path) is not None:
                            raise RuntimeError('Native OpenCV read forbidden input')
                        else:
                            refused.append([path, method])
                            continue
                    except OSError:
                        refused.append([path, method])
                    else:
                        raise RuntimeError('Forbidden path was readable: '+path)
            pred_root = out/'predictions'
            pred_root.mkdir()
            records = []
            for name, row in zip(config['names'], config['records']):
                p = Path('/input')/row['rgb_file']
                if rd.digest(p) != row['rgb_sha256']:
                    raise ValueError('Synthetic RGB drift')
                rgb = np.load(p, allow_pickle=False)
                if rgb.dtype != np.uint8 or rgb.shape != (96, 144, 3):
                    raise ValueError('Only toy RGB accepted')
                # A fixed arithmetic fixture function, not a trained model.
                raw = rgb[..., 0].astype(np.float32)/256
                target = pred_root/row['rgb_file']
                np.save(target, raw)
                records.append(dict(name=name, file=target.name, sha256=rd.digest(target)))
            rd.write_json(pred_root/'seal.json', dict(status='sealed_all_predictions', plan_sha256=rd.PLAN_SHA,
                source_sha256=config['source_sha256'], input_sha256=config['input_sha256'], predictions=records))
            rd.write_json(out/'probe.json', dict(status='passed', forbidden_reads_refused=refused,
                synthetic_arithmetic_only=True, model_loaded=False))
        elif args.mode == 'evaluate':
            seal = rd.verify_prediction_seal(args.predictions, config['names'],
                                            config['source_sha256'], config['input_sha256'])
            predictions = []
            for row in seal['predictions']:
                q = np.load(args.predictions/row['file'], allow_pickle=False)
                if q.shape != (96, 144) or q.dtype != np.float32:
                    raise ValueError('Synthetic prediction schema mismatch')
                predictions.append(q)
            # All three seals, payload digests, dtypes and shapes precede the first label access.
            reports = []
            for i, row in enumerate(config['records']):
                label = Path('/labels')/row['label_file']
                reads.append(label.name)
                if label.is_symlink() or label.stat().st_size != 576*864*4+128 or rd.digest(label) != row['label_sha256']:
                    raise ValueError('Synthetic reference drift')
                gt = np.load(label, allow_pickle=False)
                if gt.dtype != np.float32 or gt.shape != (576, 864):
                    raise ValueError('Only small synthetic labels accepted; no real-data mode')
                report, nodes, pairs = rd.evaluate_arrays(gt, predictions[i], i)
                rd.write_json(out/('synthetic-%d.json'%i), report)
                payload = dict(native_indices=nodes['indices'], reference_Z=nodes['z'], cell=nodes['cell'])
                for kind in ('local', 'cross'):
                    payload[kind+'_candidate_pairs'] = np.concatenate([c['candidates'] for c in pairs[kind]])
                    payload[kind+'_eligible'] = np.concatenate([c['eligible'] for c in pairs[kind]])
                    payload[kind+'_source_cell'] = np.concatenate([np.full(len(c['candidates']),c['cell']) for c in pairs[kind]])
                np.savez(out/('synthetic-%d-pairs.npz'%i), **payload)
                reports.append(report)
            rd.write_json(out/'assessment.json', dict(synthetic_only=True, **rd.assess(reports)))
        else:
            raise ValueError('Worker mode required')
    finally:
        rd.write_json(out/'reference-access.json', dict(synthetic_label_attempts=reads, real_data_read=False))
        if 'torch' in sys.modules or 'depth_anything_v2' in sys.modules:
            raise RuntimeError('Model runtime imported during CPU acceptance')
        rd.write_json(out/'resources.json', dict(wall_seconds=time.monotonic()-start,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            model_runtime_imported=False, GPU_used=False))


def command(python, mounts, arguments):
    rt = runtime()
    cmd = rt.sandbox(python, mounts, arguments)
    index = cmd.index('/code/prepare_sensor_anchors_cpu.py')
    cmd[index] = '/code/check_relative_evaluator_cpu.py'
    return cmd


def run(args):
    rd, rt = core(), runtime()
    if rd.digest(args.plan) != rd.PLAN_SHA:
        raise ValueError('Frozen relative protocol content drift')
    root, source, fixtures = args.output.absolute(), args.source.resolve(), args.fixtures.resolve()
    if (root.exists() or source == root or source in root.parents or fixtures == root
            or fixtures in root.parents or root in fixtures.parents):
        raise ValueError('Fresh external disjoint output required')
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = 'failed_synthetic_CPU_acceptance'
    try:
        bundle = root/'code'
        bundle.mkdir()
        sources = {'relative_diagnostics.py': source/'src/geosplat_r/relative_diagnostics.py',
                   'check_relative_evaluator_cpu.py': source/'scripts/check_relative_evaluator_cpu.py',
                   'prepare_sensor_anchors_cpu.py': source/'scripts/prepare_sensor_anchors_cpu.py'}
        for name, path in sources.items():
            shutil.copyfile(path, bundle/name)
        hashes = {name: rd.digest(bundle/name) for name in sources}
        rd.write_json(root/'source-manifest.json', hashes)
        config = synthetic_config(fixtures, hashes)
        altered = root/'altered-labels'
        altered.mkdir()
        # Different valid synthetic label bytes, never mounted in either predictor.
        for row in config['records']:
            data = (fixtures/'labels'/row['label_file']).read_bytes()
            (altered/row['label_file']).write_bytes(data[:-1]+bytes([data[-1]^1]))
        config['forbidden'] = [str(fixtures/'labels/0.npy'), str(fixtures/'sentinel.png'),
            str(altered/'0.npy'), '/labels/0.npy', '/archive/data.7z', '/model/weights.pth', '/input/extra.png']
        # Predictor gets no label values/digests; only input records and denial targets.
        narrow = {k: config[k] for k in ['kind','names','native_shape','work_shape','source_sha256','input_sha256','forbidden']}
        narrow['records'] = [{k: row[k] for k in ['rgb_file','rgb_sha256']} for row in config['records']]
        rd.write_json(root/'predictor-config.json', narrow)
        rd.write_json(root/'evaluator-config.json', config)
        base = ['--ro-bind', str(bundle), '/code']
        for index in [0, 1]:
            stage = root/('predictor-%d'%index)
            stage.mkdir()
            if index == 1:
                # Change evaluator-only files after the first output was sealed.
                for p in altered.iterdir():
                    data = p.read_bytes()
                    p.write_bytes(data[:-2]+bytes([data[-2]^1])+data[-1:])
            mounts = base+['--ro-bind',str(fixtures/'rgb'),'/input',
                '--ro-bind',str(root/'predictor-config.json'),'/config.json','--bind',str(stage),'/output']
            cmd = command(args.python,mounts,['--mode','predict-probe','--config','/config.json','--output','/output/result'])
            rt.execute(cmd,root,'predictor-%d'%index,timeout=60)
            if index == 0:
                before = rd.digest(stage/'result/predictions/seal.json')
                rt.execute(cmd,root,'existing-output-refused',timeout=60,expected_failure=True)
                if rd.digest(stage/'result/predictions/seal.json') != before:
                    raise RuntimeError('Prior prediction seal changed')
        a,b = [root/('predictor-%d/result/predictions'%i) for i in [0,1]]
        if {p.name:rd.digest(p) for p in a.iterdir()} != {p.name:rd.digest(p) for p in b.iterdir()}:
            raise RuntimeError('Prediction or input seal changed with evaluator-only labels')
        for index, bad in [(0,False),(1,True)]:
            stage=root/('evaluator-%d'%index)
            stage.mkdir()
            predictions=a
            if bad:
                predictions=root/'corrupt-third-prediction'
                shutil.copytree(a,predictions)
                target=predictions/'2.npy'
                data=target.read_bytes();target.write_bytes(data[:-1]+bytes([data[-1]^1]))
            mounts=base+['--ro-bind',str(predictions),'/predictions','--ro-bind',str(fixtures/'labels'),'/labels',
                '--ro-bind',str(root/'evaluator-config.json'),'/config.json','--bind',str(stage),'/output']
            cmd=command(args.python,mounts,['--mode','evaluate','--config','/config.json',
                '--predictions','/predictions','--output','/output/result'])
            rt.execute(cmd,root,'evaluator-%d'%index,timeout=600,expected_failure=bad)
        audit=json.loads((root/'evaluator-1/result/reference-access.json').read_text())
        if audit['synthetic_label_attempts']:
            raise RuntimeError('A label was opened before third prediction verification')
        if json.loads((root/'evaluator-0/result/assessment.json').read_text())['status'] != 'passed_ordinal_screen':
            raise RuntimeError('Perfect synthetic fixture failed closed-form acceptance')
        if any(rd.digest(bundle/name) != h for name,h in hashes.items()):
            raise RuntimeError('Executed source drift')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > 1024**3:
            raise RuntimeError('Synthetic output budget exceeded')
        status='passed_synthetic_CPU_acceptance'
    finally:
        rd.write_json(root/'completion.json',dict(status=status,synthetic_only=True,real_GT_read=False,
            model_loaded=False,GPU_used=False,plan_sha256=rd.PLAN_SHA,wall_seconds=time.monotonic()-start,
            error=None if sys.exc_info()[0] is None else str(sys.exc_info()[1])))
    print((root/'completion.json').read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['run','predict-probe','evaluate'],default='run')
    for key in ['source','plan','fixtures','python','output','config','predictions']:
        ap.add_argument('--'+key,type=Path)
    args=ap.parse_args()
    required=['source','plan','fixtures','python','output'] if args.mode=='run' else ['config','output']
    for key in required:
        if getattr(args,key) is None:ap.error('Explicit location required: '+key)
    if args.mode=='run':run(args)
    else:worker(args)


if __name__=='__main__':
    main()
