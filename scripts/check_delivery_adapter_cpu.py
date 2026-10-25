"""CPU-only source-bound admission/isolation check; no prediction/evaluation CLI."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def worker(config, output):
    import os
    import cv2
    import numpy as np
    io = module(Path('/code/delivery_relative_io.py'), 'delivery_io')
    cfg = json.loads(config.read_text())
    io.admit_predictor('/inputs', cfg['predictor'])
    denied = []
    for path in cfg['forbidden']:
        for name, read in [('python',lambda:Path(path).read_bytes()),
                           ('os',lambda:os.close(os.open(path,os.O_RDONLY))),
                           ('numpy',lambda:np.fromfile(path,dtype=np.uint8)),
                           ('opencv',lambda:cv2.imread(path))]:
            try:
                value = read()
            except OSError:
                denied.append([path,name])
                continue
            if name == 'opencv' and value is None:
                denied.append([path,name])
                continue
            raise AssertionError('Forbidden input visible: '+path)
    if Path('/dev/dxg').exists() or 'torch' in sys.modules:
        raise AssertionError('CPU check exposed GPU/model runtime')
    io.rd.write_json(output/'result.json',dict(status='passed_CPU_admission_isolation',
        predictor_assets=4,forbidden_reads_refused=denied,GT_read=False,model_loaded=False))


def run(args):
    source = args.source.resolve()
    io = module(source/'src/geosplat_r/delivery_relative_io.py','delivery_io')
    paths = {n:source/'src/geosplat_r'/n for n in ['delivery_relative_io.py','relative_io.py','relative_diagnostics.py']}
    paths['check_delivery_adapter_cpu.py'] = source/'scripts/check_delivery_adapter_cpu.py'
    paths['prepare_sensor_anchors_cpu.py'] = source/'scripts/prepare_sensor_anchors_cpu.py'
    hashes = {n:io.rd.digest(p) for n,p in paths.items()}
    predictor, _ = io.load_contexts(args.plan,args.input_manifest,args.base_plan,args.review_plan,hashes)
    output = args.output.resolve()
    protected = [source,args.data.resolve()] + [p.resolve() for p in [args.plan,args.input_manifest,args.base_plan,args.review_plan]]
    if output.exists() or any(output==p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Fresh disjoint output required')
    output.mkdir(parents=True,exist_ok=False)
    bundle = output/'code'
    bundle.mkdir()
    for n,p in paths.items():
        shutil.copyfile(p,bundle/n)
    # The unchanged runtime helper runs its fixed /code filename; mount this checker there.
    import cv2
    import numpy as np
    sentinel = output/'host-only-sentinel.png'
    if not cv2.imwrite(str(sentinel),np.full((2,2,3),71,dtype=np.uint8)):
        raise OSError('Sentinel creation failed')
    forbidden = [str(sentinel),'/archive/data.7z','/metadata/images.txt','/inputs/other.JPG','/weights/model.pth']
    forbidden += [str(args.data.resolve().parent.parent/'evaluator/labels'/n) for n in io.NAMES]
    io.rd.write_json(output/'config.json',dict(predictor=predictor,forbidden=forbidden))
    io.rd.write_json(output/'source-manifest.json',hashes)
    result = output/'worker'
    result.mkdir()
    mounts = ['--ro-bind',str(bundle),'/code',
              '--ro-bind',str(bundle/'check_delivery_adapter_cpu.py'),'/code/prepare_sensor_anchors_cpu.py',
              '--ro-bind',str(output/'config.json'),'/config.json','--bind',str(result),'/output']
    for rel in predictor['inputs']:
        mounts += ['--ro-bind',str(args.data/rel),'/inputs/'+rel]
    rt = module(source/'scripts/prepare_sensor_anchors_cpu.py','cpu_runtime')
    rt.execute(rt.sandbox(args.python,mounts,['--worker','--config','/config.json','--output','/output']),
               output,'admit-isolate',timeout=60)
    status = json.loads((result/'result.json').read_text())
    io.rd.write_json(output/'completion.json',dict(status=status['status'],GT_read=False,
                                                  GPU_used=False,formal_predictions=0))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    for flag in ['source','plan','input-manifest','base-plan','review-plan','data','python','output','config']:
        parser.add_argument('--'+flag,type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.config,args.output)
    else:
        if any(getattr(args,n) is None for n in ['source','plan','input_manifest','base_plan','review_plan','data','python','output']):
            parser.error('Explicit source, plans, manifest, data, Python and fresh output required')
        print(json.dumps(run(args)))
