"""CPU margin stages: metadata admission, descriptors, or later released evaluation."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_module(source):
    return load(source/'src/geosplat_r/ordinal_margin.py','ordinal_margin')


def role_rows(plan,manifest,stage,plan_sha256):
    if manifest.get('protocol')!='ordinal-margin-asset-locations-v1' or manifest.get('plan_sha256')!=plan_sha256:
        raise ValueError('Location manifest identity mismatch')
    expected={r['scene']+'/'+r['relative_path']:r for r in plan['input_assets']}
    locations={r['key']:r['path'] for r in manifest['files']}
    if (len(locations)!=len(manifest['files']) or set(locations)!=set(expected)
            or len(set(locations.values()))!=len(locations)):
        raise ValueError('Exact22 explicit asset locations required')
    rows=[]
    for key,spec in expected.items():
        path=Path(locations[key])
        if not path.is_absolute() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError('Absolute non-symlink asset locations required')
        if stage=='describe' and spec['role']!='scorer':continue
        rows.append(dict(key=key,path=str(path),role=spec['role'],bytes=spec['bytes'],sha256=spec['sha256']))
    if len({r['path'] for r in rows})!=len(rows):raise ValueError('Asset location aliases')
    return rows


def denied_reads(paths):
    import os
    import numpy as np
    import cv2
    refused=[]
    for path in paths:
        for method,read in [('python',lambda:Path(path).read_bytes()),
                            ('os',lambda:os.close(os.open(path,os.O_RDONLY))),
                            ('numpy',lambda:np.fromfile(path,dtype=np.uint8)),
                            ('opencv',lambda:cv2.imread(path))]:
            try:value=read()
            except OSError:refused.append([path,method]);continue
            if method=='opencv' and value is None:refused.append([path,method]);continue
            raise RuntimeError('Forbidden input readable: '+path)
    return refused


def worker(args):
    m=load(Path('/code/ordinal_margin.py'),'margin');start=time.monotonic();access=[]
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        cfg=json.loads(args.config.read_text())
        if cfg['plan_sha256']!=m.PLAN_SHA or Path('/dev/dxg').exists() or 'torch' in sys.modules:
            raise ValueError('CPU protocol/device boundary mismatch')
        refused=denied_reads(cfg['forbidden'])
        if args.stage=='probe':result=dict(status='passed_CPU_probe')
        elif args.stage=='validate':result=m.inspect_assets(cfg['files'])
        elif args.stage=='describe':
            records=m.descriptors(cfg['files'],args.output/'descriptors',cfg['source_sha256'])
            result=dict(status='passed_descriptors' if all(x['status']=='passed' for x in records) else 'blocked_prediction_descriptor_support',records=records)
        elif args.stage=='evaluate':result=m.evaluate(cfg['files'],Path('/descriptors'),args.output/'evaluation',cfg['source_sha256'],access)
        else:raise ValueError('Explicit worker stage required')
        m.rd.write_json(args.output/'result.json',dict(result,forbidden_reads_refused=refused,
            GPU_used=False,model_loaded=False))
    finally:
        m.rd.write_json(args.output/'evaluator-access.json',dict(semantic_reads=access))
        m.rd.write_json(args.output/'resources.json',dict(wall_seconds=time.monotonic()-start,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            error=None if sys.exc_info()[0] is None else str(sys.exc_info()[1])))


def run(args):
    if args.stage not in ['validate','describe','evaluate'] or (args.stage=='evaluate')!=(args.descriptors is not None):
        raise ValueError('Stage/descriptor role mismatch')
    source=args.source.resolve();m=source_module(source);plan=m.checked_plan(args.plan)
    manifest=json.loads(args.asset_manifest.read_text());rows=role_rows(plan,manifest,args.stage,m.PLAN_SHA)
    output=args.output.resolve()
    protected=[source,args.plan.resolve(),args.asset_manifest.resolve()]+[Path(x['path']).resolve() for x in manifest['files']]
    if args.descriptors is not None:protected.append(args.descriptors.resolve())
    if output.exists() or any(output==p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Fresh disjoint output required')
    sources={'ordinal_margin.py':source/'src/geosplat_r/ordinal_margin.py',
        'relative_diagnostics.py':source/'src/geosplat_r/relative_diagnostics.py',
        'run_ordinal_margin_cpu.py':source/'scripts/run_ordinal_margin_cpu.py',
        'prepare_sensor_anchors_cpu.py':source/'scripts/prepare_sensor_anchors_cpu.py'}
    hashes={n:m.digest(p) for n,p in sources.items()}
    if args.stage=='evaluate':m.verified_descriptors(args.descriptors,rows,hashes,load_values=False)
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic();status='failed_'+args.stage
    try:
        bundle=output/'code';bundle.mkdir()
        for name,path in sources.items():shutil.copyfile(path,bundle/name)
        m.rd.write_json(output/'source-manifest.json',hashes)
        import cv2,numpy as np
        sentinel=output/'host-only.png'
        if not cv2.imwrite(str(sentinel),np.full((2,2,3),71,dtype=np.uint8)):raise OSError('Sentinel failed')
        forbidden=[str(sentinel),'/weights/model.pth','/raw/data.7z','/labels/native.JPG']
        if args.stage in ['describe','probe']:
            forbidden += [r['path'] for r in manifest['files'] if r['key'] not in {x['key'] for x in rows}]
        files=[dict(r,path='/assets/'+r['key']) for r in rows]
        cfg=output/'config.json';m.rd.write_json(cfg,dict(plan_sha256=m.PLAN_SHA,files=files,source_sha256=hashes,forbidden=forbidden))
        common=['--ro-bind',str(bundle),'/code','--ro-bind',str(cfg),'/config.json']
        rt=load(source/'scripts/prepare_sensor_anchors_cpu.py','readonly_CPU_runtime')
        def command(mounts,stage):
            cmd=rt.sandbox(args.python,mounts,['--worker','--stage',stage,'--config','/config.json','--output','/output/result'])
            cmd[cmd.index('/code/prepare_sensor_anchors_cpu.py')]='/code/run_ordinal_margin_cpu.py'
            return cmd
        probe=output/'probe';probe.mkdir()
        rt.execute(command(common+['--bind',str(probe),'/output'],'probe'),output,'CPU-probe',timeout=60)
        mounts=common.copy()
        for row in rows:mounts+=['--ro-bind',row['path'],'/assets/'+row['key']]
        if args.stage=='evaluate':mounts+=['--ro-bind',str(args.descriptors),'/descriptors']
        stage=output/'stage';stage.mkdir()
        rt.execute(command(mounts+['--bind',str(stage),'/output'],args.stage),output,args.stage,timeout=600)
        stage_status=json.loads((stage/'result/result.json').read_text())['status']
        if any(m.digest(bundle/n)!=h for n,h in hashes.items()):raise RuntimeError('Executed source drift')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>256*1024**2:
            raise RuntimeError('Output budget exceeded; preserve and stop')
        status=stage_status
    finally:
        m.rd.write_json(output/'completion.json',dict(status=status,stage=args.stage,wall_seconds=time.monotonic()-start,
            error=None if sys.exc_info()[0] is None else str(sys.exc_info()[1])))
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['validate','describe','evaluate','probe'],required=True)
    parser.add_argument('--worker',action='store_true')
    for flag in ['source','plan','asset-manifest','python','output','descriptors','config']:
        parser.add_argument('--'+flag,type=Path)
    args=parser.parse_args()
    for flag in (['config','output'] if args.worker else ['source','plan','asset_manifest','python','output']):
        if getattr(args,flag) is None:parser.error('Explicit location required: '+flag)
    if args.worker:worker(args)
    else:print(run(args))
