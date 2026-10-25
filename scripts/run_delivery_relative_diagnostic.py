"""Isolated relative diagnostic: prepare real IO, synthetic GPU preflight, or later released stages.

Predict/evaluate are explicit separate stages. Preflight never mounts real images or GT.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time


def load(name, filename, repo_path):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        path = Path(__file__).resolve().parents[1]/repo_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bridge():
    return load('delivery_relative_io', 'delivery_relative_io.py', 'src/geosplat_r/delivery_relative_io.py')


def runtime():
    return load('readonly_CPU_runtime', 'prepare_sensor_anchors_cpu.py', 'scripts/prepare_sensor_anchors_cpu.py')


def validate_stage_args(args):
    if args.stage not in ['preflight','predict','evaluate']:
        raise ValueError('Explicit public stage required')
    required = ['model_source','weights'] if args.stage=='preflight' else (['data','model_source','weights'] if args.stage=='predict' else ['predictions','labels'])
    forbidden = ['data','predictions','labels'] if args.stage=='preflight' else (['predictions','labels'] if args.stage=='predict' else ['data','model_source','weights'])
    if any(getattr(args,k) is None for k in required) or any(getattr(args,k) is not None for k in forbidden):
        raise ValueError('Missing required or cross-role stage location')


def validate_context(config, stage):
    io=bridge()
    if (config['protocol']!=io.PROTOCOL or config['plan_sha256']!=io.PLAN_SHA
            or config['names']!=io.NAMES or config['native_wh']!=[6048,4032]
            or config['work_wh']!=[1008,672]):
        raise ValueError('Frozen delivery context mismatch')
    if stage in ['preflight','predict'] and ('model' not in config or 'labels' in config):
        raise ValueError('Predictor/evaluator roles mixed')
    if stage=='evaluate' and ('labels' not in config or 'model' in config or 'inputs' in config):
        raise ValueError('Evaluator/predictor roles mixed')


def device_mounts():
    if not Path('/dev/dxg').is_char_device() or not Path('/usr/lib/wsl/lib/libcuda.so.1').is_file():
        raise RuntimeError('Reviewed WSL GPU interface unavailable; no CPU or broad-mount fallback')
    return ['--dev-bind','/dev/dxg','/dev/dxg','--ro-bind','/sys','/sys',
            '--setenv','LD_LIBRARY_PATH','/usr/lib/wsl/lib','--setenv','CUDA_VISIBLE_DEVICES','0']


def sandbox_command(python, mounts, arguments, gpu=False):
    cmd = runtime().sandbox(python, mounts + (device_mounts() if gpu else []), arguments)
    cmd[cmd.index('/code/prepare_sensor_anchors_cpu.py')] = '/code/run_delivery_relative_diagnostic.py'
    return cmd


def denied_reads(paths):
    import os
    import cv2
    import numpy as np
    refused = []
    for path in paths:
        for method in ['python','os','numpy','opencv']:
            try:
                if method == 'python':Path(path).read_bytes()
                elif method == 'os':
                    fd=os.open(path,os.O_RDONLY);os.close(fd)
                elif method == 'numpy':np.load(path,allow_pickle=False)
                elif cv2.imread(path) is not None:raise RuntimeError('Native forbidden read')
                else:
                    refused.append([path,method]);continue
            except OSError:refused.append([path,method])
            else:raise RuntimeError('Forbidden path readable: '+path)
    return refused


def make_dummy():
    import numpy as np
    y,x=np.indices((672,1008))
    return np.stack([(x+3*y)%256,(2*x+y)%256,(x+y)%256],axis=-1).astype(np.uint8)


def install_forward_probe(model,torch):
    """Observe direct self.forward calls without adding another model pass."""
    forwards=[]
    original=model.forward
    def checked(tensor):
        if list(tensor.shape)!=[1,3,518,784] or tensor.dtype!=torch.float32 or tensor.device.type!='cuda' or torch.is_grad_enabled():
            raise ValueError('Frozen CUDA forward shape/type/gradient contract mismatch')
        forwards.append(list(tensor.shape))
        return original(tensor)
    model.forward=checked
    return forwards,original


def verify_imported_model_files(modules,records):
    allowed={'/model-source/'+r['path'] for r in records}
    for name,module in list(modules.items()):
        if name=='depth_anything_v2' or name.startswith('depth_anything_v2.'):
            filename=getattr(module,'__file__',None)
            if filename is not None and str(Path(filename).resolve()) not in allowed:
                raise ValueError('Imported model module outside frozen source: '+name)


def gpu_worker(config, output, preflight):
    io=bridge();rd=io.rd
    import numpy as np
    import cv2
    if np.__version__!='1.26.4' or cv2.__version__!='4.11.0':raise ValueError('Frozen preprocessing environment mismatch')
    cv2.setNumThreads(4)
    refused=denied_reads(config['forbidden'])
    if not preflight:views=io.admit_predictor('/inputs',config)
    model_spec=config['model']
    for row in model_spec['source_files']:
        p=Path('/model-source')/row['path']
        if Path(row['path']).is_absolute() or '..' in Path(row['path']).parts or rd.digest(p)!=row['sha256']:
            raise ValueError('Model source drift')
    weight=Path('/weights/small.pth')
    if weight.stat().st_size!=model_spec['weights_bytes'] or rd.digest(weight)!=model_spec['weights_sha256']:
        raise ValueError('Weight identity mismatch')
    import torch
    if torch.__version__!='2.4.1+cu124' or importlib.util.find_spec('xformers') is not None:
        raise ValueError('Frozen float32/no-xformers environment required')
    torch.set_num_threads(4);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in isolated namespace')
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one local GPU required')
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0,'/model-source')
    from depth_anything_v2.dpt import DepthAnythingV2
    verify_imported_model_files(sys.modules,model_spec['source_files'])
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval()
    forwards,original_forward=install_forward_probe(model,torch)
    records=[]
    pred_root=Path(output)/'predictions'
    if not preflight:pred_root.mkdir()
    for i in range(1 if preflight else 3):
        if preflight:image=make_dummy()
        else:
            image=cv2.imread(str(Path('/inputs/rgb')/views[i]['name']),cv2.IMREAD_COLOR)
            if image is None or image.shape!=(4032,6048,3):raise ValueError('Native RGB decode mismatch')
            image=cv2.resize(image,(1008,672),interpolation=cv2.INTER_AREA)
        torch.cuda.synchronize();tick=time.perf_counter()
        raw=model.infer_image(image,input_size=518)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        if raw.shape!=(672,1008) or raw.dtype!=np.float32:raise ValueError('Raw output schema mismatch')
        finite=np.isfinite(raw)
        stats=dict(shape=list(raw.shape),dtype=str(raw.dtype),nonfinite=int((~finite).sum()),
                   negative=int((finite&(raw<0)).sum()),zero=int((raw==0).sum()),
                   minimum=float(raw[finite].min()) if finite.any() else None,
                   maximum=float(raw[finite].max()) if finite.any() else None,
                   inference_seconds=seconds)
        if preflight:
            if not finite.all() or (raw<0).any():raise RuntimeError('Synthetic forward produced invalid output; stop')
        else:
            dest=pred_root/('%d.npy'%i);np.save(dest,raw)
            records.append(dict(name=config['names'][i],file=dest.name,sha256=rd.digest(dest),stats=stats))
    model.forward=original_forward
    verify_imported_model_files(sys.modules,model_spec['source_files'])
    expected=1 if preflight else 3
    if len(forwards)!=expected:raise RuntimeError('Unexpected forward count')
    if not preflight:
        io.seal_predictions(pred_root,config)
    rd.write_json(Path(output)/'GPU-result.json',dict(status='passed_synthetic_forward' if preflight else 'sealed_three_predictions',
        synthetic_only=preflight,forward_count=len(forwards),forward_shapes=forwards,output_stats=stats if preflight else records,
        forbidden_reads_refused=refused,real_RGB_decoded=0 if preflight else 3,GT_read=False,
        formal_predictions_written=0 if preflight else 3,torch_version=torch.__version__,CUDA_build=torch.version.cuda,
        GPU_name=torch.cuda.get_device_name(0),GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        training_mode=model.training,optimizer_created=False,network_downloads_possible=False))


def worker(args):
    io=bridge();rd=io.rd;start=time.monotonic();access=[]
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        config=json.loads(args.config.read_text());validate_context(config, args.stage)
        if args.stage=='probe':
            rd.write_json(args.output/'probe.json',dict(status='passed',denied=denied_reads(config['forbidden']),
                CUDA_device_exposed=Path('/dev/dxg').exists(),model_loaded=False))
        elif args.stage in ['preflight','predict']:
            gpu_worker(config,args.output,args.stage=='preflight')
        elif args.stage=='evaluate':
            # CPU evaluator has no model/source/weights/GPU-device mounts.
            if Path('/dev/dxg').exists() or Path('/weights/small.pth').exists():raise RuntimeError('Evaluator exposed GPU/weights')
            io.evaluate_sealed(args.predictions,'/labels',config,args.output/'evaluation',access)
        else:raise ValueError('Worker stage required')
    finally:
        rd.write_json(args.output/'reference-access.json',dict(attempts=access))
        rd.write_json(args.output/'resources.json',dict(wall_seconds=time.monotonic()-start,
            max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            error=None if sys.exc_info()[0] is None else str(sys.exc_info()[1])))


class BoardSampler:
    def __init__(self,output):self.output=output;self.samples=[]
    def start(self):
        self.process=subprocess.Popen(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits','-lms','100'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def consume():
            with self.output.open('x') as f:
                for line in self.process.stdout:
                    f.write(line);f.flush()
                    parts=next(csv.reader([line]))
                    try:self.samples.append(int(parts[1]))
                    except (ValueError,IndexError):pass
        self.thread=threading.Thread(target=consume,daemon=True);self.thread.start()
    def stop(self):
        self.process.terminate();self.process.wait(timeout=5);self.thread.join(timeout=5)
        return dict(samples=len(self.samples),sampled_whole_board_peak_MiB=max(self.samples) if self.samples else None,sampling_period_ms=100)


def run(args):
    validate_stage_args(args)
    io=bridge();rd=io.rd;rt=runtime()
    if rd.digest(args.plan)!=io.PLAN_SHA:raise ValueError('Frozen protocol content drift')
    plan=json.loads(args.plan.read_text());root=args.output.resolve();source=args.source.resolve()
    inputs=[p.resolve() for p in [args.data,args.labels,args.predictions,args.model_source,args.weights,args.plan,args.input_manifest,args.base_plan,args.review_plan] if p is not None]
    if root.exists() or root==source or source in root.parents or any(root==p or p in root.parents or root in p.parents for p in inputs):
        raise ValueError('Fresh external disjoint output required')
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic();status='failed_'+args.stage
    try:
        bundle=root/'code';bundle.mkdir()
        code={'delivery_relative_io.py':source/'src/geosplat_r/delivery_relative_io.py',
            'relative_io.py':source/'src/geosplat_r/relative_io.py',
            'relative_diagnostics.py':source/'src/geosplat_r/relative_diagnostics.py',
            'prepare_sensor_anchors_cpu.py':source/'scripts/prepare_sensor_anchors_cpu.py',
            'run_delivery_relative_diagnostic.py':source/'scripts/run_delivery_relative_diagnostic.py'}
        for name,path in code.items():shutil.copyfile(path,bundle/name)
        hashes={name:rd.digest(bundle/name) for name in code};rd.write_json(root/'source-manifest.json',hashes)
        pred_config,eval_config=io.load_contexts(args.plan,args.input_manifest,args.base_plan,args.review_plan,hashes)
        config=eval_config if args.stage=='evaluate' else pred_config
        hidden=root/'unmounted-sentinel.png'
        # Valid tiny PNG bytes; no real label or RGB file accessed for isolation probes.
        import cv2,numpy as np
        if not cv2.imwrite(str(hidden),np.full((2,2,3),71,dtype=np.uint8)):raise OSError('Cannot create sentinel')
        config['forbidden']=[str(hidden),'/labels/forbidden.JPG','/archive/data.7z','/metadata/images.txt','/inputs/rgb/other.JPG']
        if args.stage=='preflight':config['forbidden']+=['/inputs/cameras.json']
        rd.write_json(root/'worker-config.json',config)
        mounts=['--ro-bind',str(bundle),'/code','--ro-bind',str(root/'worker-config.json'),'/config.json']
        probe=root/'CPU-probe';probe.mkdir()
        rt.execute(sandbox_command(args.python,mounts+['--bind',str(probe),'/output'],
            ['--worker','--stage','probe','--config','/config.json','--output','/output/result']),root,'CPU-probe',timeout=60)
        if args.stage in ['preflight','predict']:
            if args.model_source is None or args.weights is None:raise ValueError('Explicit model source and weights required')
            model=root/'model-source';model.mkdir()
            for row in pred_config['model']['source_files']:
                origin=args.model_source/row['path'];dest=model/row['path']
                if origin.is_symlink() or rd.digest(origin)!=row['sha256']:raise ValueError('Model source identity drift')
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,dest)
            spec=pred_config['model']
            if args.weights.is_symlink() or args.weights.stat().st_size!=spec['weights_bytes'] or rd.digest(args.weights)!=spec['weights_sha256']:
                raise ValueError('Weight identity drift')
            mounts+=['--ro-bind',str(model),'/model-source','--ro-bind',str(args.weights),'/weights/small.pth']
            if args.stage=='predict':
                if args.data is None:raise ValueError('Explicit selected predictor data required')
                io.admit_predictor(args.data,pred_config)
                for rel in sorted(pred_config['inputs']):mounts+=['--ro-bind',str(args.data/rel),'/inputs/'+rel]
        else:
            if args.predictions is None or args.labels is None:raise ValueError('Explicit sealed predictions and label root required')
            io.sealed_predictions(args.predictions,eval_config)  # Before any reference access or mounts.
            mounts+=['--ro-bind',str(args.predictions),'/predictions']
            for row in eval_config['labels']:
                if (args.labels/row['image']).is_symlink():raise ValueError('Reference symlink')
                mounts+=['--ro-bind',str(args.labels/row['image']),'/labels/'+row['image']]
        stage=root/'stage';stage.mkdir();mounts+=['--bind',str(stage),'/output']
        arguments=['--worker','--stage',args.stage,'--config','/config.json','--output','/output/result']
        if args.stage=='evaluate':arguments+=['--predictions','/predictions']
        sampler=BoardSampler(root/'board-memory.csv') if args.stage in ['preflight','predict'] else None
        if sampler:sampler.start()
        try:
            rt.execute(sandbox_command(args.python,mounts,arguments,gpu=sampler is not None),root,args.stage,timeout=600)
        finally:
            if sampler:rd.write_json(root/'board-memory.json',sampler.stop())
        if sampler:
            memory=json.loads((root/'board-memory.json').read_text())
            if not memory['samples'] or memory['sampled_whole_board_peak_MiB']>12*1024:raise RuntimeError('Whole-board memory gate failed/unknown')
            gpu=json.loads((stage/'result/GPU-result.json').read_text())
            if gpu['peak_reserved_MiB']>12*1024:raise RuntimeError('Torch memory gate failed')
        if any(rd.digest(bundle/name)!=h for name,h in hashes.items()):raise RuntimeError('Executed source drift')
        status='passed_'+args.stage
    finally:
        rd.write_json(root/'completion.json',dict(status=status,stage=args.stage,wall_seconds=time.monotonic()-start,
            formal_three_image_prediction=args.stage=='predict',real_GT_evaluation=args.stage=='evaluate',
            error=None if sys.exc_info()[0] is None else str(sys.exc_info()[1])))
    print((root/'completion.json').read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['preflight','predict','evaluate','probe'],required=True)
    ap.add_argument('--worker',action='store_true')
    for key in ['source','plan','input-manifest','base-plan','review-plan','python','data','labels','predictions','model-source','weights','output','config']:
        ap.add_argument('--'+key,type=Path)
    args=ap.parse_args()
    required=['config','output'] if args.worker else ['source','plan','input_manifest','base_plan','review_plan','python','output']
    for name in required:
        if getattr(args,name) is None:ap.error('Explicit location required: '+name)
    if args.worker:worker(args)
    else:
        if args.stage=='probe':ap.error('Probe is an internal CPU-only stage')
        run(args)


if __name__=='__main__':main()
