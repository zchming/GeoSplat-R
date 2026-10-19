"""Minimal adapter of gsplat's example training behavior; no algorithm additions.

Training loop/order follows gsplat/examples/simple_trainer.py at
937e29912570c372bed6747a5c9bf85fed877bae, Nerfstudio Team, Apache-2.0.
Modifications: explicit train-only assets, GPU image caching, post-update checkpoint,
resource logging, optional uniform depth loss and separate evaluation.
See docs/licenses/gsplat-Apache-2.0.txt.
"""
import hashlib
import csv
import datetime
import os
import importlib.util
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fused_ssim import fused_ssim
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

UPSTREAM_SHA = '937e29912570c372bed6747a5c9bf85fed877bae'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_upstream(source):
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual != UPSTREAM_SHA:
        raise ValueError(f'Unexpected upstream commit: {actual}')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('Upstream source must be unmodified')
    sys.path.insert(0,str(source/'examples'))
    spec = importlib.util.spec_from_file_location('gsplat_upstream_trainer',source/'examples/simple_trainer.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def load_views(root):
    records=json.loads((root/'cameras.json').read_text())
    views=[]
    for r in records:
        if Path(r['name']).name != r['name']:
            raise ValueError('Image names must be basenames')
        path=root/'images'/r['name']
        if sha256(path)!=r['image_sha256']:
            raise ValueError(f'Image hash mismatch: {r["name"]}')
        pixels=np.asarray(Image.open(path).convert('RGB')).copy()
        if pixels.shape != (r['height'],r['width'],3):
            raise ValueError('Image/intrinsics dimensions disagree')
        views.append({**r,'pixels':torch.from_numpy(pixels).float()/255.,
                      'view':torch.linalg.inv(torch.tensor(r['c2w'],dtype=torch.float32)).cuda()[None],
                      'intrinsics':torch.tensor(r['K'],dtype=torch.float32).cuda()[None]})
    return views


def render(params, view, degree=3, depth=False):
    return rasterization(means=params['means'],quats=params['quats'],
        scales=params['scales'].exp(),opacities=params['opacities'].sigmoid(),
        colors=torch.cat([params['sh0'],params['shN']],dim=1),
        viewmats=view['view'],Ks=view['intrinsics'],width=view['width'],height=view['height'],
        packed=True,sh_degree=degree,near_plane=.01,far_plane=1e10,
        render_mode='RGB+ED' if depth else 'RGB',rasterize_mode='classic')


class MemorySampler:
    """Whole-board readings include desktop/other processes, unlike Torch peaks."""
    def __init__(self,output):
        self.output=output;self.peak=0;self.samples=0;self.error=None;self.utilization=[];self.first_timestamp=None;self.last_timestamp=None
    def start(self):
        self.process=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def consume():
            with self.output.open('x') as f:
                f.write('timestamp,board_used_mib,gpu_utilization_percent\n')
                for line in self.process.stdout:
                    f.write(line);f.flush()
                    try:
                        parts=next(csv.reader([line]))
                        self.peak=max(self.peak,int(parts[1]));self.samples+=1
                        self.utilization.append(int(parts[2]))
                        timestamp=datetime.datetime.strptime(parts[0],'%Y/%m/%d %H:%M:%S.%f')
                        if self.first_timestamp is None:self.first_timestamp=timestamp
                        self.last_timestamp=timestamp
                    except ValueError:
                        self.error='Could not parse nvidia-smi output'
        self.thread=threading.Thread(target=consume,daemon=True);self.thread.start()
    def stop(self):
        self.process.terminate();self.thread.join(timeout=5)
        return {'whole_board_sampled_peak_mib':self.peak,'sampling_period_ms':200,
                'samples':self.samples,'sampler_error':self.error,
                'gpu_timestamp_sampling_span_seconds':(self.last_timestamp-self.first_timestamp).total_seconds() if self.samples else None,
                'gpu_utilization_mean_percent':float(np.mean(self.utilization)) if self.utilization else None,
                'gpu_utilization_peak_percent':max(self.utilization) if self.utilization else None}


def train(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    total_vram=torch.cuda.get_device_properties(0).total_memory
    free_vram,_=torch.cuda.mem_get_info()
    memory_fraction=min(.95,(free_vram-1024*2**20)/total_vram)
    if memory_fraction<=0:raise RuntimeError('Insufficient VRAM for 1 GiB headroom')
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    upstream=load_upstream(args.source)
    upstream.set_random_seed(args.seed)
    data=args.data.resolve()
    accessed=[]
    def audit(event,arguments):
        if event=='open' and isinstance(arguments[0],(str,bytes)):
            p=Path(os.fsdecode(arguments[0])).resolve()
            # No sibling test/raw/preparation metadata access after input validation.
            if data.parent in p.parents and not (data==p or data in p.parents):
                raise RuntimeError(f'Training attempted to open a non-training asset: {p}')
            if data in p.parents: accessed.append(str(p.relative_to(data)))
    sys.addaudithook(audit)
    # Training reads only this explicit three-image directory, never any split parent.
    views=load_views(data)
    for view in views:view['pixels']=view['pixels'].cuda()
    scene_manifest=getattr(args,'scene_manifest',None)
    scene_input=None
    if scene_manifest is not None:
        from geosplat_r.scene_input import validate_scene_input
        scene_input=validate_scene_input(data,views,scene_manifest)
    elif [v['name'] for v in views] != ['image001.png','image010.png','image019.png']:
        raise ValueError('This authorized substep requires the frozen fern 3-view split')
    initial=np.load(data/'points.npz')
    centers=np.array([v['c2w'] for v in views])[:,:3,3]
    scene_scale=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
    parser=type('TrainOnlyPoints',(),{'points':initial['points'],'points_rgb':initial['rgb']})()
    strategy=DefaultStrategy()
    depth_root=getattr(args,'depth_priors',None)
    depth_lambda=getattr(args,'depth_lambda',0.)
    depth_start=getattr(args,'depth_start',1000)
    depth_priors=None
    if depth_root is not None:
        from geosplat_r.depth import load_depth_priors,uniform_inverse_depth_loss
        depth_priors,depth_manifest=load_depth_priors(depth_root,views,sha256(data/'cameras.json'),sha256(data/'points.npz'),sha256)
    config={'upstream':UPSTREAM_SHA,'seed':args.seed,'resolution':[views[0]['width'],views[0]['height']],
            'train_images':[{'name':v['name'],'sha256':v['image_sha256']} for v in views],
            'cameras_sha256':sha256(data/'cameras.json'),'points_sha256':sha256(data/'points.npz'),
            'max_steps':args.max_steps,'executed_steps':args.stop_after or args.max_steps,
            'scene_scale':scene_scale,'strategy':asdict(strategy),'packed':True,'precision':'float32',
            'sh_degree':3,'sh_degree_interval':1000,'rgb_loss':'0.8 L1 + 0.2 (1-fused SSIM valid)',
            'batch_size':1,'near_plane':.01,'far_plane':1e10,
            'camera_optimization':False,'depth_loss':depth_lambda>0,'appearance_optimization':False,
            'image_cache':'3 CUDA float32 RGBs','checkpoint_semantics':'after all requested optimizer updates',
            'schedule_reference':'full max_steps; pilot does not compress schedule',
            'training_implementation_sha256':sha256(Path(__file__)),
            'memory_fraction':memory_fraction,'initial_free_mib':free_vram/2**20,
            'memory_headroom_mib':1024,'timing_protocol':'time.time, perf_counter, CUDA events and GPU timestamps'}
    if depth_priors is not None:
        config['depth_protocol']={'lambda':depth_lambda,'start_zero_based_step':depth_start,
            'loss':'mean valid |1/ED - aligned inverse Z| / median train-anchor inverse Z',
            'weighting':'uniform; no robust loss or reliability weights',
            'mask':'fixed positive target and 2px border; detached alpha>=0.5; finite positive ED',
            'normalization':'per-image count of currently valid pixels; no hidden loss-sum change',
            'render_mode':'RGB+ED for lambda-zero and positive controls',
            'priors_manifest_sha256':sha256(depth_root/'manifest.json'),
            'depth_implementation_sha256':sha256(Path(__file__).with_name('depth.py'))}
    if scene_input is not None:
        config['scene_input']={'manifest_sha256':sha256(scene_manifest),'input':scene_input,
            'adapter_sha256':sha256(Path(__file__).with_name('scene_input.py'))}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    source_dir=args.output/'source';source_dir.mkdir()
    (source_dir/'baseline.py').write_bytes(Path(__file__).read_bytes())
    if scene_input is not None:
        (source_dir/'scene_input.py').write_bytes(Path(__file__).with_name('scene_input.py').read_bytes())
    if depth_priors is not None:(source_dir/'depth.py').write_bytes(Path(__file__).with_name('depth.py').read_bytes())
    torch.cuda.reset_peak_memory_stats()
    sampler=MemorySampler(args.output/'gpu-memory.csv');sampler.start()
    started=time.perf_counter();wall_started=time.time();status='failed';error=None;records=[]
    try:
        params,optimizers=upstream.create_splats_with_optimizers(parser,scene_scale=scene_scale,device='cuda')
        initial_digest=hashlib.sha256()
        for k,p in params.items():initial_digest.update(k.encode());initial_digest.update(p.detach().cpu().numpy().tobytes())
        state=strategy.initialize_state(scene_scale=scene_scale);strategy.check_sanity(params,optimizers)
        schedule=torch.optim.lr_scheduler.ExponentialLR(optimizers['means'],gamma=.01**(1/args.max_steps))
        generator=torch.Generator().manual_seed(args.seed)
        order=[];iteration_count=args.stop_after or args.max_steps
        torch.cuda.synchronize();loop_started=time.perf_counter();loop_wall_started=time.time()
        event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True)
        event_start.record()
        for step in range(iteration_count):
            if not order:order=torch.randperm(len(views),generator=generator).tolist()
            view_index=order.pop(0);view=views[view_index];pixels=view['pixels'][None]
            rgb_depth,alpha,info=render(params,view,degree=min(step//1000,3),depth=depth_priors is not None)
            rgb=rgb_depth[...,:3]
            strategy.step_pre_backward(params,optimizers,state,step,info)
            l1=(rgb-pixels).abs().mean()
            ssim=1-fused_ssim(rgb.permute(0,3,1,2),pixels.permute(0,3,1,2),padding='valid')
            loss=.8*l1+.2*ssim
            if depth_priors is not None:
                prior=depth_priors[view_index]
                depth_term,depth_fraction=uniform_inverse_depth_loss(rgb_depth[...,3],alpha,
                    prior['target'],prior['mask'],prior['normalizer'])
                weight=depth_lambda if step>=depth_start else 0.
                if weight:loss=loss+weight*depth_term
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            schedule.step()
            strategy.step_post_backward(params,optimizers,state,step,info,packed=True)
            if (step+1)%1000==0 or step==0 or step+1==iteration_count:
                torch.cuda.synchronize()
                row={'updates':step+1,'loss':loss.item(),'gaussians':len(params['means']),
                     'elapsed_perf_counter_seconds':time.perf_counter()-loop_started,
                     'elapsed_wall_clock_seconds':time.time()-loop_wall_started,
                     'allocated_mib':torch.cuda.memory_allocated()/2**20,
                     'reserved_mib':torch.cuda.memory_reserved()/2**20}
                if depth_priors is not None:row.update({'rgb_l1':l1.item(),'one_minus_ssim':ssim.item(),
                    'depth_normalized_l1':depth_term.item(),'depth_valid_fraction':depth_fraction.item(),
                    'depth_weight':weight})
                records.append(row);print(json.dumps(row),flush=True)
        event_end.record();torch.cuda.synchronize();training_seconds=time.perf_counter()-loop_started
        training_wall_seconds=time.time()-loop_wall_started
        stream_ms=event_start.elapsed_time(event_end)
        assert all(torch.isfinite(p).all() for p in params.values()),'Nonfinite final parameter'
        torch.save({'splats':params.state_dict(),'optimizer_updates':iteration_count,'config':config},args.output/'checkpoint.pt')
        (args.output/'training-curve.json').write_text(json.dumps(records,indent=2)+'\n')
        (args.output/'training-access.json').write_text(json.dumps(sorted(set(accessed)),indent=2)+'\n')
        status='passed'
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        torch.cuda.synchronize()
        result={'status':status,'error':error,'elapsed_perf_counter_seconds':time.perf_counter()-started,
                'elapsed_wall_clock_seconds':time.time()-wall_started,
                'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,**sampler.stop()}
        if status=='passed':result.update({'training_perf_counter_seconds':training_seconds,
            'training_wall_clock_seconds':training_wall_seconds,'cuda_stream_elapsed_seconds':stream_ms/1000,
            'optimizer_updates':iteration_count,'final_gaussians':len(params['means']),
            'initial_gaussians':len(initial['points']),'initial_parameters_sha256':initial_digest.hexdigest()})
        (args.output/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
