"""Run the unmodified pinned gsplat Runner.train with a train-only dataset adapter."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.baseline import MemorySampler,UPSTREAM_SHA,load_upstream,sha256


class TrainOnlyDataset(torch.utils.data.Dataset):
    def __init__(self,root):
        self.records=json.loads((root/'cameras.json').read_text());self.items=[]
        if [x['name'] for x in self.records]!=['image001.png','image010.png','image019.png']:
            raise ValueError('Frozen fern 3-view training data required')
        for i,r in enumerate(self.records):
            p=root/'images'/r['name'];assert sha256(p)==r['image_sha256']
            image=np.array(Image.open(p).convert('RGB'))
            assert image.shape==(r['height'],r['width'],3)
            self.items.append({'image':torch.from_numpy(image),
                'camtoworld':torch.tensor(r['c2w'],dtype=torch.float32),
                'K':torch.tensor(r['K'],dtype=torch.float32),'image_id':i})
    def __len__(self):return len(self.items)
    def __getitem__(self,index):return self.items[index]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ['data','source','output']:ap.add_argument('--'+field,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    total=torch.cuda.get_device_properties(0).total_memory
    free,_=torch.cuda.mem_get_info()
    fraction=min(.95,(free-1024*2**20)/total)
    if fraction<=0:raise RuntimeError('Insufficient free VRAM for 1 GiB headroom')
    torch.cuda.set_per_process_memory_fraction(fraction)
    upstream=load_upstream(args.source);upstream.set_random_seed(42)
    data=args.data.resolve();accessed=[]
    def audit(event,arguments):
        if event=='open' and isinstance(arguments[0],(str,bytes)):
            p=Path(os.fsdecode(arguments[0])).resolve()
            if data.parent in p.parents and not (data==p or data in p.parents):
                raise RuntimeError(f'Non-training asset access: {p}')
            if data in p.parents:accessed.append(str(p.relative_to(data)))
    sys.addaudithook(audit)
    dataset=TrainOnlyDataset(data);points=np.load(data/'points.npz')
    centers=np.array([r['c2w'] for r in dataset.records])[:,:3,3]
    scale=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
    parser=type('Points',(),{'points':points['points'],'points_rgb':points['rgb']})()
    cfg=upstream.Config(disable_viewer=True,data_dir=str(data),data_factor=1,result_dir=str(args.output),
        max_steps=30000,eval_steps=[],save_steps=[30000],save_ply=False,disable_video=True,
        packed=True,tb_every=0)
    runner=object.__new__(upstream.Runner)
    runner.cfg=cfg;runner.device='cuda:0';runner.local_rank=0;runner.world_rank=0;runner.world_size=1
    runner.trainset=dataset;runner.scene_scale=scale
    runner.pose_optimizers=[];runner.app_optimizers=[];runner.bil_grid_optimizers=[]
    runner.writer=SummaryWriter(log_dir=str(args.output/'tb'))
    for attribute,name in [('ckpt_dir','ckpts'),('stats_dir','stats'),('render_dir','renders'),('ply_dir','ply')]:
        p=args.output/name;p.mkdir();setattr(runner,attribute,str(p))
    sampler=MemorySampler(args.output/'gpu-memory.csv');sampler.start()
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();status='failed';error=None
    try:
        runner.splats,runner.optimizers=upstream.create_splats_with_optimizers(parser,scene_scale=scale,device='cuda')
        digest=hashlib.sha256()
        for k,p in runner.splats.items():digest.update(k.encode());digest.update(p.detach().cpu().numpy().tobytes())
        runner.cfg.strategy.check_sanity(runner.splats,runner.optimizers)
        runner.strategy_state=runner.cfg.strategy.initialize_state(scene_scale=scale)
        torch.cuda.synchronize();train_tick=time.perf_counter()
        # Exact upstream method body; no evaluation calls are scheduled.
        runner.train()
        torch.cuda.synchronize();seconds=time.perf_counter()-train_tick
        assert all(torch.isfinite(p).all() for p in runner.splats.values())
        config={'upstream':UPSTREAM_SHA,'seed':42,'max_steps':30000,'resolution':[504,378],
                'engine':'unmodified gsplat Runner.train',
                'input_adapter':'3 CPU uint8 training images and frozen calibrated cameras only',
                'sampling':'upstream DataLoader shuffle=True with 4 workers; differs from local dedicated generator',
                'points_sha256':sha256(data/'points.npz'),'cameras_sha256':sha256(data/'cameras.json'),
                'memory_fraction':fraction,'initial_free_mib':free/2**20,'memory_headroom_mib':1024,
                'wrapper_sha256':sha256(Path(__file__)),
                'upstream_trainer_sha256':sha256(args.source/'examples/simple_trainer.py')}
        torch.save({'splats':runner.splats.state_dict(),'optimizer_updates':30000,'config':config},args.output/'checkpoint.pt')
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (args.output/'training-access.json').write_text(json.dumps(sorted(set(accessed)),indent=2)+'\n')
        status='passed'
    except BaseException as exc:error=repr(exc);raise
    finally:
        runner.writer.close()
        result={'status':status,'error':error,'wall_seconds':time.perf_counter()-tick,
                'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,**sampler.stop()}
        if status=='passed':result.update({'training_loop_seconds':seconds,'optimizer_updates':30000,
            'initial_gaussians':len(points['points']),'final_gaussians':len(runner.splats['means']),
            'initial_parameters_sha256':digest.hexdigest()})
        (args.output/'resources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
