"""Generate and align Small-model priors using only frozen fern training images."""
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.baseline import sha256,MemorySampler
from geosplat_r.depth import align_inverse_depth
from geosplat_r.isolation import audit_training_reads

WEIGHTS_SHA='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'
SOURCE_SHA='a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['data','anchors','source','source-manifest','weights','output']:
        ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    access=audit_training_reads(args.data)
    if sha256(args.weights)!=WEIGHTS_SHA:raise ValueError('Unexpected Small weights; no network fallback')
    source=json.loads(args.source_manifest.read_text())
    if source['sha']!=SOURCE_SHA:raise ValueError('Unexpected depth-model source revision')
    for r in source['files']:
        if sha256(args.source/r['path'])!=r['sha256']:raise ValueError('Depth-model source changed')
    views=json.loads((args.data/'cameras.json').read_text())
    if [v['name'] for v in views]!=['image001.png','image010.png','image019.png']:raise ValueError('Frozen train split required')
    anchors=json.loads((args.anchors/'manifest.json').read_text())
    assert anchors['cameras_sha256']==sha256(args.data/'cameras.json')
    assert anchors['points_sha256']==sha256(args.data/'points.npz')
    assert [r['image'] for r in anchors['records']]==[v['name'] for v in views]
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.reset_peak_memory_stats()
    sampler=MemorySampler(args.output/'gpu-memory.csv');sampler.start()
    start=time.time();perf=time.perf_counter();status='failed';error=None;records=[]
    try:
        sys.path.insert(0,str(args.source))
        from depth_anything_v2.dpt import DepthAnythingV2
        model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
        model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True),strict=True)
        model=model.cuda().eval()
        for view,anchor in zip(views,anchors['records']):
            p=args.data/'images'/view['name']
            assert sha256(p)==view['image_sha256']
            image=cv2.imread(str(p));assert image.shape==(378,504,3)
            transformed,_=model.image2tensor(image,518)
            input_shape=list(transformed.shape);del transformed
            torch.cuda.synchronize();tick=time.time()
            raw=model.infer_image(image,input_size=518)
            torch.cuda.synchronize();seconds=time.time()-tick
            a=args.anchors/anchor['file'];assert sha256(a)==anchor['sha256']
            with np.load(a) as values:xy=values['xy'];invz=values['inverse_z']
            target,valid,diagnostics=align_inverse_depth(raw,xy,invz)
            name=Path(view['name']).stem;dest=args.output/(name+'.npz')
            np.savez(dest,raw_inverse_depth=raw,inverse_z=target,valid=valid)
            import matplotlib
            lo,hi=np.percentile(target[valid],[2,98])
            color=matplotlib.colormaps['viridis'](np.clip((target-lo)/(hi-lo),0,1))[...,:3]
            color[~valid]=0
            panel=np.concatenate([cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(color*255).astype(np.uint8)],axis=1)
            Image.fromarray(panel).save(args.output/(name+'-prior.png'))
            records.append({'image':view['name'],'image_sha256':view['image_sha256'],
                'file':dest.name,'sha256':sha256(dest),'alignment':diagnostics,
                'inference_wall_seconds':seconds,'network_input_shape':input_shape})
        result={'protocol':'fern-undistorted-v1','model':'Depth Anything V2 Small relative inverse depth',
            'weights_sha256':WEIGHTS_SHA,'source_revision':SOURCE_SHA,
            'source_manifest_sha256':sha256(args.source_manifest),'source_files':source['files'],
            'cameras_sha256':sha256(args.data/'cameras.json'),'points_sha256':sha256(args.data/'points.npz'),
            'anchors_manifest_sha256':sha256(args.anchors/'manifest.json'),
            'records':records,'precision':'float32','input_size':518,'xformers':False,
            'target_space':'affine aligned inverse camera-axis Z; arbitrary frozen normalized scene units',
            'alignment_selection':'train observed SfM anchors only; all OLS samples equal weight',
            'semantics_source':'https://arxiv.org/html/2406.09414v2#S5.SS2',
            'limitations':'Fitted residuals are not independent depth accuracy; no visibility ground truth or confidence weighting'}
        (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        status='passed'
    except BaseException as exc:error=repr(exc);raise
    finally:
        torch.cuda.synchronize()
        resources={'status':status,'error':error,'elapsed_wall_seconds':time.time()-start,
            'elapsed_perf_counter_seconds':time.perf_counter()-perf,
            'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,**sampler.stop()}
        (args.output/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
        (args.output/'training-access.json').write_text(json.dumps(sorted(set(access)),indent=2)+'\n')
        print(json.dumps(resources,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'},indent=2))


if __name__=='__main__':main()
