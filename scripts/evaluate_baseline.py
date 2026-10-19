"""Separate, post-training evaluation and real Gaussian visualizations."""
import argparse
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from PIL import Image, ImageDraw
from gsplat import export_splats
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from geosplat_r.baseline import MemorySampler, load_views, render, sha256


def save_rgb(path,array):
    Image.fromarray((np.clip(array,0,1)*255).round().astype(np.uint8)).save(path)


def scene_plot(params,train_views,test_views,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz=params['means'].detach().cpu().numpy()
    color=np.clip(params['sh0'][:,0].detach().cpu().numpy()*.28209479177387814+.5,0,1)
    opacity=params['opacities'].sigmoid().detach().cpu().numpy()
    ids=np.flatnonzero(opacity>.05)
    ids=ids[np.linspace(0,len(ids)-1,min(len(ids),25000)).astype(int)]
    fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(projection='3d')
    ax.scatter(*xyz[ids].T,c=color[ids],s=.4,alpha=.45,rasterized=True)
    for group,views,c in [('train',train_views,'red'),('test',test_views,'blue')]:
        for v in views:
            matrix=np.array(v['c2w']);p=matrix[:3,3];direction=matrix[:3,2]
            ax.scatter(*p,c=c,s=35);ax.quiver(*p,*direction,length=.3,color=c)
            ax.text(*p,group+' '+v['name'][5:8],color=c,fontsize=8)
    low,high=np.percentile(xyz[ids],[1,99],axis=0)
    cameras=np.array([v['c2w'] for v in train_views+test_views])[:,:3,3]
    low=np.minimum(low,cameras.min(0));high=np.maximum(high,cameras.max(0))
    for setter,lo,hi in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],low,high):setter(lo,hi)
    ax.set_box_aspect(high-low)
    ax.set(xlabel='World X',ylabel='World Y',zlabel='World Z',
           title=f'Trained Gaussian centers (display subset {len(ids):,}/{len(xyz):,}); camera axes')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


@torch.no_grad()
def evaluate(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.80)
    os.environ['TORCH_HOME']=str(args.weights)
    alex=args.weights/'hub/checkpoints/alexnet-owt-7be5be79.pth'
    if not alex.is_file() or sha256(alex)!='7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02':
        raise ValueError('Explicit verified AlexNet weights are required; no automatic network download')
    data=torch.load(args.checkpoint,map_location='cuda',weights_only=True)
    params=data['splats']
    train_views=load_views(args.train_data);test_views=load_views(args.test_data)
    assert {v['name'] for v in train_views}.isdisjoint(v['name'] for v in test_views)
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    ssim_options={'gaussian_kernel':True,'sigma':1.5,'kernel_size':11,'data_range':1.,'k1':.01,'k2':.03}
    # Independent metric sanity checks before consuming any test prediction.
    sample=torch.rand(1,3,64,64,device='cuda')
    assert abs(structural_similarity_index_measure(sample,sample,**ssim_options).item()-1)<1e-6
    assert abs(lpips(sample,sample).item())<1e-6;lpips.reset()
    torch.cuda.reset_peak_memory_stats()
    sampler=MemorySampler(args.output/'gpu-memory.csv');sampler.start()
    tic=time.perf_counter();results={};montages=[]
    try:
        for split,views in [('train',train_views),('test',test_views)]:
            records=[]
            for v in views:
                output=args.output/split/Path(v['name']).stem;output.mkdir(parents=True)
                for _ in range(3):render(params,v,depth=True)
                torch.cuda.synchronize()  # warm-up excluded from 20-pass rendering mean
                begin=time.perf_counter()
                for _ in range(20):rgb_depth,alpha,_=render(params,v,depth=True)
                torch.cuda.synchronize()
                render_seconds=(time.perf_counter()-begin)/20
                rgb=rgb_depth[...,:3].clamp(0,1);gt=v['pixels'].cuda()[None]
                pred=rgb.permute(0,3,1,2);target=gt.permute(0,3,1,2)
                mse=(pred-target).square().mean()
                psnr=(-10*torch.log10(mse)).item()
                mse_cpu=np.mean((rgb[0].cpu().numpy().astype(np.float64)-v['pixels'].numpy().astype(np.float64))**2)
                assert abs(psnr-(-10*np.log10(mse_cpu)))<1e-4
                ssim=structural_similarity_index_measure(pred,target,**ssim_options).item()
                perceptual=lpips(pred,target).item();lpips.reset()
                record={'image':v['name'],'psnr':psnr,'ssim':ssim,'lpips_alex':perceptual,
                        'rgb_depth_render_seconds':render_seconds}
                records.append(record)
                rendered=rgb[0].cpu().numpy();truth=v['pixels'].numpy()
                error=np.mean(np.abs(rendered-truth),axis=-1)
                depth=rgb_depth[0,...,3].cpu().numpy();a=alpha[0,...,0].cpu().numpy()
                valid=(a>=.5)&np.isfinite(depth)&(depth>0)
                np.save(output/'depth-z.npy',depth);np.save(output/'alpha.npy',a)
                Image.fromarray(valid.astype(np.uint8)*255).save(output/'depth-valid.png')
                save_rgb(output/'render.png',rendered);save_rgb(output/'gt.png',truth)
                import matplotlib
                error_rgb=matplotlib.colormaps['magma'](np.clip(error/.25,0,1))[...,:3]
                if valid.any():
                    near,far=np.percentile(depth[valid],[2,98]);normalized=np.clip((depth-near)/max(far-near,1e-6),0,1)
                else:near,far=0,1;normalized=np.zeros_like(depth)
                depth_rgb=matplotlib.colormaps['viridis'](normalized)[...,:3];depth_rgb[~valid]=0
                save_rgb(output/'error.png',error_rgb);save_rgb(output/'depth.png',depth_rgb)
                w,h=v['width'],v['height'];canvas=Image.new('RGB',(w*4,h+46),'white')
                draw=ImageDraw.Draw(canvas)
                for i,(arr,label) in enumerate([(truth,'GT'),(rendered,'Vanilla 3DGS'),
                        (error_rgb,'Mean absolute RGB error: 0 to 0.25'),(depth_rgb,'Expected Z depth; alpha >= 0.5')]):
                    panel=Image.fromarray((arr.clip(0,1)*255).round().astype(np.uint8))
                    canvas.paste(panel,(i*w,46));draw.text((i*w+5,23),label,fill='black')
                draw.text((5,5),f"{split} {v['name']} | PSNR {psnr:.3f} SSIM {ssim:.4f} LPIPS-Alex {perceptual:.4f}",fill='black')
                canvas.save(output/'comparison.png');montages.append((split,canvas))
                record.update({'valid_depth_fraction_alpha_0_5':float(valid.mean()),
                               'depth_visualization_percentile_2_98':[float(near),float(far)]})
            results[split]={'per_image':records,'mean':{k:float(np.mean([r[k] for r in records]))
                for k in ['psnr','ssim','lpips_alex','rgb_depth_render_seconds']}}
        test_panels=[canvas for split,canvas in montages if split=='test']
        sheet=Image.new('RGB',(test_panels[0].width,sum(c.height for c in test_panels)),'white')
        y=0
        for c in test_panels:sheet.paste(c,(0,y));y+=c.height
        sheet.save(args.output/'test-comparison.png')
        export_splats(**{k:params[k] for k in ['means','scales','quats','opacities','sh0','shN']},
                      format='ply',save_to=str(args.output/'scene.ply'))
        scene_plot(params,train_views,test_views,args.output/'scene-3d.png')
        # Render actual novel camera positions on the closed training-camera trajectory.
        import imageio.v2 as imageio
        from scipy.spatial.transform import Rotation,Slerp
        matrices=np.array([v['c2w'] for v in train_views])
        frames=[]
        for segment in range(3):
            a,b=matrices[segment],matrices[(segment+1)%3]
            rotations=Slerp([0,1],Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))
            for t in np.linspace(0,1,16,endpoint=False):
                c2w=np.eye(4);c2w[:3,:3]=rotations(t).as_matrix();c2w[:3,3]=(1-t)*a[:3,3]+t*b[:3,3]
                v={**train_views[0],'view':torch.linalg.inv(torch.tensor(c2w,dtype=torch.float32,device='cuda'))[None]}
                frame=render(params,v)[0][0].clamp(0,1).cpu().numpy()
                frames.append((frame*255).round().astype(np.uint8))
        imageio.mimsave(args.output/'novel-view-path.mp4',frames,fps=12,macro_block_size=1)
        lpips_file=Path(__import__('torchmetrics').__file__).parent/'functional/image/lpips_models/alex.pth'
        results.update({'checkpoint_sha256':sha256(args.checkpoint),'optimizer_updates':data['optimizer_updates'],
            'evaluation_seconds':time.perf_counter()-tic,
            'versions':{p:version(p) for p in ['torch','torchvision','gsplat','torchmetrics','numpy']},
            'metric_protocol':{'rgb':'sRGB [0,1], clamp prediction, no mask/crop',
                'psnr':'-10 log10 per-image RGB MSE, then arithmetic image mean',
                'ssim':ssim_options,'lpips':'AlexNet, normalize=True; per-image reset',
                'alexnet_sha256':sha256(alex),'lpips_calibration_sha256':sha256(lpips_file)},
            'depth':'Expected camera-axis Z, arbitrary normalized scene units, not ground truth geometry',
            'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
    finally:
        memory=sampler.stop()
    results.update(memory)
    (args.output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ['checkpoint','train-data','test-data','weights','output']:
        ap.add_argument('--'+field,type=Path,required=True)
    evaluate(ap.parse_args())
