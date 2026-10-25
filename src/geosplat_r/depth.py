"""Fixed train-anchor affine inverse-depth alignment and uniform depth loss."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch


def align_inverse_depth(raw,xy,inverse_z):
    raw=np.asarray(raw,dtype=np.float64);xy=np.asarray(xy,dtype=np.float64)
    y=np.asarray(inverse_z,dtype=np.float64)
    if len(y)<20 or not np.isfinite(y).all() or (y<=0).any():raise ValueError('At least 20 positive train anchors required')
    sample=cv2.remap(raw,xy[:,0].astype(np.float32)[None],xy[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]
    if not np.isfinite(raw).all() or np.std(sample)<1e-6:raise ValueError('Invalid or constant model prediction')
    A=np.stack([sample,np.ones_like(sample)],axis=1)
    slope,shift=np.linalg.lstsq(A,y,rcond=None)[0]
    if not np.isfinite([slope,shift]).all() or slope<=0:raise ValueError('Nonpositive inverse-depth alignment slope')
    target=slope*raw+shift
    mask=np.isfinite(target)&(target>1e-6)
    mask[:2]=False;mask[-2:]=False;mask[:,:2]=False;mask[:,-2:]=False
    if mask.mean()<.5:raise ValueError('Too few positive aligned target pixels')
    normalizer=float(np.median(y))
    diagnostics={'slope':float(slope),'shift':float(shift),'normalizer':normalizer,'anchors':len(y),
        'anchor_pearson_correlation':float(np.corrcoef(sample,y)[0,1]),
        'anchor_relative_l1':float(np.abs(A@np.array([slope,shift])-y).mean()/normalizer),
        'valid_fraction':float(mask.mean()),'fit':'unweighted ordinary least squares to train inverse Z'}
    return target.astype(np.float32),mask,diagnostics


def uniform_inverse_depth_loss(z,alpha,target,mask,normalizer):
    z=z.squeeze();alpha=alpha.squeeze()
    valid=mask & (alpha.detach()>=.5) & torch.isfinite(z) & (z.detach()>1e-6)
    count=valid.sum()
    # Boolean masks carry no gradients; the valid expected-Z values remain differentiable.
    safe_z=torch.where(valid,z,torch.ones_like(z))
    residual=(safe_z.reciprocal()-target).abs()/normalizer
    loss=torch.where(valid,residual,torch.zeros_like(residual)).sum()/count.clamp_min(1)
    return loss,count.to(z.dtype)/mask.numel()


def load_depth_priors(root,views,cameras_sha,points_sha,sha256):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['cameras_sha256']!=cameras_sha or manifest['points_sha256']!=points_sha:
        raise ValueError('Depth prior camera/point provenance mismatch')
    if [r['image'] for r in manifest['records']]!=[v['name'] for v in views]:
        raise ValueError('Depth priors must match the frozen training views')
    outputs=[]
    for r,v in zip(manifest['records'],views):
        if Path(r['file']).name!=r['file']:raise ValueError('Prior files must be basenames')
        if r['image_sha256']!=v['image_sha256']:raise ValueError('Depth prior RGB hash mismatch')
        p=root/r['file']
        if sha256(p)!=r['sha256']:raise ValueError('Depth prior file hash mismatch')
        with np.load(p) as a:target=a['inverse_z'].copy();mask=a['valid'].copy()
        if target.shape!=(v['height'],v['width']) or mask.shape!=target.shape or mask.dtype!=np.bool_:
            raise ValueError('Depth prior dimensions/mask mismatch')
        if not np.isfinite(target).all() or (target[mask]<=0).any():raise ValueError('Invalid depth prior')
        scale=r['alignment']['normalizer']
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid depth normalizer')
        outputs.append({'target':torch.from_numpy(target).cuda(),'mask':torch.from_numpy(mask).cuda(),'normalizer':scale})
    return outputs,manifest
