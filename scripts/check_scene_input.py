"""Load and verify train-only RGB/priors on the GPU without rendering or optimization."""
import argparse
import json
import sys
from pathlib import Path

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.baseline import load_views,sha256
from geosplat_r.depth import load_depth_priors
from geosplat_r.isolation import audit_training_reads
from geosplat_r.scene_input import validate_scene_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data','scene-manifest','priors','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    access=audit_training_reads(args.data)
    torch.cuda.reset_peak_memory_stats()
    views=load_views(args.data)
    scene=validate_scene_input(args.data,views,args.scene_manifest)
    for view in views:
        view['pixels']=view['pixels'].cuda()
    priors,manifest=load_depth_priors(args.priors,views,sha256(args.data/'cameras.json'),
        sha256(args.data/'points.npz'),sha256)
    expected={'cameras.json','points.npz','input-manifest.json',*[f'images/{v["name"]}' for v in views]}
    if set(access)!=expected or len(priors)!=3 or manifest['protocol']!=scene['protocol']:
        raise ValueError('Unexpected training input reads or prior protocol')
    torch.cuda.synchronize()
    result={'status':'passed','scene':scene['scene'],'actual_training_asset_reads':sorted(set(access)),
        'input_manifest_sha256':sha256(args.scene_manifest),'priors_manifest_sha256':sha256(args.priors/'manifest.json'),
        'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'optimizer_updates':0,'rendering_performed':False,'training_performed':False}
    (args.output/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
