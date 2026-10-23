"""Apply the frozen fern remapping rule to explicitly selected LLFF subsets."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.camera import pinhole_remap,forward_simple_radial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['source','plan','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--scene',choices=['flower','leaves','room','fortress'],required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    selection = next(s for s in json.loads(args.plan.read_text())['scenes'] if s['scene']==args.scene)
    source = json.loads((args.source/'download.json').read_text())
    if source['status'] != 'passed' or any(source[k]!=selection[k] for k in ['scene','train_indices','test_indices','unused_indices']):
        raise ValueError('Downloaded source differs from frozen scene selection')
    camera = json.loads((args.source/'calibration.json').read_text())['camera_source']
    size = (504,378)
    K,D,N,x,y,roi = pinhole_remap(camera,size)
    gy,gx = np.mgrid[0:size[1],0:size[0]]
    independent = forward_simple_radial(np.stack([gx,gy],axis=-1),K,N,D[0])
    error = float(np.max(np.abs(independent-np.stack([x,y],axis=-1))))
    if error>=5e-5 or not np.all(cv2.remap(np.full((378,504),255,np.uint8),x,y,
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)==255):
        raise ValueError('Frozen analytic/no-padding remap checks failed; protocol review required')
    records = []
    for split in ['train','test']:
        out = args.output/split/'images'
        out.mkdir(parents=True)
        for name in selection[split+'_names']:
            local = f'{split}/images/{name}'
            entry = next(r for r in source['files'] if r['local']==local)
            path = args.source/local
            if sha(path)!=entry['sha256']:
                raise ValueError('Source RGB hash mismatch')
            image = np.asarray(Image.open(path).convert('RGB'))
            if image.shape!=(378,504,3):
                raise ValueError('Native dimension mismatch')
            Image.fromarray(cv2.remap(image,x,y,cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)).save(out/name)
            records.append({'local':local,'source_sha256':sha(path),'sha256':sha(out/name)})
    shutil.copyfile(args.source/'poses_bounds.npy',args.output/'poses_bounds.npy')
    records.append({'local':'poses_bounds.npy','sha256':sha(args.output/'poses_bounds.npy')})
    protocol = args.scene+'-undistorted-v1'
    manifest = {k:source[k] for k in ['all_image_names','train_indices','test_indices','unused_indices']}
    manifest.update({'protocol':protocol,'source_manifest_sha256':sha(args.source/'download.json'),'files':records})
    (args.output/'download.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'intrinsics.json').write_text(json.dumps({'protocol':protocol,'K':N.tolist(),
        'width':504,'height':378,'distortion':[0.,0.,0.,0.]},indent=2)+'\n')
    evidence = {'protocol':protocol,'calibration_sha256':sha(args.source/'calibration.json'),
        'opencv':cv2.__version__,'source_K':K.tolist(),'source_distortion':D.tolist(),'new_K':N.tolist(),
        'size':list(size),'alpha':0.,'center_principal_point':True,'focal_guard_factor':1.001,
        'interpolation':'INTER_LINEAR, uint8 sRGB','cropping':False,'evaluation_mask':False,
        'padding_pixels':0,'opencv_roi_reported':list(roi),
        'analytic_forward_map_max_abs_error_px':error,
        'selection_inputs':'Public calibration and native dimensions only; no image statistics or metrics'}
    (args.output/'protocol.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    main()
