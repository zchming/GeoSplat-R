"""Build a separate, frozen fern pinhole image protocol from audited calibration."""
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
from geosplat_r.camera import forward_simple_radial,pinhole_remap


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ['source','calibration','output']:ap.add_argument('--'+field,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    audited=json.loads(args.calibration.read_text())
    camera=audited['camera_source'];size=(504,378)
    K,D,N,x,y,roi=pinhole_remap(camera,size)
    gy,gx=np.mgrid[0:size[1],0:size[0]]
    independent=forward_simple_radial(np.stack([gx,gy],axis=-1),K,N,D[0])
    error=float(np.max(np.abs(independent-np.stack([x,y],axis=-1))))
    assert error<5e-5,error
    marker=np.full((size[1],size[0]),255,dtype=np.uint8)
    assert np.all(cv2.remap(marker,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)==255)
    source_manifest=json.loads((args.source/'download.json').read_text())
    assert source_manifest['train_indices']==[1,10,19] and source_manifest['test_indices']==[0,8,16]
    records=[]
    for split in ['train','test']:
        out=args.output/split/'images';out.mkdir(parents=True)
        for i in source_manifest[split+'_indices']:
            name=source_manifest['all_image_names'][i];p=args.source/split/'images'/name
            # Source provenance must agree before processing any input image.
            entry=next(r for r in source_manifest['files'] if r['local']==f'{split}/images/{name}')
            assert digest(p)==entry['sha256']
            image=np.array(Image.open(p).convert('RGB'))
            assert image.shape==(378,504,3)
            remapped=cv2.remap(image,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            Image.fromarray(remapped).save(out/name)
            records.append({'local':f'{split}/images/{name}','source_sha256':digest(p),'sha256':digest(out/name)})
    shutil.copyfile(args.source/'poses_bounds.npy',args.output/'poses_bounds.npy')
    records.append({'local':'poses_bounds.npy','sha256':digest(args.output/'poses_bounds.npy')})
    manifest={k:source_manifest[k] for k in ['all_image_names','train_indices','test_indices','unused_indices']}
    manifest.update({'protocol':'fern-undistorted-v1','source_manifest_sha256':digest(args.source/'download.json'),
                     'files':records,'intrinsics_file':'intrinsics.json'})
    (args.output/'download.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'intrinsics.json').write_text(json.dumps({'protocol':'fern-undistorted-v1',
        'K':N.tolist(),'width':504,'height':378,'distortion':[0.,0.,0.,0.]},indent=2)+'\n')
    protocol={'name':'fern-undistorted-v1','calibration_sha256':digest(args.calibration),
        'opencv':cv2.__version__,'source_K':K.tolist(),'source_distortion':D.tolist(),'new_K':N.tolist(),
        'size':[504,378],'alpha':0.,'center_principal_point':True,'focal_guard_factor':1.001,
        'opencv_roi_reported':list(roi),'interpolation':'INTER_LINEAR, uint8 sRGB',
        'cropping':False,'evaluation_mask':False,'padding_pixels':0,
        'map_bounds':[float(x.min()),float(x.max()),float(y.min()),float(y.max())],
        'analytic_forward_map_max_abs_error_px':error,
        'selection_inputs':'camera calibration and image dimensions only; no image content or test metrics',
        'caveat':'Resampled test GT defines a new evaluation protocol; cross-protocol differences are not algorithm gains'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
