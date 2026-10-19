"""Export observed train-only SfM anchors without changing the prepared assets."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pycolmap
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.isolation import audit_training_reads


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['data','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--scene-manifest',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.time();access=audit_training_reads(args.data)
    views=json.loads((args.data/'cameras.json').read_text())
    names=[v['name'] for v in views]
    scene_input=None
    if args.scene_manifest is not None:
        from geosplat_r.scene_input import validate_scene_input
        scene_input=validate_scene_input(args.data,views,args.scene_manifest)
    elif names!=['image001.png','image010.png','image019.png']:raise ValueError('Frozen train split required')
    model=pycolmap.Reconstruction(args.data/'triangulated')
    assert {im.name for im in model.images.values()}==set(names)
    by_name={im.name:im for im in model.images.values()}
    def c2w(im):
        w=np.eye(4);w[:3]=im.cam_from_world().matrix();return np.linalg.inv(w)
    original_centers=np.array([c2w(by_name[n])[:3,3] for n in names])
    center=original_centers.mean(0);scale=1/np.linalg.norm(original_centers-center,axis=1).max()
    for v in views:
        expected=c2w(by_name[v['name']]);expected[:3,3]=(expected[:3,3]-center)*scale
        np.testing.assert_allclose(expected,v['c2w'],rtol=0,atol=1e-6)
    xyz=np.array([(p.xyz-center)*scale for p in model.points3D.values()])
    frozen=np.load(args.data/'points.npz')['points']
    assert len(xyz)==len(frozen)==(scene_input['initial_point_count'] if scene_input else 488)
    nearest=np.sqrt(((xyz[:,None]-frozen[None])**2).sum(-1).min(1)).max()
    assert nearest<1e-5,nearest
    records=[]
    for v in views:
        im=by_name[v['name']];K=np.array(v['K']);matrix=np.array(v['c2w']);rows=[]
        for pid,p in model.points3D.items():
            observed=[e for e in p.track.elements if e.image_id==im.image_id]
            if not observed:continue
            assert all(model.images[e.image_id].name in names for e in p.track.elements)
            world=(p.xyz-center)*scale;camera=(world-matrix[:3,3])@matrix[:3,:3]
            if camera[2]<=0:continue
            xy=(camera@K.T)[:2]/camera[2]
            measurement=np.array(im.points2D[observed[0].point2D_idx].xy)
            if np.linalg.norm(xy-measurement)>2:continue
            x,y=xy;w,h=v['width'],v['height']
            if 2<=x<w-3 and 2<=y<h-3:rows.append((x,y,camera[2],pid))
        # Keep the nearest observed point when anchors land in the same pixel.
        closest={}
        for row in sorted(rows,key=lambda r:r[2]):closest.setdefault(tuple(np.rint(row[:2]).astype(int)),row)
        rows=np.array(list(closest.values()),dtype=np.float64)
        assert len(rows)>=20,(v['name'],len(rows))
        target=args.output/(Path(v['name']).stem+'.npz')
        np.savez(target,xy=rows[:,:2],inverse_z=1/rows[:,2],point_ids=rows[:,3].astype(np.int64))
        records.append({'image':v['name'],'file':target.name,'sha256':sha(target),'anchors':len(rows)})
    result={'protocol':scene_input['protocol'] if scene_input else 'fern-undistorted-v1','cameras_sha256':sha(args.data/'cameras.json'),
        'points_sha256':sha(args.data/'points.npz'),'points':len(frozen),'cloud_max_nearest_distance':float(nearest),
        'selection':'observed training tracks; positive Z; <=2px reprojection; 2px border; nearest per rounded pixel',
        'records':records,'elapsed_wall_seconds':time.time()-start}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'training-access.json').write_text(json.dumps(sorted(set(access)),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
