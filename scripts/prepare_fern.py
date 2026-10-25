"""Known-camera COLMAP triangulation using only the three frozen training RGBs."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--intrinsics', type=Path, help='Explicit pinhole intrinsics for a derived image protocol')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter(); wall_tick=time.time()
    intrinsics=json.loads(args.intrinsics.read_text()) if args.intrinsics else None
    manifest = json.loads((args.source/'download.json').read_text())
    poses = np.load(args.source/'poses_bounds.npy')[:, :15].reshape(-1, 3, 5)
    names = manifest['all_image_names']
    # LLFF c2w columns: down, right, backward -> OpenCV: right, down, forward.
    def camera(index):
        pose = poses[index]
        c2w = np.eye(4)
        c2w[:3, :3] = pose[:, [1, 0, 2]] * np.array([1, 1, -1])
        c2w[:3, 3] = pose[:, 3]
        assert np.allclose(c2w[:3, :3].T@c2w[:3, :3], np.eye(3), atol=1e-5)
        H, W, f = pose[:, 4]
        w, h = int(W)//8, int(H)//8
        K = np.array([[f*w/W, 0, w/2], [0, f*h/H, h/2], [0, 0, 1]])
        if intrinsics:
            assert [w,h]==[intrinsics['width'],intrinsics['height']]
            assert np.allclose(intrinsics['distortion'],0)
            K=np.array(intrinsics['K'],dtype=np.float64)
            assert K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
        return {'name': names[index], 'source_index': index, 'width': w, 'height': h,
                'K': K.tolist(), 'c2w': c2w.tolist()}
    for split in ['train', 'test']:
        target = args.output/split/'images'; target.mkdir(parents=True)
        ids = manifest[split+'_indices']
        records = [camera(i) for i in ids]
        for record in records:
            name = record['name']
            shutil.copyfile(args.source/split/'images'/name, target/name)
        (target.parent/'cameras.json').write_text(json.dumps(records, indent=2))
    images = args.output/'train/images'
    train_cameras = json.loads((args.output/'train/cameras.json').read_text())
    K = np.array(train_cameras[0]['K'])
    reader = pycolmap.ImageReaderOptions(camera_model='PINHOLE',
              camera_params=','.join(map(str,[K[0,0],K[1,1],K[0,2],K[1,2]])))
    extract = pycolmap.FeatureExtractionOptions(num_threads=4, use_gpu=False)
    db_path = args.output/'train/database.db'
    pycolmap.extract_features(db_path, images, image_names=[x['name'] for x in train_cameras],
                             camera_mode=pycolmap.CameraMode.SINGLE,
                             reader_options=reader, extraction_options=extract, device=pycolmap.Device.cpu)
    match = pycolmap.FeatureMatchingOptions(num_threads=4, use_gpu=False)
    pycolmap.match_exhaustive(db_path, matching_options=match, device=pycolmap.Device.cpu)
    with pycolmap.Database.open(db_path) as db:
        db_images = db.read_all_images()
        db_cameras = db.read_all_cameras()
    assert {im.name for im in db_images} == {x['name'] for x in train_cameras}
    reference = args.output/'train/reference'; reference.mkdir()
    lines = [f'{c.camera_id} {c.model_name} {c.width} {c.height} '+ ' '.join(map(str,c.params))
             for c in db_cameras]
    (reference/'cameras.txt').write_text('\n'.join(lines)+'\n')
    by_name = {r['name']:r for r in train_cameras}; lines=[]
    for im in db_images:
        w2c = np.linalg.inv(by_name[im.name]['c2w'])
        q = pycolmap.Rotation3d(w2c[:3,:3]).quat
        values = [im.image_id, q[3], *q[:3], *w2c[:3,3], im.camera_id, im.name]
        lines.append(' '.join(map(str,values))+'\n')
    (reference/'images.txt').write_text('\n'.join(lines)+'\n')
    (reference/'points3D.txt').write_text('')
    reconstruction = pycolmap.Reconstruction(reference)
    options = pycolmap.IncrementalPipelineOptions(num_threads=4, random_seed=42,
              ba_refine_focal_length=False, ba_refine_principal_point=False,
              ba_refine_extra_params=False, fix_existing_frames=True)
    options.mapper.fix_existing_frames=True
    options.triangulation.ignore_two_view_tracks=False
    options.triangulation.random_seed=42
    out_model = args.output/'train/triangulated'; out_model.mkdir()
    result = pycolmap.triangulate_points(reconstruction, db_path, images, out_model,
                                         options=options, refine_intrinsics=False)
    assert result.num_points3D() >= 4, 'Too few triangulated points; no random fallback'
    # Reject any unexpected camera refinement and any track outside the selected RGBs.
    ids = {im.image_id for im in db_images}
    for im in result.images.values():
        assert np.allclose(im.cam_from_world().matrix(),
                           np.linalg.inv(by_name[im.name]['c2w'])[:3], atol=1e-6)
    points=list(result.points3D.values())
    for point in points:
        assert {e.image_id for e in point.track.elements}.issubset(ids)
    # Normalize using training camera centers only; test cameras get the frozen same transform.
    centers=np.array([r['c2w'] for r in train_cameras])[:,:3,3]
    center=centers.mean(0); scale=1/np.linalg.norm(centers-center,axis=1).max()
    xyz=(np.array([p.xyz for p in points])-center)*scale
    rgb=np.array([p.color for p in points],dtype=np.uint8)
    np.savez(args.output/'train/points.npz',points=xyz.astype(np.float32),rgb=rgb)
    for split in ['train','test']:
        p=args.output/split/'cameras.json'; records=json.loads(p.read_text())
        for r in records:
            matrix=np.array(r['c2w']);matrix[:3,3]=(matrix[:3,3]-center)*scale;r['c2w']=matrix.tolist()
            p_img=args.output/split/'images'/r['name'];r['image_sha256']=hashlib.sha256(p_img.read_bytes()).hexdigest()
        p.write_text(json.dumps(records,indent=2)+'\n')
    stats={'pycolmap':pycolmap.__version__, 'train_names':[r['name'] for r in train_cameras],
           'test_names':[names[i] for i in manifest['test_indices']], 'point_count':len(points),
           'normalization':{'center':center.tolist(),'scale':scale,'source':'3 training camera centers only'},
           'initialization':'SIFT CPU, exhaustive matching, fixed-camera COLMAP triangulation; two-view tracks allowed',
           'points_reprojection_mean_px':float(np.mean([p.error for p in points])),
           'elapsed_perf_counter_seconds':time.perf_counter()-tick,
           'wall_clock_seconds':time.time()-wall_tick,
           'intrinsics_protocol':intrinsics['protocol'] if intrinsics else 'legacy-pinhole-approximation',
           'extraction':extract.todict(),'matching':match.todict(),'triangulation_options':options.todict()}
    (args.output/'preparation.json').write_text(json.dumps(stats,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in stats.items() if not k.endswith('options')},indent=2,default=str))


if __name__=='__main__':
    main()
