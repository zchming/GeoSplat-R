"""Audit derived scene geometry and freeze a train-only input manifest; no training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.scene_input import validate_scene_input,same_colored_point_multiset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['data','source','raw','plan','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--scene',choices=['flower','leaves'],required=True)
    args = parser.parse_args()
    manifest_path = args.data/'train/input-manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    args.output.mkdir(parents=True,exist_ok=False)
    scene = next(s for s in json.loads(args.plan.read_text())['scenes'] if s['scene']==args.scene)
    download = json.loads((args.raw/'download.json').read_text())
    derived = json.loads((args.source/'download.json').read_text())
    intrinsics = json.loads((args.source/'intrinsics.json').read_text())
    preparation = json.loads((args.data/'preparation.json').read_text())
    if download['status']!='passed' or derived['source_manifest_sha256']!=sha(args.raw/'download.json'):
        raise ValueError('Raw/derived source provenance mismatch')
    for row in download['files']:
        if row['raw_persisted'] and sha(args.raw/row['local'])!=row['sha256']:
            raise ValueError('Downloaded input content changed')
    poses = np.load(args.raw/'poses_bounds.npy',allow_pickle=False)[:,:15].reshape(-1,3,5)
    original = []
    for pose in poses:
        matrix = np.eye(4)
        matrix[:3,:3] = pose[:,[1,0,2]]*np.array([1,1,-1])
        matrix[:3,3] = pose[:,3]
        original.append(matrix)
    centers = np.array([original[i][:3,3] for i in scene['train_indices']])
    center = centers.mean(0)
    scale = 1/np.linalg.norm(centers-center,axis=1).max()
    np.testing.assert_allclose(preparation['normalization']['center'],center,rtol=0,atol=1e-12)
    np.testing.assert_allclose(preparation['normalization']['scale'],scale,rtol=0,atol=1e-12)
    views = {}
    for split in ['train','test']:
        records = json.loads((args.data/split/'cameras.json').read_text())
        if [v['name'] for v in records]!=scene[split+'_names']:
            raise ValueError('Frozen split/name order differs')
        if sorted(p.name for p in (args.data/split/'images').iterdir())!=sorted(scene[split+'_names']):
            raise ValueError('Unexpected RGB in split directory')
        for record,index in zip(records,scene[split+'_indices']):
            if record['source_index']!=index or [record['width'],record['height']]!=[504,378]:
                raise ValueError('Index/native dimensions differ')
            local = f'{split}/images/{record["name"]}'
            source = next(r for r in derived['files'] if r['local']==local)
            if sha(args.source/local)!=source['sha256'] or sha(args.data/local)!=source['sha256']:
                raise ValueError('Prepared/derived RGB content mismatch')
            if record['image_sha256']!=source['sha256']:
                raise ValueError('Prepared RGB provenance mismatch')
            expected = original[index].copy()
            expected[:3,3] = (expected[:3,3]-center)*scale
            np.testing.assert_allclose(expected,record['c2w'],rtol=0,atol=1e-8)
            np.testing.assert_allclose(intrinsics['K'],record['K'],rtol=0,atol=1e-12)
        views[split] = records
    if set(scene['train_names']) & set(scene['test_names']):
        raise ValueError('Train/test overlap')
    model = pycolmap.Reconstruction(args.data/'train/triangulated')
    if {im.name for im in model.images.values()}!=set(scene['train_names']):
        raise ValueError('Triangulation contains non-training images')
    with pycolmap.Database.open(args.data/'train/database.db') as database:
        if {im.name for im in database.read_all_images()}!=set(scene['train_names']):
            raise ValueError('Feature database contains non-training images')
    for point in model.points3D.values():
        if not all(model.images[e.image_id].name in scene['train_names'] for e in point.track.elements):
            raise ValueError('Triangulated track includes a non-training observation')
    with np.load(args.data/'train/points.npz',allow_pickle=False) as cloud:
        points = cloud['points'].copy()
        colors = cloud['rgb'].copy()
    model_points = list(model.points3D.values())
    expected = (np.array([p.xyz for p in model_points])-center)*scale
    if len(points)!=len(expected) or len(points)<4:
        raise ValueError('Initialization point count mismatch')
    if not same_colored_point_multiset(points,colors,expected,[p.color for p in model_points]):
        raise ValueError('Colored initialization differs from the train triangulation multiset')
    manifest = {'scene':args.scene,'protocol':args.scene+'-undistorted-v1','split':'train',
        'train_images':[{'name':v['name'],'sha256':v['image_sha256']} for v in views['train']],
        'resolution':[504,378],'cameras_sha256':sha(args.data/'train/cameras.json'),
        'points_sha256':sha(args.data/'train/points.npz'),'initial_point_count':len(points)}
    with manifest_path.open('x') as stream:
        stream.write(json.dumps(manifest,indent=2)+'\n')
    validate_scene_input(args.data/'train',views['train'],manifest_path)
    report = {'status':'passed','scene':args.scene,'point_count':len(points),
        'reprojection_mean_px':preparation['points_reprojection_mean_px'],
        'train_names':scene['train_names'],'test_names':scene['test_names'],
        'normalization':preparation['normalization'],'database_images_training_only':True,
        'all_point_tracks_training_only':True,'prepared_pose_and_source_projection_check':'passed',
        'input_manifest_sha256':sha(manifest_path),'test_cameras_sha256':sha(args.data/'test/cameras.json'),
        'test_image_sha256':{v['name']:v['image_sha256'] for v in views['test']},
        'caveat':'Public full-view calibrated cameras; not three-view camera recovery or independent geometry truth'}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
