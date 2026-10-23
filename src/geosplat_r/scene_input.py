"""Explicit train-only input manifests for the approved LLFF scenes."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SCENE_TRAINS = {'fortress':['image001.png', 'image021.png', 'image041.png'],
                'flower':['image001.png','image017.png','image033.png'],
                'leaves':['image001.png','image012.png','image025.png'],
                'room':['DJI_20200226_143851_396.png','DJI_20200226_143918_576.png','DJI_20200226_143946_704.png']}
SCENE_TESTS = {'fortress':['image000.png', 'image008.png', 'image016.png', 'image024.png', 'image032.png', 'image040.png'],
               'flower':['image000.png','image008.png','image016.png','image024.png','image032.png'],
               'leaves':['image000.png','image008.png','image016.png','image024.png'],
               'room':['DJI_20200226_143850_006.png','DJI_20200226_143901_419.png','DJI_20200226_143912_008.png','DJI_20200226_143926_241.png','DJI_20200226_143938_168.png','DJI_20200226_143948_113.png']}


def same_colored_point_multiset(points, colors, expected, expected_colors):
    """COLMAP hash-map iteration order may change on serialization; retain multiplicity."""
    def rows(xyz,rgb):
        return Counter(np.asarray(p,dtype=np.float32).tobytes()+np.asarray(c,dtype=np.uint8).tobytes()
                       for p,c in zip(xyz,rgb))
    if len(points)!=len(colors) or len(expected)!=len(expected_colors):
        return False
    return rows(points,colors)==rows(expected,expected_colors)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_scene_input(data, views, manifest_path):
    """Reject a test/parent manifest; training consumes only its own input directory."""
    data = Path(data).resolve()
    if Path(manifest_path).resolve() != data/'input-manifest.json':
        raise ValueError('Scene input manifest must be inside the explicit training directory')
    manifest = json.loads(Path(manifest_path).read_text())
    fields = {'scene','protocol','split','train_images','resolution','cameras_sha256',
              'points_sha256','initial_point_count'}
    if set(manifest) != fields or manifest['split'] != 'train':
        raise ValueError('Only the train-only input manifest schema is accepted')
    scene = manifest['scene']
    if scene not in SCENE_TRAINS or manifest['protocol'] != scene+'-undistorted-v1':
        raise ValueError('Scene/protocol outside the approved development inputs')
    if [v['name'] for v in views] != SCENE_TRAINS[scene]:
        raise ValueError('Frozen training views differ from the approved scene')
    if manifest['train_images'] != [{'name':v['name'],'sha256':v['image_sha256']} for v in views]:
        raise ValueError('Train RGB provenance mismatch')
    if manifest['resolution'] != [504,378] or any(
            [v['width'],v['height']] != manifest['resolution'] for v in views):
        raise ValueError('Frozen native image dimensions differ')
    for name,key in [('cameras.json','cameras_sha256'),('points.npz','points_sha256')]:
        if sha(data/name) != manifest[key]:
            raise ValueError('Scene input camera/point hash mismatch')
    with np.load(data/'points.npz',allow_pickle=False) as values:
        points,colors = values['points'],values['rgb']
        if (points.shape != (manifest['initial_point_count'],3) or colors.shape != points.shape
                or len(points)<4 or not np.isfinite(points).all()):
            raise ValueError('Invalid train-only initialization')
    return manifest
