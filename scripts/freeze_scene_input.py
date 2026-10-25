"""Freeze existing train-only LLFF assets for the supported development scenes.

This CPU command records prepared RGB/camera/point identities. It does not
triangulate, infer depth, change assets, read test views or initialize CUDA.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geosplat_r.scene_input import SCENE_TRAINS, sha, validate_scene_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--scene', required=True, choices=sorted(SCENE_TRAINS))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if data.name != 'train' or output != data / 'input-manifest.json':
        raise ValueError('Use an explicit train directory and its input-manifest.json')
    if output.exists():
        raise FileExistsError(output)
    views = json.loads((data / 'cameras.json').read_text())
    if [v['name'] for v in views] != SCENE_TRAINS[args.scene]:
        raise ValueError('Prepared views differ from the frozen training selection')
    for view in views:
        path = data / 'images' / view['name']
        if sha(path) != view['image_sha256']:
            raise ValueError('Prepared RGB hash mismatch')
        with Image.open(path) as image:
            if image.size != (504, 378) or [view['width'], view['height']] != [504, 378]:
                raise ValueError('Expected the native 504x378 prepared images')
    with np.load(data / 'points.npz', allow_pickle=False) as values:
        points, colors = values['points'], values['rgb']
        if (points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape
                or len(points) < 4 or not np.isfinite(points).all()):
            raise ValueError('Invalid train-only initialization')
        point_count = len(points)
    manifest = dict(scene=args.scene, protocol=args.scene + '-undistorted-v1',
                    split='train', train_images=[{'name': v['name'], 'sha256': v['image_sha256']}
                                                 for v in views],
                    resolution=[504, 378], cameras_sha256=sha(data / 'cameras.json'),
                    points_sha256=sha(data / 'points.npz'), initial_point_count=point_count)
    with output.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    validate_scene_input(data, views, output)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
