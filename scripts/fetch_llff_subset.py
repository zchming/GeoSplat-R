"""Fetch the approved flower/leaves ZIP members, refusing full-archive fallback."""
import argparse
import hashlib
import io
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from audit_fern import parse_images

PLAN_SHA = '9997c5bdf334313730e1527beedadd6e50611d954cc9f101ddfc92fd19e53a5b'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_range(url, start, end, total):
    request = urllib.request.Request(url + f'?subset_range={start}-{end}',
        headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206:
            raise RuntimeError('Range ignored; refusing full archive download')
        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
            raise RuntimeError('Content-Range mismatch')
        data = response.read(end-start+2)
    if len(data) != end-start+1:
        raise RuntimeError('Range length mismatch')
    return data


def fetch_scene(scene, output):
    output.mkdir(parents=True, exist_ok=False)
    start_time = time.time()
    records, requests = [], []
    metadata = {}
    status, error = 'failed', None
    try:
        for entry in scene['selected_members']:
            offset = entry['local_header_offset']
            header = read_range(scene['url'], offset, offset+29, scene['archive_bytes'])
            requests.append({'member':entry['name'], 'kind':'header', 'bytes':len(header)})
            h = struct.unpack('<4s5H3I2H', header)
            if (h[0] != b'PK\x03\x04' or h[2] != entry['flags'] or h[3] != entry['method']
                    or h[6] != int(entry['crc32'],16) or h[7] != entry['compressed_bytes']
                    or h[8] != entry['raw_bytes']):
                raise ValueError('Member header differs from approved index')
            first = offset+30+h[-2]+h[-1]
            block = read_range(scene['url'], first, first+entry['compressed_bytes']-1, scene['archive_bytes'])
            requests.append({'member':entry['name'], 'kind':'payload', 'bytes':len(block)})
            raw = zlib.decompress(block, -15)
            if len(raw) != entry['raw_bytes'] or zlib.crc32(raw) != int(entry['crc32'],16):
                raise ValueError('Member content size/CRC mismatch')
            relative = entry['name'].removeprefix(scene['scene']+'/')
            local = None
            if relative.startswith('images_8/'):
                name = Path(relative).name
                split = 'train' if name in scene['train_names'] else 'test'
                if name not in scene[split+'_names']:
                    raise ValueError('Unexpected image outside approved split')
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != (504,378) or image.mode != 'RGB':
                        raise ValueError(f'Native dimensions/mode require protocol review: {image.size}, {image.mode}')
                    image.verify()
                local = f'{split}/images/{name}'
            elif relative == 'poses_bounds.npy':
                local = relative
                metadata[relative] = raw
            elif relative in ['sparse/0/cameras.bin','sparse/0/images.bin']:
                metadata[relative] = raw
            else:
                raise ValueError('Unexpected approved member type')
            if local:
                path = output/local
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(raw)
            records.append({**entry,'local':local,'sha256':digest(raw),
                'raw_persisted':local is not None})
            print(scene['scene'], entry['name'], len(block), flush=True)
        cameras = metadata['sparse/0/cameras.bin']
        if len(cameras) != 64 or struct.unpack_from('<Q',cameras)[0] != 1:
            raise ValueError('Expected one SIMPLE_RADIAL camera; protocol review required')
        camera_id, model, width, height = struct.unpack_from('<iiQQ',cameras,8)
        if model != 2:
            raise ValueError('Camera model requires protocol review')
        parameters = np.frombuffer(cameras,dtype='<f8',offset=32).tolist()
        official = parse_images(metadata['sparse/0/images.bin'])
        poses = np.load(io.BytesIO(metadata['poses_bounds.npy']),allow_pickle=False)[:,:15].reshape(-1,3,5)
        if len(poses) != scene['image_count'] or len(official) != len(poses):
            raise ValueError('Pose/image count mismatch')
        mappings = []
        for index,(pose,image) in enumerate(zip(poses,official)):
            matrix = np.eye(4)
            matrix[:3,:3] = pose[:,[1,0,2]]*np.array([1,1,-1])
            matrix[:3,3] = pose[:,3]
            delta = float(np.max(np.abs(matrix-image['c2w'])))
            if delta >= 1e-8 or image['camera_id'] != camera_id:
                raise ValueError('Public camera/name mapping mismatch')
            if not np.allclose(pose[:,4],[height,width,parameters[0]],atol=1e-8):
                raise ValueError('Public calibration/pose metadata mismatch')
            if [int(width)//8,int(height)//8] != [504,378]:
                raise ValueError('Calibration/native image size mismatch')
            mappings.append({'index':index,'image':scene['image_names'][index],
                'original_name':image['name'],'c2w_max_abs_error':delta})
        calibration = {'scene':scene['scene'],'camera_source':{'model_id':model,'camera_id':camera_id,
            'width':width,'height':height,'params':parameters},'image_mapping':mappings,
            'policy':'Public cameras only; source feature observations skipped, never persisted or used for initialization'}
        (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
        status = 'passed'
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        report = {k:scene[k] for k in ['scene','url','revision','archive_bytes','image_names',
            'train_indices','test_indices','unused_indices']}
        report['all_image_names'] = report.pop('image_names')
        report.update({'status':status,'error':error,'files':records,'requests':requests,
            'downloaded_compressed_bytes':sum(q['bytes'] for q in requests if q['kind']=='payload'),
            'header_bytes':sum(q['bytes'] for q in requests if q['kind']=='header'),
            'elapsed_wall_seconds':time.time()-start_time,'protocol':'public-calibrated native LLFF images_8'})
        (output/'download.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if digest(args.plan.read_bytes()) != PLAN_SHA:
        raise ValueError('Plan differs from the approved 21-member download candidate')
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    for scene in plan['scenes']:
        fetch_scene(scene,args.output/scene['scene'])


if __name__ == '__main__':
    main()
