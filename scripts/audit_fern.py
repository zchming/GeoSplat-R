"""Offline fern camera/coverage audit; metadata observations never enter training."""
import argparse
import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation
from fetch_fern import fetch_range


def parse_images(data):
    stream=io.BytesIO(data)
    def read(fmt):
        return struct.unpack(fmt,stream.read(struct.calcsize(fmt)))
    records=[]
    for _ in range(read('<Q')[0]):
        v=read('<i7di');name=bytearray()
        while (c:=stream.read(1))!=b'\0':
            if not c:raise ValueError('Unterminated image name')
            name.extend(c)
        count=read('<Q')[0]
        # Skip all feature observations. Never expose 2D/3D test measurements.
        stream.seek(24*count,1)
        w2c=np.eye(4);w2c[:3,:3]=Rotation.from_quat([*v[2:5],v[1]]).as_matrix();w2c[:3,3]=v[5:8]
        records.append({'id':v[0],'name':name.decode(),'camera_id':v[8],'c2w':np.linalg.inv(w2c)})
    assert stream.tell()==len(data)
    return sorted(records,key=lambda x:x['name'])


def member(url,entry):
    _,compressed,length,offset=entry
    header=fetch_range(url,offset,offset+29)
    h=struct.unpack('<4s5H3I2H',header)
    start=offset+30+h[-2]+h[-1]
    block=fetch_range(url,start,start+compressed-1)
    raw=zlib.decompress(block,-15) if h[3]==8 else block
    assert len(raw)==length
    if h[6]:assert zlib.crc32(raw)==h[6]
    return raw,{'member':entry[0],'compressed_bytes':compressed,'raw_bytes':length,
                'sha256':hashlib.sha256(raw).hexdigest()}


def archive_index(url, total_bytes):
    tail=fetch_range(url,total_bytes-65536,total_bytes-1)
    end=struct.unpack_from('<4s4H2IH',tail,tail.rfind(b'PK\x05\x06'))
    size,offset=end[5:7];cd=fetch_range(url,offset,offset+size-1)
    cursor=0;entries=[]
    while cursor<len(cd):
        h=struct.unpack_from('<4s6H3I5H2I',cd,cursor)
        assert h[0]==b'PK\x01\x02'
        nl,xl,cl=h[10:13];name=cd[cursor+46:cursor+46+nl].decode()
        entries.append([name,h[8],h[9],h[-1]]);cursor+=46+nl+xl+cl
    return {'url':url,'archive_bytes':total_bytes,'entries':entries}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive-index',type=Path)
    ap.add_argument('--url')
    ap.add_argument('--archive-bytes',type=int)
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.archive_index:
        idx=json.loads(args.archive_index.read_text())
    else:
        if not args.url or not args.archive_bytes:
            ap.error('Provide --archive-index or both --url and --archive-bytes')
        idx=archive_index(args.url,args.archive_bytes)
    entries={x[0]:x for x in idx['entries']}
    (args.output/'archive-index.json').write_text(json.dumps(idx,indent=2))
    prefix='nerf_llff_data/fern/sparse/0/'
    image_raw,image_prov=member(idx['url'],entries[prefix+'images.bin'])
    cameras_raw,camera_prov=member(idx['url'],entries[prefix+'cameras.bin'])
    official=parse_images(image_raw)
    count=struct.unpack_from('<Q',cameras_raw)[0];assert count==1
    camera_id,model_id,width,height=struct.unpack_from('<iiQQ',cameras_raw,8)
    params=np.frombuffer(cameras_raw,dtype='<f8',offset=32).tolist()
    assert model_id in [0,1,2], 'Unsupported camera model'
    print('Official camera',model_id,params,flush=True)
    # Do not persist source images.bin: it contains full-view feature observations.
    poses=np.load(args.raw/'poses_bounds.npy')[:,:15].reshape(-1,3,5)
    assert len(poses)==len(official)==20
    differences=[]
    for i,(pose,record) in enumerate(zip(poses,official)):
        c2w=np.eye(4);c2w[:3,:3]=pose[:,[1,0,2]]*np.array([1,1,-1]);c2w[:3,3]=pose[:,3]
        delta=float(np.max(np.abs(c2w-record['c2w'])))
        differences.append({'downsampled_name':f'image{i:03d}.png','original_name':record['name'],'matrix_max_abs_error':delta})
        assert delta<1e-8, differences[-1]
        assert np.allclose(pose[:,4],[height,width,params[0]],atol=1e-8)
    points=np.load(args.data/'train/points.npz')['points']
    train=json.loads((args.data/'train/cameras.json').read_text())
    test=json.loads((args.data/'test/cameras.json').read_text())
    centers=np.array([x['c2w'] for x in train])[:,:3,3]
    if model_id in [0,2]:fx,fy,cx,cy=params[0],params[0],params[1],params[2]
    else:fx,fy,cx,cy=params
    intrinsic_errors=[]
    for r in train+test:
        source_K=np.array([[fx*r['width']/width,0,cx*r['width']/width],
                          [0,fy*r['height']/height,cy*r['height']/height],[0,0,1]])
        intrinsic_errors.append(float(np.max(np.abs(source_K-np.array(r['K'])))))
    assert max(intrinsic_errors)<1e-8
    out=[]
    for split,views in [('train',train),('test',test)]:
        for r in views:
            c2w=np.array(r['c2w']);K=np.array(r['K']);camera_xyz=(points-c2w[:3,3])@c2w[:3,:3]
            proj=camera_xyz@K.T;xy=proj[:,:2]/proj[:,2:]
            valid=(camera_xyz[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<r['width'])&(xy[:,1]>=0)&(xy[:,1]<r['height'])
            visible=xy[valid]
            hull=ConvexHull(visible).volume/(r['width']*r['height'])
            cells=np.floor(visible/[r['width'],r['height']]*[16,12]).astype(int)
            gy,gx=np.mgrid[0:r['height']:8,0:r['width']:8]
            distances=cKDTree(visible).query(np.stack([gx.ravel(),gy.ravel()],1))[0]
            nearest=int(np.argmin(np.linalg.norm(centers-c2w[:3,3],axis=1)))
            angles=[float(np.degrees(np.arccos(np.clip(np.dot(c2w[:3,2],np.array(t['c2w'])[:3,2]),-1,1)))) for t in train]
            out.append({'split':split,'image':r['name'],'positive_in_frame_initial_points':int(valid.sum()),
                        'projected_point_hull_image_fraction':float(hull),
                        'occupied_grid_fraction_16x12':len(np.unique(cells,axis=0))/192,
                        'nearest_projected_point_distance_p50_p90_px':np.percentile(distances,[50,90]).tolist(),
                        'nearest_training_camera':train[nearest]['name'],
                        'nearest_training_camera_distance_normalized':float(np.linalg.norm(centers[nearest]-c2w[:3,3])),
                        'minimum_training_forward_axis_angle_degrees':min(angles)})
    distortion={}
    if model_id==2:
        f,cx,cy,k=params
        gy,gx=np.mgrid[0:378:1,0:504:1]
        x=(gx*8-cx)/f;y=(gy*8-cy)/f
        radial=k*(x*x+y*y)
        displacement=np.sqrt((f/8*x*radial)**2+(f/8*y*radial)**2)
        distortion={'ignored_simple_radial_k':k,'p50_p95_max_shift_px':[
            float(np.median(displacement)),float(np.percentile(displacement,95)),float(displacement.max())],
            'interpretation':'Nominal pinhole-to-source distortion field, not measured correspondence error'}
    result={'camera_source':{'model_id':model_id,'width':width,'height':height,'params':params,'distortion':distortion},
            'downloaded_metadata_provenance':[image_prov,camera_prov],
            'mapping_max_abs_error':max(x['matrix_max_abs_error'] for x in differences),
            'intrinsics_max_abs_error':max(intrinsic_errors),
            'mapping':differences,'initial_point_count':len(points),'projection_diagnostics':out,
            'scope':'Diagnostic only; source feature observations skipped, no source points downloaded, no training inputs modified',
            'limitations':'Projection/grid/hull statistics ignore occlusion and are not depth accuracy or a causal ablation'}
    (args.output/'camera-and-coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
