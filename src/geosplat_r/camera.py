"""Calibration-only SIMPLE_RADIAL to valid pinhole remapping, with no image statistics."""
import cv2
import numpy as np


def pinhole_remap(camera, size):
    if camera['model_id'] != 2:
        raise ValueError('This protocol requires the audited COLMAP SIMPLE_RADIAL model')
    width,height=size
    f,cx,cy,k=camera['params']
    sx,sy=width/camera['width'],height/camera['height']
    K=np.array([[f*sx,0,cx*sx],[0,f*sy,cy*sy],[0,0,1.]],dtype=np.float64)
    D=np.array([k,0.,0.,0.])
    new_K,roi=cv2.getOptimalNewCameraMatrix(K,D,size,0,size,centerPrincipalPoint=True)
    # Predeclared geometry-only guard against interpolation outside the source frame.
    new_K[:2,:2]*=1.001
    map_x,map_y=cv2.initUndistortRectifyMap(K,D,None,new_K,size,cv2.CV_32FC1)
    if not (np.isfinite(map_x).all() and np.isfinite(map_y).all()):
        raise ValueError('Nonfinite remap')
    if not (map_x.min()>=0 and map_x.max()<width-1 and map_y.min()>=0 and map_y.max()<height-1):
        raise ValueError('Remap would use padding; do not silently crop or change the protocol')
    return K,D,new_K,map_x,map_y,roi


def forward_simple_radial(pixels, source_K, new_K, k):
    homogeneous=np.concatenate([pixels,np.ones((*pixels.shape[:-1],1))],axis=-1)
    normalized=homogeneous@np.linalg.inv(new_K).T
    xy=normalized[...,:2]/normalized[...,2:]
    distorted=xy*(1+k*np.sum(xy*xy,axis=-1,keepdims=True))
    projected=np.concatenate([distorted,np.ones((*distorted.shape[:-1],1))],axis=-1)@source_K.T
    return projected[...,:2]/projected[...,2:]
