"""Render imported FSGS parameters using the pinned native visibility rule.

FSGS's CUDA in_frustum culls centers at Z <= 0.2, independently of the camera
projection's znear. The standard project renderer uses Z < 0.01 instead.
This explicit adapter leaves the standard renderer and frozen evaluations alone.
"""
import torch
from gsplat.rendering import rasterization

# gsplat rejects Z < near_plane; the next float32 after 0.2 implements <= 0.2.
NATIVE_NEAR_PLANE = 0.20000001788139343


def render_native_compatible(checkpoint, view, degree=3, depth=False):
    if checkpoint.get('method') != 'FSGS native training; gsplat rendering':
        raise ValueError('Expected an explicitly converted FSGS checkpoint')
    params = checkpoint['splats']
    return rasterization(means=params['means'], quats=params['quats'],
                         scales=params['scales'].exp(), opacities=params['opacities'].sigmoid(),
                         colors=torch.cat([params['sh0'], params['shN']], dim=1),
                         viewmats=view['view'], Ks=view['intrinsics'],
                         width=view['width'], height=view['height'], packed=True,
                         sh_degree=degree, near_plane=NATIVE_NEAR_PLANE, far_plane=1e10,
                         render_mode='RGB+ED' if depth else 'RGB', rasterize_mode='classic')
