"""CPU-only measured-track geometry for a frozen known-camera sensor diagnostic.

No labels, depth models, full-scene reconstructions or fitting to reference geometry.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1.1e-6:
        raise ValueError('Invalid rounded Hamilton quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def work_to_native_corner(xy):
    return 6 * (np.asarray(xy, dtype=np.float64) + .5)


def native_center_to_work(ij):
    return (np.asarray(ij, dtype=np.float64) + .5) / 6 - .5


def forward(params, xyz):
    """Independent full THIN_PRISM_FISHEYE forward projection, corner coordinates."""
    a = np.asarray(xyz, dtype=np.float64)
    if a.shape[-1] != 3 or not np.isfinite(a).all() or (a[..., 2] <= 0).any():
        raise ValueError('Projection requires positive finite camera Z')
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx, sy = params
    uv = a[..., :2] / a[..., 2:]
    r = np.linalg.norm(uv, axis=-1)
    factor = np.ones_like(r)
    np.divide(np.arctan(r), r, out=factor, where=r > 1e-15)
    u, v = np.moveaxis(uv * factor[..., None], -1, 0)
    r2 = u*u + v*v
    radial = k1*r2 + k2*r2**2 + k3*r2**3 + k4*r2**4
    du = u*radial + 2*p1*u*v + p2*(r2+2*u*u) + sx*r2
    dv = v*radial + 2*p2*u*v + p1*(r2+2*v*v) + sy*r2
    return np.stack([fx*(u+du)+cx, fy*(v+dv)+cy], axis=-1)


def camera_arrays(view):
    spec = view['camera']
    if spec['model'] != 'THIN_PRISM_FISHEYE' or [spec['width'], spec['height']] != [6048, 4032]:
        raise ValueError('Frozen native camera contract mismatch')
    params = np.array(spec['params'], dtype=np.float64)
    if params.shape != (12,) or not np.isfinite(params).all() or (params[:2] <= 0).any():
        raise ValueError('Invalid camera parameters')
    params[:4] /= 6
    r = rotation(view['qvec_wxyz'])
    t = np.array(view['tvec'], dtype=np.float64)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('Invalid camera translation')
    return params, r, t


def components(edges):
    """Union measured feature identifiers; reject ambiguous components wholesale."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for a, b in sorted(edges):
        ra, rb = find(tuple(a)), find(tuple(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for node in sorted(parent):
        groups.setdefault(find(node), []).append(node)
    accepted, rejected = [], []
    for group in sorted(groups.values()):
        if len({i for i, _ in group}) != len(group):
            rejected.append(group)
        elif len(group) in (2, 3):
            accepted.append(group)
        else:
            raise ValueError('Unexpected selected-view track cardinality')
    return accepted, rejected


def triangulate(observations, views, rays):
    """Measured rays -> DLT -> positive-Z/angle/reprojection/border gates."""
    rows = []
    for (i, _), ray in zip(observations, rays):
        _, r, t = camera_arrays(views[i])
        p = np.column_stack([r, t])
        rows.extend([ray[0] * p[2] - p[0], ray[1] * p[2] - p[1]])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1]
    if not np.isfinite(h).all() or abs(h[3]) < 1e-12:
        return None, 'nonfinite_homogeneous'
    xyz = h[:3] / h[3]
    details, directions = [], []
    for (i, feature), ray in zip(observations, rays):
        params, r, t = camera_arrays(views[i])
        q = r @ xyz + t
        if not np.isfinite(q).all() or q[2] <= 0:
            return None, 'nonpositive_Z'
        xy = np.asarray(views[i]['xy'][feature], dtype=np.float64)
        if not (2 <= xy[0] <= 1005 and 2 <= xy[1] <= 669):
            return None, 'measurement_border'
        projected = forward(params, q) - .5
        error = float(np.linalg.norm(projected - xy))
        if error > 2:
            return None, 'reprojection_above_2px'
        world_ray = r.T @ np.r_[ray, 1.]
        directions.append(world_ray / np.linalg.norm(world_ray))
        details.append(dict(image_index=i, feature_index=feature, xy=xy.tolist(),
                            Z_m=float(q[2]), reprojection_work_px=error))
    angles = [np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))
              for j, a in enumerate(directions) for b in directions[:j]]
    angle = float(max(angles))
    if angle < 1.5:
        return None, 'angle_below_1_5deg'
    return dict(xyz_m=xyz.tolist(), observations=details, max_ray_angle_deg=angle), None


def select_anchors(tracks, views):
    rows = [[] for _ in views]
    for track in tracks:
        signature = tuple((x['image_index'], x['feature_index']) for x in track['observations'])
        for obs in track['observations']:
            rows[obs['image_index']].append(dict(**obs, track_signature=signature))
    result = []
    for view, candidates in zip(views, rows):
        chosen = {}
        for row in sorted(candidates, key=lambda x: (x['Z_m'], x['track_signature'])):
            pixel = tuple(np.floor(np.asarray(row['xy']) + .5).astype(int))
            chosen.setdefault(pixel, row)
        anchors = list(chosen.values())
        grid = np.zeros((9, 12), dtype=int)
        for row in anchors:
            x, y = row['xy']
            grid[min(int(y*9/672), 8), min(int(x*12/1008), 11)] += 1
        cells = int((grid > 0).sum())
        result.append(dict(name=view['name'], anchors=anchors, candidate_observations=len(candidates),
                           collisions_removed=len(candidates)-len(anchors), anchor_count=len(anchors),
                           grid_counts=grid.tolist(), occupied_cells=cells,
                           support_passed=len(anchors) >= 60 and cells >= 12))
    return result


def validate_predictor(data, config):
    """Exactly three RGB and one camera manifest, no links or label-root arguments."""
    data = Path(data)
    paths = {'cameras.json'} | {'rgb/' + name for name in config['names']}
    files = {str(p.relative_to(data)) for p in data.rglob('*') if p.is_file()}
    if files != paths or any(p.is_symlink() for p in data.rglob('*')):
        raise ValueError('Predictor inventory must contain exactly four regular assets')
    accepted = []
    for rel in sorted(paths):
        p = data / rel
        spec = config['inputs'][rel]
        if p.stat().st_size != spec['bytes'] or sha256(p) != spec['sha256']:
            raise ValueError('Predictor asset digest mismatch: ' + rel)
        accepted.append(dict(path=rel, **spec))
    views = json.loads((data / 'cameras.json').read_text())
    if [v['name'] for v in views] != config['names'] or len({v['image_id'] for v in views}) != 3:
        raise ValueError('Selected camera mapping mismatch')
    for view in views:
        camera_arrays(view)
    return views, accepted


def sample_inverse_z(q, xy, threshold=1e-6):
    """Bilinear inverse-Z sampling, four positive contributors; never fill holes."""
    q = np.asarray(q, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    if q.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Expected inverse-Z image and N by 2 work coordinates')
    finite = np.isfinite(xy).all(axis=1)
    safe = np.where(np.isfinite(xy), xy, 0)
    x0, y0 = np.floor(safe).astype(np.int64).T
    inside = finite & (x0 >= 0) & (y0 >= 0) & (x0+1 < q.shape[1]) & (y0+1 < q.shape[0])
    x = np.clip(x0, 0, q.shape[1]-2)
    y = np.clip(y0, 0, q.shape[0]-2)
    values = np.stack([q[y, x], q[y, x+1], q[y+1, x], q[y+1, x+1]], axis=1)
    valid = inside & np.isfinite(values).all(axis=1) & (values > threshold).all(axis=1)
    dx, dy = safe.T - np.array([x0, y0])
    weights = np.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], axis=1)
    result = np.full(len(xy), np.nan)
    result[valid] = np.sum(values[valid] * weights[valid], axis=1)
    return result, valid
