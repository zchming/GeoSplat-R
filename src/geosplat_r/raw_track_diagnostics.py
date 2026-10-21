"""Fixed-camera raw-observation sensitivity diagnostic, never training weights."""
from collections import Counter
import struct

import numpy as np
from geosplat_r.train_diagnostics import NAMES, describe, fold_for_id, spearman

PLAN_SHA256 = 'b3d4830ae4fb7e92f79c1ff4f793459376c2219a3113bf7d07083d1722e49c77'
PROPOSAL_SHA256 = '116ece1a1a4547200bb32df5a058a4fbb8a7a10fa8fc30f894e4fbb63ff06bf3'
H = 1e-3
EPS = 1e-12


class Binary:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, size):
        if size < 0 or self.pos + size > len(self.data):
            raise ValueError('Truncated COLMAP binary')
        start = self.pos
        self.pos += size
        return self.data[start:self.pos]

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt)))

    def count(self, minimum_bytes):
        n, = self.unpack('Q')
        if n > (len(self.data) - self.pos) // minimum_bytes:
            raise ValueError('Impossible binary count')
        return n

    def name(self):
        end = self.data.find(b'\0', self.pos)
        if end < 0 or end - self.pos > 255:
            raise ValueError('Unterminated/long image name')
        value = self.take(end - self.pos + 1)[:-1].decode('utf8')
        if value not in NAMES:
            raise ValueError('Non-training image in raw model')
        return value

    def done(self):
        if self.pos != len(self.data):
            raise ValueError('Trailing binary data')


def read_cameras(data):
    b, result = Binary(data), {}
    for _ in range(b.count(56)):
        cid, model, width, height = b.unpack('iiQQ')
        if cid <= 0 or cid in result or model != 1 or (width, height) != (504, 378):
            raise ValueError('Expected unique 504x378 PINHOLE cameras')
        fx, fy, cx, cy = b.unpack('dddd')
        if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
            raise ValueError('Invalid camera parameters')
        result[cid] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64)
    b.done()
    return result


def quaternion_rotation(q):
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or not np.isclose(q @ q, 1., rtol=0, atol=1e-8):
        raise ValueError('Invalid Hamilton quaternion')
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def read_images(data):
    b, images = Binary(data), {}
    count = b.count(73)
    if count != 3:
        raise ValueError('Exactly three registered training images required')
    for _ in range(count):
        iid, *values = b.unpack('idddddddi')
        q, t, cid = values[:4], np.array(values[4:7]), values[7]
        name = b.name()
        if iid <= 0 or iid in images or name in [im['name'] for im in images.values()]:
            raise ValueError('Duplicate/nonpositive image ID or name')
        if not np.isfinite(t).all():
            raise ValueError('Invalid image translation')
        rotation = quaternion_rotation(q)
        n = b.count(24)
        points = np.frombuffer(b.take(n * 24), dtype=[('xy', '<f8', (2,)), ('pid', '<i8')])
        if not np.isfinite(points['xy']).all() or (points['pid'] < -1).any():
            raise ValueError('Invalid raw feature observation')
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation.T, -rotation.T @ t
        images[iid] = dict(name=name, camera_id=cid, c2w=pose,
                           xy=points['xy'].copy(), point_ids=points['pid'].copy())
    b.done()
    if set(im['name'] for im in images.values()) != set(NAMES):
        raise ValueError('Wrong training images')
    return images


def read_tracks(data):
    b, tracks = Binary(data), {}
    for _ in range(b.count(59)):
        pid, = b.unpack('Q')
        if pid in tracks or pid > 2**63-1:
            raise ValueError('Duplicate/out-of-range point ID')
        b.take(35)  # XYZ, RGB and fitted error deliberately NOT interpreted.
        n = b.count(8)
        if not 1 <= n <= 3:
            raise ValueError('Track must contain one to three distinct training images')
        entries = [b.unpack('ii') for _ in range(n)]
        if len({iid for iid, _ in entries}) != n or any(idx < 0 for _, idx in entries):
            raise ValueError('Duplicate track image or negative index')
        tracks[pid] = entries
    b.done()
    return tracks


def camera_arrays(record):
    k, c = np.asarray(record['K'], dtype=np.float64), np.asarray(record['c2w'], dtype=np.float64)
    if (k.shape != (3, 3) or c.shape != (4, 4) or not np.isfinite(k).all()
            or not np.isfinite(c).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(c[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(c[:3, :3].T @ c[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(c[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('Invalid prepared camera')
    return k, c, np.linalg.inv(c)[:3]


def accept_model(blobs):
    """G0: schema, exact registered views, poses, raw observations and bidirectional tracks."""
    cameras, manifest, preparation = blobs['cameras'], blobs['manifest'], blobs['preparation']
    if (manifest['scene'] != 'leaves' or manifest['split'] != 'train'
            or manifest['protocol'] != 'leaves-undistorted-v1' or manifest['resolution'] != [504, 378]
            or [v['name'] for v in cameras] != NAMES or preparation['train_names'] != NAMES
            or preparation['intrinsics_protocol'] != 'leaves-undistorted-v1'):
        raise ValueError('Frozen training manifest/protocol mismatch')
    if manifest['train_images'] != [{'name': c['name'], 'sha256': c['image_sha256']} for c in cameras]:
        raise ValueError('Image metadata provenance mismatch')
    raw_cameras = read_cameras(blobs['cameras_bin'])
    images, tracks = read_images(blobs['images_bin']), read_tracks(blobs['tracks_bin'])
    if len(tracks) != manifest['initial_point_count'] or len(tracks) != preparation['point_count']:
        raise ValueError('Original point count mismatch')
    ordered = {im['name']: (iid, im) for iid, im in images.items()}
    centers = np.array([ordered[name][1]['c2w'][:3, 3] for name in NAMES])
    center = centers.mean(axis=0)
    spread = np.linalg.norm(centers - center, axis=1).max()
    if spread <= EPS:
        raise ValueError('Degenerate camera normalization')
    scale = 1 / spread
    normalization = preparation['normalization']
    if (normalization['source'] != '3 training camera centers only'
            or not np.allclose(center, normalization['center'], rtol=0, atol=1e-8)
            or not np.isclose(scale, normalization['scale'], rtol=0, atol=1e-8)):
        raise ValueError('Training camera normalization mismatch')
    for cam in cameras:
        k, c, _ = camera_arrays(cam)
        _, im = ordered[cam['name']]
        if im['camera_id'] not in raw_cameras or (cam['width'], cam['height']) != (504, 378):
            raise ValueError('Missing camera or incorrect dimensions')
        pose = im['c2w'].copy()
        pose[:3, 3] = (pose[:3, 3] - center) * scale
        if not np.allclose(k, raw_cameras[im['camera_id']], rtol=0, atol=1e-8) or not np.allclose(c, pose, rtol=0, atol=1e-6):
            raise ValueError('Prepared/raw camera correspondence mismatch')
    links = set()
    observations = {}
    for pid, entries in tracks.items():
        obs = {}
        for iid, idx in entries:
            if iid not in images or idx >= len(images[iid]['xy']):
                raise ValueError('Invalid track reference')
            im = images[iid]
            if int(im['point_ids'][idx]) != pid or (iid, idx) in links:
                raise ValueError('Nonreciprocal/duplicate track reference')
            links.add((iid, idx))
            obs[im['name']] = im['xy'][idx].copy()
        observations[pid] = obs
    reverse = {(iid, idx) for iid, im in images.items()
               for idx, pid in enumerate(im['point_ids']) if pid >= 0}
    if reverse != links:
        raise ValueError('Orphan image point association')
    inventory = {'accepted': True, 'registered_images': NAMES, 'points': len(tracks),
                 'raw_feature_counts': {im['name']: len(im['xy']) for im in images.values()},
                 'track_length_counts': dict(Counter(str(len(t)) for t in tracks.values())),
                 'linked_observations': len(links), 'raw_measurement_source': 'images.bin points2D.xy',
                 'ignored_as_numeric_input': ['points3D XYZ', 'RGB', 'fitted reprojection error'],
                 'reference_limit': 'Fixed public cameras and SfM-selected tracks, not an unbiased full-pipeline holdout'}
    return cameras, observations, inventory


def triangulated_projection(source_xy, sources, target):
    rows = []
    for xy, cam in zip(np.asarray(source_xy).reshape(2, 2), sources):
        k, _, p = camera_arrays(cam)
        uv = np.linalg.solve(k, np.r_[xy, 1.])
        rows.extend([uv[0] * p[2] - p[0], uv[1] * p[2] - p[1]])
    a = np.asarray(rows)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite_source')
    _, s, vt = np.linalg.svd(a)
    if s[0] <= 0 or s[-2] <= EPS * s[0] or abs(vt[-1, 3]) <= EPS:
        raise ValueError('degenerate_dlt')
    xyz = vt[-1, :3] / vt[-1, 3]
    for cam in [*sources, target]:
        _, _, p = camera_arrays(cam)
        z = (p @ np.r_[xyz, 1.])[2]
        if not np.isfinite(z) or z <= EPS:
            raise ValueError('nonpositive_camera_z')
    k, _, p = camera_arrays(target)
    pixel = k @ (p @ np.r_[xyz, 1.])
    uv = pixel[:2] / pixel[2]
    if not np.isfinite(uv).all() or not np.isfinite(xyz).all():
        raise ValueError('nonfinite_projection')
    return uv, xyz


def source_score(source_xy, sources, target):
    """Does not accept the held observation, full SfM point, prior or any error label."""
    result = {'score': None, 'angle_control': None, 'projection': None, 'xyz': None,
              'jacobian': None, 'status': 'unknown', 'angle_status': 'unknown'}
    try:
        uv, xyz = triangulated_projection(source_xy, sources, target)
    except (ValueError, np.linalg.LinAlgError) as error:
        result['status'] = 'base_' + str(error)
        return result
    result.update(projection=uv.tolist(), xyz=xyz.tolist())
    rays = []
    for xy, cam in zip(np.asarray(source_xy).reshape(2, 2), sources):
        k, c, _ = camera_arrays(cam)
        ray = c[:3, :3] @ np.linalg.solve(k, np.r_[xy, 1.])
        rays.append(ray / np.linalg.norm(ray))
    sine = float(np.linalg.norm(np.cross(*rays)))
    if sine > 0 and np.isfinite(1 / sine):
        result.update(angle_control=1 / sine, angle_status='available')
    else:
        result['angle_status'] = 'zero_or_nonfinite_sine'
    jac = np.empty((2, 4))
    z = np.asarray(source_xy, dtype=np.float64).reshape(4)
    for d in range(4):
        delta = np.eye(4)[d] * H
        try:
            plus, _ = triangulated_projection(z + delta, sources, target)
            minus, _ = triangulated_projection(z - delta, sources, target)
            jac[:, d] = (plus - minus) / (2 * H)
        except (ValueError, np.linalg.LinAlgError) as error:
            result['status'] = 'probe_' + str(error)
            return result
    score = float(np.linalg.norm(jac))
    if not np.isfinite(score):
        result['status'] = 'nonfinite_sensitivity'
        return result
    result.update(score=score, jacobian=jac.tolist(), status='available')
    return result


def evaluate_tracks(cameras, observations, check_budget=lambda: None):
    by_name = {cam['name']: cam for cam in cameras}
    rows = []
    for pid, obs in sorted(observations.items()):
        check_budget()
        for target in NAMES:
            row = {'point_id': int(pid), 'fold': fold_for_id(pid), 'image': target,
                   'track_length': len(obs), 'score': None, 'angle_control': None,
                   'residual_px': None, 'projection': None, 'xyz': None, 'jacobian': None,
                   'target_xy': None, 'status': 'missing_three_view_track', 'angle_status': 'unknown'}
            if set(obs) == set(NAMES):
                sources = [name for name in NAMES if name != target]
                scored = source_score([obs[n] for n in sources], [by_name[n] for n in sources], by_name[target])
                row.update(scored)
                # The target observation enters ONLY after the complete score computation.
                row['target_xy'] = np.asarray(obs[target]).tolist()
                if scored['projection'] is not None:
                    r = float(np.linalg.norm(np.asarray(scored['projection']) - obs[target]))
                    if not np.isfinite(r):
                        raise ValueError('Nonfinite held-observation error')
                    row['residual_px'] = r
            rows.append(row)
    return rows


def contrast(rows, key):
    x = np.array([r[key] for r in rows], dtype=np.float64)
    y = np.array([r['residual_px'] for r in rows], dtype=np.float64)
    result = {'n': len(rows), 'low': None, 'high': None, 'high_minus_low_mean': None, 'reason': None}
    if not len(rows):
        result['reason'] = 'no_pairs'
        return result
    q1, q2 = np.quantile(x, [1/3, 2/3], method='linear')
    result.update(q1=float(q1), q2=float(q2))
    if q1 >= q2:
        result['reason'] = 'tied_tertiles'
        return result
    lo, hi = y[x <= q1], y[x >= q2]
    result.update(low=describe(lo), high=describe(hi))
    if min(len(lo), len(hi)) < 10:
        result['reason'] = 'fewer_than_ten_in_extreme_group'
    else:
        result['high_minus_low_mean'] = float(hi.mean() - lo.mean())
    return result


def score_summary(rows, key):
    paired = [r for r in rows if r[key] is not None and r['residual_px'] is not None]
    return {'n': len(paired), 'correlation': spearman([r[key] for r in paired], [r['residual_px'] for r in paired]),
            'contrast': contrast(paired, key),
            'folds': {str(f): spearman([r[key] for r in paired if r['fold'] == f],
                                      [r['residual_px'] for r in paired if r['fold'] == f]) for f in range(5)}}


def summarize(rows, track_count):
    if len(rows) != track_count * 3 or len({(r['point_id'], r['image']) for r in rows}) != len(rows):
        raise ValueError('Incomplete/duplicate track records')
    result = {'per_image': {}, 'all_observability_passed': True, 'all_direction_passed': True,
              'statistical_significance_claimed': False, 'weights_authorized': False}
    for name in NAMES:
        own = [r for r in rows if r['image'] == name]
        full = [r for r in own if r['track_length'] == 3]
        s = score_summary(full, 'score')
        a = score_summary(full, 'angle_control')
        common = [r for r in full if r['score'] is not None and r['angle_control'] is not None and r['residual_px'] is not None]
        s_common, a_common = score_summary(common, 'score'), score_summary(common, 'angle_control')
        rho, ac = s_common['correlation']['rho'], a_common['correlation']['rho']
        paired = [r for r in full if r['score'] is not None and r['residual_px'] is not None]
        excluded = [r for r in own if r not in paired]  # plain JSON-compatible dictionaries
        valid = (s['n'] >= 60 and bool(full) and s['n'] / len(full) >= .5
                 and all(s['folds'][str(f)]['n'] >= 10 for f in range(5))
                 and s['correlation']['rho'] is not None and s['contrast']['reason'] is None)
        positive = sum(f['rho'] is not None and f['rho'] > 0 for f in s['folds'].values())
        direction = (s['correlation']['rho'] is not None and s['correlation']['rho'] >= .2
                     and positive >= 4 and s['contrast']['high_minus_low_mean'] is not None
                     and s['contrast']['high_minus_low_mean'] > 0)
        result['per_image'][name] = {
            'all_tracks': len(own), 'three_view_tracks': len(full), 'score': s, 'angle_control': a,
            'same_subset': {'n': len(common), 'score': s_common, 'angle_control': a_common,
                            'rho_score_minus_angle': None if rho is None or ac is None else rho - ac},
            'status_counts': dict(Counter(r['status'] for r in own)),
            'angle_status_counts': dict(Counter(r['angle_status'] for r in full)),
            'included_residuals': describe([r['residual_px'] for r in paired]),
            'excluded_residuals': describe([r['residual_px'] for r in excluded if r['residual_px'] is not None]),
            'score_missing_but_reference_available': sum(r['score'] is None and r['residual_px'] is not None for r in own),
            'positive_folds': positive, 'observability_passed': bool(valid), 'direction_passed': bool(direction)}
        result['all_observability_passed'] &= bool(valid)
        result['all_direction_passed'] &= bool(direction)
    result['decision'] = ('eligible_for_separate_cross_scene_review_only' if result['all_observability_passed'] and result['all_direction_passed']
                          else 'insufficient_support_stop' if not result['all_observability_passed'] else 'direction_failed_stop')
    return result
