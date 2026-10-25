"""Frozen CPU cross-view prior checks; produces no reliability weights."""
from collections import Counter

import cv2
import numpy as np

from geosplat_r.train_diagnostics import (describe, fold_for_id, grouped_crossfit,
                                         sample_raw, spearman)

PLAN_SHA256 = '89f58dbe44710d38e6a438ebfe26eff6cb0c622d527ae0556686a2e959867d52'


def validate_camera(camera):
    k = np.asarray(camera['K'], dtype=np.float64)
    c = np.asarray(camera['c2w'], dtype=np.float64)
    if (camera['width'], camera['height']) != (504, 378):
        raise ValueError('Frozen camera dimensions changed')
    if k.shape != (3, 3) or c.shape != (4, 4) or not (np.isfinite(k).all() and np.isfinite(c).all()):
        raise ValueError('Invalid camera matrices')
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1])
            or not np.array_equal(c[3], [0, 0, 0, 1])):
        raise ValueError('Pinhole homogeneous camera required')
    if not np.allclose(c[:3, :3].T @ c[:3, :3], np.eye(3), atol=1e-6, rtol=0):
        raise ValueError('Nonrigid camera rotation')
    if not np.isclose(np.linalg.det(c[:3, :3]), 1, atol=1e-6, rtol=0):
        raise ValueError('Camera rotation orientation changed')
    if not np.isfinite(np.linalg.inv(k)).all():
        raise ValueError('Noninvertible intrinsics')


def project_pair(xy, target_q, target_camera, source_camera, source_raw,
                 source_parameters, target_status='passed'):
    """Feature kernel has no held inverse-Z, 3D point, mask or source-track input."""
    row = dict(status='unavailable', reason=None, source_u=None, source_v=None,
               source_z=None, q_projected=None, q_source=None, sign=None, score=None)

    def unavailable(reason):
        row['reason'] = reason
        return row

    if target_status != 'passed':
        return unavailable('target_fit_unavailable')
    if target_q is None or not np.isfinite(target_q):
        return unavailable('nonfinite_target_prediction')
    if target_q <= 0:
        return unavailable('nonpositive_target_prediction')
    if source_parameters is None:
        return unavailable('source_fit_unavailable')
    ki = np.asarray(target_camera['K'], dtype=np.float64)
    ci = np.asarray(target_camera['c2w'], dtype=np.float64)
    kj = np.asarray(source_camera['K'], dtype=np.float64)
    cj = np.asarray(source_camera['c2w'], dtype=np.float64)
    # Axial Z, not unit-length ray depth. All arithmetic remains float64.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        local = np.linalg.solve(ki, np.r_[xy, 1.]) / target_q
        world = ci[:3, :3] @ local + ci[:3, 3]
        source = cj[:3, :3].T @ (world - cj[:3, 3])
    if not np.isfinite(source).all():
        return unavailable('nonfinite_source_coordinates')
    row['source_z'] = float(source[2])
    if source[2] <= 0:
        return unavailable('behind_source_camera')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        uv = (kj @ source)[:2] / source[2]
    if not np.isfinite(uv).all():
        return unavailable('nonfinite_source_projection')
    row.update(source_u=float(uv[0]), source_v=float(uv[1]))
    if not (2 <= uv[0] < 501 and 2 <= uv[1] < 375):
        return unavailable('outside_source_border')
    q = cv2.remap(np.asarray(source_raw, dtype=np.float64),
                  np.array([[uv[0]]], np.float32), np.array([[uv[1]]], np.float32),
                  cv2.INTER_LINEAR)[0, 0]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        q = source_parameters['slope'] * q + source_parameters['shift']
    if not np.isfinite(q):
        return unavailable('nonfinite_source_prediction')
    row['q_source'] = float(q)
    if q <= 0:
        return unavailable('nonpositive_source_prediction')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        projected = 1. / source[2]
    if not np.isfinite(projected) or projected <= 0:
        return unavailable('nonfinite_disagreement')
    row['q_projected'] = float(projected)
    # Algebraically identical symmetric relative discrepancy; avoid overflow.
    scale = max(projected, q)
    a, b = projected / scale, q / scale
    value = 2 * abs(a - b) / (a + b)
    if not np.isfinite(value):
        return unavailable('nonfinite_disagreement')
    row.update(status='available', score=float(value), sign=int(np.sign(q - projected)))
    return row


def fit_views(assets, check=lambda: None):
    """Recompute exactly D1 folds from raw fields; never load saved fold fits."""
    fits, by_view = [], {}
    for asset in assets:
        check()
        anchors = asset['anchors']
        y = np.asarray(anchors['inverse_z'])
        if not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError('Invalid anchor inverse Z')
        samples = sample_raw(asset['prior']['raw_inverse_depth'], anchors['xy'])
        folds, records = grouped_crossfit(samples, anchors['xy'], y, anchors['point_ids'], check)
        by_view[asset['name']] = (folds, records)
        for fold in folds:
            fits.append({'image': asset['name'], **{k: v for k, v in fold.items()
                if k in ['fold', 'fit_count', 'held_count', 'status', 'reason',
                         'parameters', 'error', 'nonpositive_predictions']}})
    return fits, by_view


def evaluate_views(assets, check=lambda: None):
    for asset in assets:
        validate_camera(asset['camera'])
    fits, by_view = fit_views(assets, check)
    targets, directions = [], []
    for asset in assets:
        for original in by_view[asset['name']][1]:
            check()
            fold = original['fold']
            row = {'image': asset['name'], **{k: original[k] for k in
                ['anchor_index', 'point_id', 'fold', 'x', 'y', 'prediction', 'normalizer']},
                'reference_error': original['error'], 'target_status': original['status'],
                'target_reason': original.get('reason'),
                'prediction_nonpositive': original['prediction_nonpositive']}
            pairs = []
            for other in assets:
                if other['name'] == asset['name']:
                    continue
                source_fit = by_view[other['name']][0][fold]
                pair = project_pair([original['x'], original['y']], original['prediction'],
                    asset['camera'], other['camera'], other['prior']['raw_inverse_depth'],
                    source_fit['parameters'] if source_fit['status'] == 'passed' else None,
                    original['status'])
                pairs.append(pair)
                directions.append({'image': asset['name'], 'source_image': other['name'],
                    'anchor_index': original['anchor_index'], 'point_id': original['point_id'],
                    'fold': fold, **pair})
            available = [p['score'] for p in pairs if p['status'] == 'available']
            row.update(source_count=len(available),
                status=['no_source', 'one_source_only', 'two_sources'][len(available)],
                score=float(np.mean(available)) if len(available) == 2 else None)
            targets.append(row)
    return fits, targets, directions


def error_summary(rows):
    return describe([r['reference_error'] for r in rows if r['reference_error'] is not None])


def tertile_contrast(rows):
    if not rows:
        return {'status': 'unavailable', 'reason': 'no_primary_pairs', 'q1': None, 'q2': None,
                'low_error': describe([]), 'high_error': describe([]), 'high_minus_low': None}
    scores = np.array([r['score'] for r in rows], dtype=np.float64)
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3], method='linear')
    low = [r for r in rows if r['score'] <= q1]
    high = [r for r in rows if r['score'] >= q2]
    reason = 'tied_tertile_boundaries' if q1 >= q2 else (
        'fewer_than_five_in_extreme_group' if min(len(low), len(high)) < 5 else None)
    lo, hi = error_summary(low), error_summary(high)
    return {'status': 'available' if reason is None else 'unavailable', 'reason': reason,
            'q1': float(q1), 'q2': float(q2), 'low_error': lo, 'high_error': hi,
            'high_minus_low': hi['mean'] - lo['mean'] if reason is None else None}


def summarize_rows(rows, with_tertiles=True):
    selected = [r for r in rows if r['score'] is not None and r['reference_error'] is not None]
    excluded = [r for r in rows if r['score'] is None or r['reference_error'] is None]
    return {'anchors': len(rows), 'reference_error': error_summary(rows),
        'primary_count': len(selected), 'primary_fraction': len(selected) / len(rows) if rows else None,
        'source_counts': {str(i): sum(r['source_count'] == i for r in rows) for i in range(3)},
        'correlation': spearman([r['score'] for r in selected], [r['reference_error'] for r in selected]),
        'included_reference_error': error_summary(selected),
        'excluded_reference_error': error_summary(excluded),
        'by_status': {status: {'count': sum(r['status'] == status for r in rows),
            'reference_error': error_summary([r for r in rows if r['status'] == status])}
            for status in ['no_source', 'one_source_only', 'two_sources']},
        'tertiles': tertile_contrast(selected) if with_tertiles else None}


def summarize(assets, fits, targets, directions):
    images = []
    for asset in assets:
        name = asset['name']
        rows = [r for r in targets if r['image'] == name]
        image = {'image': name, **summarize_rows(rows)}
        image['folds'] = [{'fold': k, **summarize_rows([r for r in rows if r['fold'] == k], False)}
                          for k in range(5)]
        pairs = [r for r in directions if r['image'] == name]
        image['direction_failures'] = dict(Counter(r['reason'] for r in pairs if r['reason']))
        valid_folds = [f for f in image['folds'] if f['correlation']['rho'] is not None]
        image['positive_fold_correlations'] = sum(f['correlation']['rho'] > 0 for f in valid_folds)
        image['screening_support_pass'] = (image['primary_fraction'] >= .5 and len(valid_folds) == 5)
        rho, delta = image['correlation']['rho'], image['tertiles']['high_minus_low']
        image['screening_direction_pass'] = (rho is not None and rho > 0
            and image['positive_fold_correlations'] >= 3 and delta is not None and delta > 0)
        images.append(image)
    ordered = []
    for asset in assets:
        for other in assets:
            if asset['name'] == other['name']:
                continue
            pairs = [r for r in directions if r['image'] == asset['name'] and r['source_image'] == other['name']]
            valid = [r for r in pairs if r['status'] == 'available']
            ordered.append({'image': asset['name'], 'source_image': other['name'], 'total': len(pairs),
                'available': len(valid), 'disagreement': describe([r['score'] for r in valid]),
                'failures': dict(Counter(r['reason'] for r in pairs if r['reason'])),
                'sign_counts': {str(s): sum(r['sign'] == s for r in valid) for s in [-1, 0, 1]}})
    return {'images': images, 'ordered_pairs': ordered, 'target_rows': len(targets),
        'direction_rows': len(directions), 'fit_records': len(fits),
        'unique_observed_points': len({r['point_id'] for r in targets}),
        'all_cv_complete': all(f['status'] == 'passed' for f in fits),
        'screening_support_pass': all(i['screening_support_pass'] for i in images),
        'screening_direction_pass': all(i['screening_direction_pass'] for i in images),
        'screening_pass': all(i['screening_support_pass'] and i['screening_direction_pass'] for i in images),
        'limits': 'Selected sparse training SfM proxy; correlated predictions/points/cameras, shared prediction term, occlusion and common-mode errors. No dense truth, significance, confidence labels or reliability weights.'}
