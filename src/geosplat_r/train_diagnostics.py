"""CPU-only grouped sparse-proxy diagnostics; never supplies training weights."""
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np

PLAN_SHA256 = 'ad66abed47aa5942531a5c4b9f202f002af919ab7550afad7041d531c946607a'
NAMES = ['image001.png', 'image012.png', 'image025.png']
EDGES = np.array([0., .01, .025, .05, .1, np.inf])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fold_for_id(point_id):
    text = 'train-error-coverage-v1|leaves|' + str(int(point_id))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % 5


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite summary values')
    return {'n': int(a.size), 'mean': float(a.mean()) if a.size else None,
            'median': float(np.median(a)) if a.size else None,
            'p90': float(np.percentile(a, 90)) if a.size else None}


def average_ranks(values):
    a = np.asarray(values, dtype=np.float64)
    order = np.argsort(a, kind='stable')
    ranks = np.empty(len(a), dtype=np.float64)
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and a[order[end]] == a[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError('Unpaired ranks')
    reason = None
    if len(x) < 5:
        reason = 'fewer_than_five_pairs'
    elif not (np.isfinite(x).all() and np.isfinite(y).all()):
        reason = 'nonfinite_pairs'
    else:
        a, b = average_ranks(x), average_ranks(y)
        if np.std(a) == 0 or np.std(b) == 0:
            reason = 'zero_rank_variance'
        else:
            return {'n': len(x), 'rho': float(np.corrcoef(a, b)[0, 1]), 'reason': None}
    return {'n': len(x), 'rho': None, 'reason': reason}


def fit_affine(samples, inverse_z):
    x, y = np.asarray(samples, dtype=np.float64), np.asarray(inverse_z, dtype=np.float64)
    if len(x) != len(y) or len(y) < 20:
        raise ValueError('fewer_than_twenty_fit_anchors')
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or (y <= 0).any():
        raise ValueError('invalid_fit_values')
    if np.std(x) < 1e-6:
        raise ValueError('constant_fit_samples')
    a, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], axis=1), y, rcond=None)[0]
    if not np.isfinite([a, b]).all() or a <= 0:
        raise ValueError('nonpositive_or_nonfinite_slope')
    return {'slope': float(a), 'shift': float(b), 'normalizer': float(np.median(y))}


def sample_raw(raw, xy):
    raw, xy = np.asarray(raw, dtype=np.float64), np.asarray(xy, dtype=np.float64)
    tile_indices(xy)
    if raw.shape != (378, 504) or not np.isfinite(raw).all():
        raise ValueError('invalid_raw_prediction')
    return cv2.remap(raw, xy[:, 0].astype(np.float32)[None],
                     xy[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]


def verify_full_fit(prior, anchors, metadata):
    params = fit_affine(sample_raw(prior['raw_inverse_depth'], anchors['xy']), anchors['inverse_z'])
    for key in ['slope', 'shift', 'normalizer']:
        if not np.isclose(params[key], metadata[key], atol=1e-10, rtol=1e-8):
            raise ValueError('Full-fit source reproduction mismatch: ' + key)


def nearest_distance(query, support, check=lambda: None):
    query, support = np.asarray(query, dtype=np.float64), np.asarray(support, dtype=np.float64)
    if len(support) == 0:
        raise ValueError('no_support_anchors')
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 4096):
        check()
        delta = query[start:start + 4096, None, :] - support[None, :, :]
        result[start:start + 4096] = np.sqrt(np.einsum('ijk,ijk->ij', delta, delta).min(axis=1)) / 630.
    return result


def tile_indices(xy):
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('invalid_anchor_coordinates')
    if ((xy < 0).any() or (xy[:, 0] >= 504).any() or (xy[:, 1] >= 378).any()):
        raise ValueError('anchor_outside_image')
    return (xy[:, 1] // 42).astype(int) * 12 + (xy[:, 0] // 42).astype(int)


def grouped_crossfit(samples, xy, inverse_z, point_ids, check=lambda: None):
    """No cached aligned targets, full-fit coefficients or masks enter prediction."""
    samples = np.asarray(samples, dtype=np.float64)
    xy, inverse_z, point_ids = np.asarray(xy), np.asarray(inverse_z), np.asarray(point_ids)
    if not (len(samples) == len(xy) == len(inverse_z) == len(point_ids)):
        raise ValueError('unpaired_anchors')
    if point_ids.dtype != np.int64 or len(np.unique(point_ids)) != len(point_ids):
        raise ValueError('point_ids_must_be_unique_int64_per_view')
    tiles = tile_indices(xy)
    assignments = np.array([fold_for_id(p) for p in point_ids])
    records = [{'anchor_index': i, 'point_id': int(p), 'fold': int(assignments[i]),
                'x': float(xy[i, 0]), 'y': float(xy[i, 1]), 'tile': int(tiles[i]),
                'prediction': None, 'normalizer': None, 'error': None,
                'full_fit_error': None, 'distance': None, 'fit_tile_count': None,
                'prediction_nonpositive': None, 'status': 'pending'} for i, p in enumerate(point_ids)]
    folds = []
    for fold in range(5):
        check()
        fit, held = assignments != fold, assignments == fold
        row = {'fold': fold, 'fit_count': int(fit.sum()), 'held_count': int(held.sum()),
               'status': 'unsupported', 'reason': None, 'parameters': None}
        ids = np.flatnonzero(held)
        if held.sum() < 5:
            row['reason'] = 'fewer_than_five_held_anchors'
        else:
            try:
                params = fit_affine(samples[fit], inverse_z[fit])
                prediction = params['slope'] * samples[held] + params['shift']
                error = np.abs(prediction - inverse_z[held]) / params['normalizer']
                if not (np.isfinite(prediction).all() and np.isfinite(error).all()):
                    raise ValueError('nonfinite_held_prediction_or_residual')
                distance = nearest_distance(xy[held], xy[fit], check)
                counts = np.bincount(tiles[fit], minlength=108)
                for j, index in enumerate(ids):
                    records[index].update(prediction=float(prediction[j]), normalizer=params['normalizer'],
                        error=float(error[j]), distance=float(distance[j]), fit_tile_count=int(counts[tiles[index]]),
                        prediction_nonpositive=bool(prediction[j] <= 0), status='passed')
                row.update(status='passed', parameters=params, error=describe(error),
                           nonpositive_predictions=int((prediction <= 0).sum()),
                           correlation=spearman(distance, error))
            except ValueError as exc:
                row.update(status='failed', reason=str(exc))
        if row['status'] != 'passed':
            for index in ids:
                records[index]['status'] = row['status']
                records[index]['reason'] = row['reason']
        folds.append(row)
    return folds, records


def distance_summary(values):
    a = np.asarray(values)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('nonfinite_support_distance')
    counts = np.histogram(a, bins=EDGES)[0]
    return {**describe(a), 'bin_counts': counts.tolist(),
            'bin_fractions': (counts / len(a)).tolist() if len(a) else [None] * 5}


def analyze_arrays(prior, anchors, check=lambda: None):
    raw = np.asarray(prior['raw_inverse_depth'], dtype=np.float64)
    valid = np.asarray(prior['valid'])
    if raw.shape != (378, 504) or valid.shape != raw.shape or valid.dtype != np.bool_:
        raise ValueError('invalid_prior_shape_or_mask')
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite_raw_prediction')
    xy = np.asarray(anchors['xy'], dtype=np.float64)
    y = np.asarray(anchors['inverse_z'], dtype=np.float64)
    tile_ids = tile_indices(xy)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('invalid_anchor_inverse_z')
    samples = sample_raw(raw, xy)
    full = fit_affine(samples, y)
    folds, records = grouped_crossfit(samples, xy, y, anchors['point_ids'], check)
    full_prediction = full['slope'] * samples + full['shift']
    for row in records:
        if row['status'] == 'passed':
            row['full_fit_error'] = float(abs(full_prediction[row['anchor_index']] - y[row['anchor_index']]) / row['normalizer'])
    for fold in folds:
        rows = [x for x in records if x['fold'] == fold['fold'] and x['status'] == 'passed']
        if rows:
            fold['full_fit_on_held'] = describe([x['full_fit_error'] for x in rows])
            fold['delta_error'] = describe([x['error'] - x['full_fit_error'] for x in rows])
    yy, xx = np.mgrid[:378, :504]
    distance = nearest_distance(np.column_stack([xx.ravel(), yy.ravel()]), xy, check).reshape(378, 504)
    counts = np.bincount(tile_ids, minlength=108)
    tiles = []
    for index in range(108):
        row, col = divmod(index, 12)
        area = np.s_[row * 42:(row + 1) * 42, col * 42:(col + 1) * 42]
        mask, dist = valid[area], distance[area]
        residuals = [x['error'] for x in records if x['tile'] == index and x['status'] == 'passed']
        tiles.append({'tile': index, 'row': row, 'column': col, 'anchor_count': int(counts[index]),
            'pixel_count': 1764, 'prior_valid_pixels': int(mask.sum()),
            'full_distance': describe(dist.ravel()), 'prior_valid_distance': describe(dist[mask]),
            'oof_count': len(residuals), 'oof_error': describe(residuals) if len(residuals) >= 5 else None,
            'error_status': 'observed' if len(residuals) >= 5 else 'insufficient_observations'})
    observed = [x for x in records if x['status'] == 'passed']
    summary = {'cv_complete': all(x['status'] == 'passed' for x in folds),
        'anchors': len(xy), 'oof_observations': len(observed), 'folds': folds,
        'full_fit': full, 'full_fit_in_sample_error': describe(abs(full_prediction - y) / full['normalizer']),
        'oof_error': describe([x['error'] for x in observed]),
        'full_fit_same_held_normalization': describe([x['full_fit_error'] for x in observed]),
        'paired_delta_error': describe([x['error'] - x['full_fit_error'] for x in observed]),
        'oof_correlation': spearman([x['distance'] for x in observed], [x['error'] for x in observed]),
        'nonpositive_oof_predictions': sum(x['prediction_nonpositive'] for x in observed),
        'full_support_distance': distance_summary(distance.ravel()),
        'prior_valid_support_distance': distance_summary(distance[valid]),
        'prior_valid_pixels': int(valid.sum()), 'pixel_count': int(valid.size),
        'zero_anchor_tiles': int((counts == 0).sum()),
        'tiles_with_observed_error': sum(x['oof_error'] is not None for x in tiles)}
    return summary, records, tiles, distance


class AllowlistedInputs:
    """Exact path/hash gate, not an untrusted-code security sandbox."""
    def __init__(self, roots, plan, manifest):
        self.roots = {k: Path(v).resolve(strict=True) for k, v in roots.items()}
        self.accesses, self.denials = [], []
        for key, path in self.roots.items():
            if path.name != key or any('test' == v.lower() or 'evaluation' in v.lower() for v in path.parts):
                raise ValueError('Explicit train/anchors/priors leaf roots required')
        if set(self.roots) != {'train', 'anchors', 'priors'}:
            raise ValueError('Unexpected input roots')
        if set(manifest) != {'protocol', 'plan_sha256', 'files'} or manifest['protocol'] != plan['protocol'] or manifest['plan_sha256'] != PLAN_SHA256:
            raise ValueError('Independent allowlist differs from frozen plan')
        expected = {(x['root'], x['path']): x for x in plan['allowlisted_files']}
        if len(manifest['files']) != len(expected) or len(expected) != 14:
            raise ValueError('Exactly fourteen approved asset files required')
        self.files = {}
        for row in manifest['files']:
            if set(row) != {'root', 'path', 'absolute_path', 'bytes', 'sha256'}:
                raise ValueError('Unexpected allowlist fields')
            key = (row['root'], row['path'])
            if key in self.files or key not in expected or Path(row['path']).is_absolute() or '..' in Path(row['path']).parts:
                raise ValueError('Unexpected or traversing input path')
            if {k: row[k] for k in expected[key]} != expected[key]:
                raise ValueError('Allowlist hash/size metadata differs')
            candidate = self.roots[row['root']] / row['path']
            resolved = candidate.resolve(strict=True)
            if resolved != candidate or Path(row['absolute_path']) != resolved or not resolved.is_file():
                raise ValueError('Symlink escape or noncanonical asset path')
            self.files[key] = row
        self.cache = {}

    def read(self, root, path):
        key = (root, path)
        if key not in self.files:
            self.denials.append({'root': root, 'path': path})
            raise ValueError('Asset outside approved train-only allowlist')
        if key not in self.cache:
            row = self.files[key]
            data = Path(row['absolute_path']).read_bytes()
            if len(data) != row['bytes'] or sha(data) != row['sha256']:
                raise ValueError('Training asset content/hash drift')
            self.cache[key] = data
            self.accesses.append({k: row[k] for k in ['root', 'path', 'bytes', 'sha256']})
        return self.cache[key]

    def json(self, root, path):
        return json.loads(self.read(root, path))

    def arrays(self, root, path):
        with np.load(io.BytesIO(self.read(root, path)), allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}
