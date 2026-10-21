"""CPU ordinal diagnostic on arrays only; no dataset/model loader or GT fitting.

Frozen terrace-relative-plan-v1 sampling, cell-balanced pair losses and controls.
Zero raw predictions are valid ties, unlike aligned inverse-Z masks.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

PLAN_SHA = 'e1a3da0170e9beb0394136f81ab8bcf0b934e4d4dd6a285d24f1e0bcd7669b50'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def sample_raw(q, xy):
    q, xy = np.asarray(q, dtype=np.float64), np.asarray(xy, dtype=np.float64)
    if q.ndim != 2 or min(q.shape) < 2 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Expected work image and N by 2 center coordinates')
    inside = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 1] >= 0)
              & (xy[:, 0] < q.shape[1]-1) & (xy[:, 1] < q.shape[0]-1))
    safe = np.where(inside[:, None], xy, 0.)
    x, y = np.floor(safe).astype(np.int64).T
    values = np.stack([q[y, x], q[y, x+1], q[y+1, x], q[y+1, x+1]], axis=1)
    valid = inside & np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    dx, dy = safe.T - np.array([x, y])
    weights = np.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], axis=1)
    result = np.full(len(xy), np.nan)
    result[valid] = np.sum(values[valid]*weights[valid], axis=1)
    return result, valid


def select_nodes(gt, image_index):
    """GT-only fixed-domain sampling, one native cell at a time."""
    gt = np.asarray(gt)
    if gt.ndim != 2 or min(gt.shape) < 36 or any(n % 6 for n in gt.shape) or image_index not in (0, 1, 2):
        raise ValueError('Expected image-sized GT and frozen image index')
    h, w = gt.shape
    ids, cell_ids, coverage = [], [], []
    counts = dict(positive_finite=0, positive_infinity=0, negative_infinity=0,
                  nan=0, zero=0, finite_negative=0, fixed_domain_valid=0)
    for c in range(108):
        cy, cx = divmod(c, 12)
        y0, y1 = cy*h//9, (cy+1)*h//9
        x0, x1 = cx*w//12, (cx+1)*w//12
        block = gt[y0:y1, x0:x1]
        finite = np.isfinite(block)
        positive = finite & (block > 0)
        counts['positive_finite'] += int(positive.sum())
        for name, mask in [('positive_infinity', np.isposinf(block)), ('negative_infinity', np.isneginf(block)),
                           ('nan', np.isnan(block)), ('zero', block == 0), ('finite_negative', finite & (block < 0))]:
            counts[name] += int(mask.sum())
        y_work = (np.arange(y0, y1)+.5)/6-.5
        x_work = (np.arange(x0, x1)+.5)/6-.5
        domain = ((y_work >= 2) & (y_work <= h/6-3))[:, None] & ((x_work >= 2) & (x_work <= w/6-3))[None, :]
        yy, xx = np.nonzero(positive & domain)
        positions = (yy+y0)*w + xx+x0
        count = len(positions)
        counts['fixed_domain_valid'] += count
        if count > 64:
            take = np.random.Generator(np.random.PCG64(42+108*image_index+c)).permutation(count)[:64]
            positions = np.sort(positions[take])
        ids.extend(positions.tolist())
        cell_ids.extend([c]*len(positions))
        coverage.append(dict(cell=c, native_valid=int(positive.sum()), domain_valid=count, sampled=len(positions)))
    ids, cell_ids = np.array(ids, dtype=np.int64), np.array(cell_ids, dtype=np.int64)
    order = np.argsort(ids, kind='stable')
    ids, cell_ids = ids[order], cell_ids[order]
    z = np.asarray(gt.flat[ids], dtype=np.float64)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError('Duplicate native node indices')
    counts['domain_excluded_valid'] = counts['positive_finite']-counts['fixed_domain_valid']
    return dict(indices=ids, cell=cell_ids, z=z, native_shape=(h, w), coverage=coverage, counts=counts)


def contrast_mask(z, pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return np.empty(0, dtype=bool)
    a, b = np.asarray(z)[pairs].T
    if not (np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError('Reference pairs require positive finite Z')
    return np.maximum(a, b)/np.minimum(a, b) >= 1.10


def make_pairs(nodes, image_index):
    if image_index not in (0, 1, 2):
        raise ValueError('Frozen image index required')
    by_cell = [np.flatnonzero(nodes['cell'] == c) for c in range(108)]
    groups = dict(local=[], cross=[])
    for c, source in enumerate(by_cell):
        local = np.asarray(list(itertools.combinations(source.tolist(), 2)), dtype=np.int64).reshape(-1, 2)
        pool_count = len(local)
        if len(local) > 256:
            rng = np.random.Generator(np.random.PCG64(100000+108*image_index+c))
            local = local[rng.permutation(len(local))[:256]]
        cy, cx = divmod(c, 12)
        target_cells = [t for t in range(108) if len(by_cell[t]) and max(abs(t//12-cy), abs(t%12-cx)) >= 3]
        drawn = []
        if len(source) and target_cells:
            rng = np.random.Generator(np.random.PCG64(200000+108*image_index+c))
            for _ in range(256):
                a = int(source[rng.integers(len(source))])
                target = target_cells[rng.integers(len(target_cells))]
                b = int(by_cell[target][rng.integers(len(by_cell[target]))])
                drawn.append(tuple(sorted((a, b))))
        cross = np.asarray(sorted(set(drawn)), dtype=np.int64).reshape(-1, 2)
        for kind, candidates, pool, draws in [('local', local, pool_count, len(local)),
                                              ('cross', cross, None, len(drawn))]:
            eligible = contrast_mask(nodes['z'], candidates)
            groups[kind].append(dict(cell=c, pool_count=pool, draws=draws,
                duplicate_draws=draws-len(candidates), candidates=candidates,
                eligible=eligible, pairs=candidates[eligible],
                included=int(eligible.sum()) >= 16))
    return groups


def relation_loss(q, z, pairs, valid=None):
    q, z = np.asarray(q, dtype=np.float64), np.asarray(z, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if q.shape != z.shape or q.ndim != 1:
        raise ValueError('Expected per-node q and reference Z')
    node_valid = np.isfinite(q) & (q >= 0)
    if valid is not None:
        node_valid &= np.asarray(valid, dtype=bool)
    if len(pairs) and ((pairs < 0).any() or (pairs >= len(q)).any()):
        raise ValueError('Pair index outside sampled nodes')
    a, b = pairs.T
    if not np.isfinite(z).all() or (z <= 0).any() or (z[a] == z[b]).any():
        raise ValueError('Pair loss requires positive finite unequal reference Z')
    good = node_valid[a] & node_valid[b]
    tie = good & (q[a] == q[b])
    correct = good & ~tie & ((q[a] > q[b]) == (z[a] < z[b]))
    loss = np.ones(len(pairs), dtype=np.float64)
    loss[tie], loss[correct] = .5, 0.
    return loss, ~good, tie


def score(nodes, groups, q, valid=None, per_cell=True):
    q = np.asarray(q, dtype=np.float64)
    ok = np.isfinite(q) & (q >= 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    result = dict(node_count=len(q), invalid_nodes=int((~ok).sum()),
                  invalid_node_fraction=float((~ok).mean()) if len(ok) else None, strata={})
    for kind, cells in groups.items():
        rows, main_means = [], []
        total_main = invalid_main = 0
        all_pairs = []
        for cell in cells:
            pairs = cell['pairs']
            loss, invalid, ties = relation_loss(q, nodes['z'], pairs, ok)
            mean = float(loss.mean()) if len(loss) else None
            if cell['included']:
                main_means.append(mean)
                total_main += len(pairs)
                invalid_main += int(invalid.sum())
            all_pairs.extend(map(tuple, pairs.tolist()))
            rows.append(dict(cell=cell['cell'], pool_count=cell['pool_count'], draws=cell['draws'],
                duplicate_draws=cell['duplicate_draws'], unique_candidates=len(cell['candidates']),
                contrast_excluded=int((~cell['eligible']).sum()), eligible_pairs=len(pairs),
                included=cell['included'], loss=mean, invalid_pairs=int(invalid.sum()), ties=int(ties.sum())))
        supported = len(main_means) >= 12 and total_main >= 256
        result['strata'][kind] = dict(loss=float(np.mean(main_means)) if main_means else None,
            included_cells=len(main_means), main_pairs=total_main, invalid_main_pairs=invalid_main,
            invalid_pair_fraction=invalid_main/total_main if total_main else None,
            canonical_unique_eligible_pairs=len(set(all_pairs)), repeated_eligible_pair_memberships=len(all_pairs)-len(set(all_pairs)),
            support_passed=supported)
        if per_cell:
            result['strata'][kind]['cells'] = rows
    means = [x['loss'] for x in result['strata'].values()]
    result['loss'] = float(np.mean(means)) if all(x is not None for x in means) else None
    result['support_passed'] = len(q) >= 1024 and all(x['support_passed'] for x in result['strata'].values())
    return result


def shuffled_nodes(q, valid, seed, image_index):
    """One node permutation, reused by every pair; invalid sites never move."""
    q = np.asarray(q, dtype=np.float64)
    good = np.asarray(valid, dtype=bool) & np.isfinite(q) & (q >= 0)
    out = q.copy()
    values = q[good]
    rng = np.random.Generator(np.random.PCG64(seed+1000*image_index))
    out[good] = values[rng.permutation(len(values))]
    return out


def average_ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    sorted_x, ranks = x[order], np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start+1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = (start+end-1)/2+1
        start = end
    return ranks


def spearman(q, z, valid):
    good = np.asarray(valid, dtype=bool) & np.isfinite(q) & (q >= 0)
    n = int(good.sum())
    if n < 2:
        return dict(value=None, n=n, status='unknown_insufficient')
    a, b = average_ranks(np.asarray(q)[good]), average_ranks(1/np.asarray(z)[good])
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return dict(value=float(np.dot(a, b)/denominator) if denominator else None,
                n=n, status='valid' if denominator else 'unknown_constant')


def pair_fingerprint(nodes, groups):
    h = hashlib.sha256()
    for value in [nodes['indices'], nodes['cell']]:
        h.update(np.asarray(value, dtype='<i8').tobytes())
    h.update(np.asarray(nodes['z'], dtype='<f8').tobytes())
    for kind in ('local', 'cross'):
        h.update(kind.encode())
        for cell in groups[kind]:
            h.update(np.asarray([cell['cell'], len(cell['candidates'])], dtype='<i8').tobytes())
            h.update(np.asarray(cell['candidates'], dtype='<i8').tobytes())
            h.update(cell['eligible'].astype(np.uint8).tobytes())
    return h.hexdigest()


def evaluate_arrays(gt, raw, image_index):
    """Pure synthetic-tested numerical evaluator; caller owns sealing and isolation."""
    gt, raw = np.asarray(gt), np.asarray(raw)
    if gt.ndim != 2 or raw.shape != (gt.shape[0]//6, gt.shape[1]//6):
        raise ValueError('Native/work grid contract mismatch')
    nodes = select_nodes(gt, image_index)
    groups = make_pairs(nodes, image_index)
    y, x = np.divmod(nodes['indices'], gt.shape[1])
    xy = np.column_stack([(x+.5)/6-.5, (y+.5)/6-.5])
    q, valid = sample_raw(raw, xy)
    model = score(nodes, groups, q, valid)
    controls = dict(constant=score(nodes, groups, np.ones(len(q)), per_cell=False),
                    oracle=score(nodes, groups, 1/nodes['z'], per_cell=False), shuffled=[])
    for seed in range(42, 62):
        shuffled = shuffled_nodes(q, valid, seed, image_index)
        controls['shuffled'].append(dict(seed=seed, **score(nodes, groups, shuffled, valid, per_cell=False)))
    losses = [c['loss'] for c in controls['shuffled']]
    controls['shuffle_mean'] = float(np.mean(losses)) if all(x is not None for x in losses) else None
    controls['shuffle_min'] = min(losses) if all(x is not None for x in losses) else None
    controls['shuffle_max'] = max(losses) if all(x is not None for x in losses) else None
    report = dict(image_index=image_index, coverage=nodes['coverage'], reference_counts=nodes['counts'],
                  pair_fingerprint=pair_fingerprint(nodes, groups), model=model, controls=controls,
                  secondary_spearman=spearman(q, nodes['z'], valid))
    return report, nodes, groups


def assess(reports):
    if len(reports) != 3 or [r['image_index'] for r in reports] != [0, 1, 2]:
        raise ValueError('Exactly three frozen image positions required')
    losses = [r['model']['loss'] for r in reports]
    macro = float(np.mean(losses)) if all(x is not None for x in losses) else None
    if not all(r['model']['support_passed'] for r in reports):
        status = 'blocked_relative_support'
    elif any(r['model']['invalid_node_fraction'] > .01 or
             any(s['invalid_pair_fraction'] > .02 for s in r['model']['strata'].values()) for r in reports):
        status = 'failed_prediction_validity'
    elif (macro > .35 or any(r['model']['loss'] > .4 or
            any(s['loss'] > .4 for s in r['model']['strata'].values()) or
            r['controls']['shuffle_mean']-r['model']['loss'] < .05 for r in reports)):
        status = 'failed_ordinal_gate'
    else:
        status = 'passed_ordinal_screen'
    return dict(status=status, scene_macro_loss=macro,
                interpretation='ordinal-only screen; synthetic results do not establish real performance')


def verify_prediction_seal(root, expected_names, source_hashes, input_hashes):
    """Verify all predictions before a caller is allowed to open any GT file."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Symlink prediction root')
    seal = json.loads((root/'seal.json').read_text())
    if (seal.get('status') != 'sealed_all_predictions' or seal.get('plan_sha256') != PLAN_SHA
            or seal.get('source_sha256') != source_hashes or seal.get('input_sha256') != input_hashes
            or [x['name'] for x in seal.get('predictions', [])] != list(expected_names)):
        raise ValueError('Prediction seal identity/status mismatch')
    names = [x['file'] for x in seal['predictions']]
    if (len(names) != 3 or len(set(names)) != 3 or any(Path(n).name != n or not n.endswith('.npy') for n in names)
            or {p.name for p in root.iterdir()} != set(names) | {'seal.json'}
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError('Prediction inventory mismatch')
    for row in seal['predictions']:
        if digest(root/row['file']) != row['sha256']:
            raise ValueError('Prediction content drift')
    return seal
