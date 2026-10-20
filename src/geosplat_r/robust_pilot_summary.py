"""Frozen four-arm development pilot; retain every pair and per-view regression."""
import math

ARMS = ('A1_l1', 'B_weak_l1', 'C_cauchy', 'A2_l1_repeat')
METRICS = ('psnr', 'ssim', 'lpips_alex')
VIEWS = {'train': ['image001.png', 'image012.png', 'image025.png'],
         'test': ['image000.png', 'image008.png', 'image016.png', 'image024.png']}
PAIRS = [('C_cauchy', 'A1_l1'), ('C_cauchy', 'B_weak_l1'),
         ('C_cauchy', 'A2_l1_repeat'), ('B_weak_l1', 'A1_l1'),
         ('A2_l1_repeat', 'A1_l1'), ('A2_l1_repeat', 'B_weak_l1')]


def summarize(records):
    """Only summarize four complete matched evaluator outputs; no metric selection."""
    if list(records) != list(ARMS):
        raise ValueError('Exactly A1, B, C, A2 in frozen order required')
    protocol = records[ARMS[0]]['metric_protocol']
    for arm, record in records.items():
        if record['optimizer_updates'] != 30000 or record['metric_protocol'] != protocol:
            raise ValueError('Incomplete or unmatched metric protocol: ' + arm)
        for split, names in VIEWS.items():
            rows = record[split]['per_image']
            if [r['image'] for r in rows] != names:
                raise ValueError('Missing, duplicated or changed views: ' + arm + '/' + split)
            for metric in METRICS:
                values = [r[metric] for r in rows]
                declared = record[split]['mean'][metric]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values + [declared]):
                    raise ValueError('Nonfinite metric: ' + arm + '/' + metric)
                if abs(sum(values) / len(values) - declared) > 1e-9:
                    raise ValueError('Mean disagrees with per-view metrics: ' + arm + '/' + metric)
    contrasts = {}
    for left, right in PAIRS:
        key = left + '_minus_' + right
        result = {}
        for split, names in VIEWS.items():
            rows = []
            for index, name in enumerate(names):
                deltas = {m: records[left][split]['per_image'][index][m]
                             - records[right][split]['per_image'][index][m] for m in METRICS}
                worse = [m for m in METRICS if deltas[m] * (-1 if m == 'lpips_alex' else 1) < 0]
                rows.append({'image': name, 'delta': deltas, 'regressed_metrics': worse})
            result[split] = {'mean_delta': {m: records[left][split]['mean'][m]
                - records[right][split]['mean'][m] for m in METRICS}, 'per_image': rows}
        contrasts[key] = result
    gate = {}
    for metric in METRICS:
        direction = -1 if metric == 'lpips_alex' else 1
        scores = {arm: direction * records[arm]['test']['mean'][metric] for arm in ARMS}
        observed_range = abs(scores['A1_l1'] - scores['A2_l1_repeat'])
        best_control = max(scores[a] for a in ['A1_l1', 'B_weak_l1', 'A2_l1_repeat'])
        gain = scores['C_cauchy'] - best_control
        gate[metric] = {'observed_bookend_range': observed_range,
                        'C_gain_over_best_control_higher_is_better': gain,
                        'margin_over_range': gain - observed_range,
                        'strictly_passed': gain > observed_range}
    return {'protocol': 'leaves-robust-depth-pilot-summary-v1',
            'arms': {arm: {split: record[split]['mean'] for split in VIEWS}
                     for arm, record in records.items()},
            'all_pairwise_contrasts': contrasts, 'prospective_test_direction_gate': gate,
            'all_three_metrics_passed': all(v['strictly_passed'] for v in gate.values()),
            'limitations': ['Known development leaves, seed44 only; no final-test evidence.',
                'Two bookends give an observed range, not confidence intervals or significance.',
                'Half-L1 is one scalar control, not exhaustive average-strength matching.',
                'All per-view regressions retained; no checkpoint, crop or view selection.']}
