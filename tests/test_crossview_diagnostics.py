"""Synthetic geometry, leakage, missingness and exact-boundary checks for D2."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import diagnose_crossview_priors as entry
from geosplat_r.crossview_diagnostics import (PLAN_SHA256, evaluate_views, project_pair,
    summarize, summarize_rows, tertile_contrast, validate_camera)
from geosplat_r.train_diagnostics import NAMES, fold_for_id, sha


def camera():
    return dict(width=504, height=378, K=[[100., 0, 250.], [0, 100., 180.], [0, 0, 1.]],
                c2w=np.eye(4).tolist())


def synthetic_assets():
    yy, xx = np.mgrid[:378, :504]
    raw = .1 + xx * .0001 + yy * .00002
    xy = np.array([[60 + 3 * i, 40 + 10 * (i % 13)] for i in range(100)], dtype=np.float64)
    y = 2 * raw[xy[:, 1].astype(int), xy[:, 0].astype(int)] + .01
    return [dict(name=name, camera=camera(),
        prior=dict(raw_inverse_depth=raw.copy(), inverse_z=raw * 2 + .01, valid=np.ones(raw.shape, bool)),
        anchors=dict(xy=xy.copy(), inverse_z=y.copy(), point_ids=np.arange(1000, 1100, dtype=np.int64)))
        for name in NAMES]


class CrossviewGeometry(unittest.TestCase):
    def pair(self, xy=(300., 210.), q=.1, source=None, raw=None, params=None, **kwargs):
        return project_pair(xy, q, camera(), source or camera(),
            np.full((378, 504), .1) if raw is None else raw,
            dict(slope=1., shift=0.) if params is None else params, **kwargs)

    def test_translated_pinhole_uses_axial_z_not_ray_length(self):
        c = camera();c['c2w'][0][3] = 1.
        result = self.pair(source=c)
        self.assertEqual('available', result['status'])
        self.assertAlmostEqual(290., result['source_u'])
        self.assertAlmostEqual(210., result['source_v'])
        self.assertAlmostEqual(10., result['source_z'])
        self.assertAlmostEqual(0., result['score'])

    def test_rotated_source_uses_transposed_c2w_rotation(self):
        c = camera();c['c2w'] = [[0., 0, 1, 0], [0, 1., 0, 0], [-1., 0, 0, 0], [0, 0, 0, 1.]]
        validate_camera(c)
        r = self.pair(source=c, raw=np.full((378, 504), .2))
        self.assertAlmostEqual(50., r['source_u'])
        self.assertAlmostEqual(240., r['source_v'])
        self.assertAlmostEqual(5., r['source_z'])
        self.assertAlmostEqual(0., r['score'])

    def test_inter_linear_coordinate_table_and_independent_source_field(self):
        yy, xx = np.mgrid[:378, :504]
        c = camera();c['c2w'][0][3] = 1.
        r = self.pair(xy=[300.3, 210.], source=c, raw=.1 + .001 * xx)
        # OpenCV INTER_LINEAR's 1/32 interpolation table rounds .3 to .3125.
        self.assertAlmostEqual(.3903125, r['q_source'])
        self.assertAlmostEqual(2 * (.3903125 - .1) / (.3903125 + .1), r['score'])
        self.assertEqual(1, r['sign'])
        self.assertEqual('available', r['status'])  # A nearer source is not discarded.

    def test_coherent_wrong_local_plane_is_not_a_reliability_label(self):
        for shift in [-1., 1.]:
            c = camera();c['c2w'][0][3] = shift
            r = self.pair(q=.05, source=c, raw=np.full((378, 504), .05))
            self.assertAlmostEqual(0., r['score'])
            self.assertAlmostEqual(.5, abs(.05 - .1) / .1)
            self.assertNotIn('reliable', r)
            altered = self.pair(q=.05, source=c, raw=np.full((378, 504), .1))
            self.assertAlmostEqual(2 / 3, altered['score'])

    def test_nonpositive_nonfinite_and_missing_fit_order(self):
        self.assertEqual('nonpositive_target_prediction', self.pair(q=-1.)['reason'])
        self.assertEqual('nonfinite_target_prediction', self.pair(q=np.nan)['reason'])
        self.assertEqual('target_fit_unavailable', self.pair(q=-1., target_status='failed')['reason'])
        r = project_pair([300., 210.], .1, camera(), camera(), np.ones((378, 504)), None)
        self.assertEqual('source_fit_unavailable', r['reason'])
        for v, reason in [(-.1, 'nonpositive_source_prediction'), (np.nan, 'nonfinite_source_prediction')]:
            r = self.pair(raw=np.full((378, 504), v))
            self.assertIsNone(r['score']);self.assertEqual(reason, r['reason'])

    def test_camera_behind_and_strict_sampling_border(self):
        c = camera();c['c2w'][2][3] = 20.
        self.assertEqual('behind_source_camera', self.pair(source=c)['reason'])
        self.assertEqual('outside_source_border', self.pair(xy=[501., 200.])['reason'])
        self.assertEqual('outside_source_border', self.pair(xy=[250., 375.])['reason'])
        self.assertEqual('available', self.pair(xy=[2., 2.])['status'])

    def test_nonrigid_wrong_size_and_reflected_cameras_rejected(self):
        for field in ['size', 'nonrigid', 'reflection', 'nonfinite']:
            c = camera()
            if field == 'size':c['width'] = 503
            if field == 'nonrigid':c['c2w'][0][0] = 2.
            if field == 'reflection':c['c2w'][0][0] = -1.
            if field == 'nonfinite':c['K'][0][0] = np.nan
            with self.assertRaises(ValueError):validate_camera(c)


class CrossviewIsolationAndSummary(unittest.TestCase):
    def test_held_y_in_every_view_cannot_change_same_fold_features_or_normalizers(self):
        assets = synthetic_assets()
        fits, targets, pairs = evaluate_views(assets)
        changed = copy.deepcopy(assets)
        for asset in changed:
            mask = np.array([fold_for_id(p) == 0 for p in asset['anchors']['point_ids']])
            asset['anchors']['inverse_z'][mask] *= 8
        f2, t2, p2 = evaluate_views(changed)
        self.assertEqual([f for f in fits if f['fold'] == 0],
                         [{**f, 'error': old['error']} for f, old in zip(
                             [f for f in f2 if f['fold'] == 0], [f for f in fits if f['fold'] == 0])])
        self.assertEqual([p for p in pairs if p['fold'] == 0], [p for p in p2 if p['fold'] == 0])
        for first, second in zip(targets, t2):
            if first['fold'] == 0:
                self.assertEqual(first['prediction'], second['prediction'])
                self.assertEqual(first['normalizer'], second['normalizer'])
                self.assertEqual(first['score'], second['score'])
                self.assertNotEqual(first['reference_error'], second['reference_error'])

    def test_full_targets_masks_and_unused_3d_coordinates_do_not_enter_features(self):
        assets = synthetic_assets()
        first = evaluate_views(assets)
        for asset in assets:
            asset['prior']['inverse_z'][:] = 99999
            asset['prior']['valid'][:] = False
            asset['anchors']['points3D'] = 'forbidden-unused-3d-coordinate-sentinel'
        self.assertEqual(first, evaluate_views(assets))

    def test_nonpositive_target_keeps_reference_error_and_all_direction_rows(self):
        assets = synthetic_assets()
        for asset in assets:
            for xy, pid in zip(asset['anchors']['xy'], asset['anchors']['point_ids']):
                if fold_for_id(pid) == 0:
                    asset['prior']['raw_inverse_depth'][int(xy[1]), int(xy[0])] = -10.
        fits, targets, directions = evaluate_views(assets)
        held = [t for t in targets if t['fold'] == 0]
        self.assertTrue(all(t['prediction_nonpositive'] and t['reference_error'] > 0 for t in held))
        self.assertTrue(all(t['score'] is None and t['source_count'] == 0 for t in held))
        self.assertEqual((15, 300, 600), (len(fits), len(targets), len(directions)))

    def test_failed_fits_remain_fixed_and_unknown(self):
        assets = synthetic_assets()
        for asset in assets:asset['prior']['raw_inverse_depth'][:] = 1.
        fits, rows, pairs = evaluate_views(assets)
        self.assertEqual((15, 300, 600), (len(fits), len(rows), len(pairs)))
        self.assertTrue(all(f['status'] == 'failed' for f in fits))
        self.assertTrue(all(r['score'] is None and r['reference_error'] is None for r in rows))
        result = summarize(assets, fits, rows, pairs)
        self.assertFalse(result['all_cv_complete']);self.assertFalse(result['screening_pass'])
        self.assertTrue(all(i['correlation']['rho'] is None for i in result['images']))

    def test_actual_one_source_and_no_source_geometry_retains_both_directions(self):
        assets = synthetic_assets()
        assets[2]['camera']['c2w'][0][3] = 1000.
        fits, rows, pairs = evaluate_views(assets)
        self.assertEqual((15, 300, 600), (len(fits), len(rows), len(pairs)))
        self.assertTrue(all(r['source_count'] == 1 and r['score'] is None for r in rows if r['image'] != NAMES[2]))
        self.assertTrue(all(r['source_count'] == 0 and r['score'] is None for r in rows if r['image'] == NAMES[2]))
        result = summarize(assets, fits, rows, pairs)
        self.assertFalse(result['screening_pass'])
        self.assertTrue(all(f['tertiles'] is None for i in result['images'] for f in i['folds']))

    def test_one_source_stays_out_of_primary_and_missing_error_distribution_is_retained(self):
        rows = [dict(score=float(i) if i < 15 else None, reference_error=float(i),
                     source_count=2 if i < 15 else 1, status='two_sources' if i < 15 else 'one_source_only')
                for i in range(20)]
        s = summarize_rows(rows)
        self.assertEqual(15, s['primary_count']);self.assertEqual({'0': 0, '1': 5, '2': 15}, s['source_counts'])
        self.assertAlmostEqual(17., s['excluded_reference_error']['mean'])
        self.assertAlmostEqual(7., s['included_reference_error']['mean'])
        self.assertEqual(5, s['by_status']['one_source_only']['reference_error']['n'])

    def test_tertiles_use_score_only_and_preserve_ties_and_insufficient_groups(self):
        rows = [dict(score=float(i), reference_error=2. * i) for i in range(15)]
        contrast = tertile_contrast(rows)
        self.assertEqual('available', contrast['status']);self.assertAlmostEqual(20., contrast['high_minus_low'])
        self.assertEqual(5, contrast['low_error']['n']);self.assertEqual(5, contrast['high_error']['n'])
        changed = [{**r, 'reference_error': 100. - r['reference_error']} for r in rows]
        other = tertile_contrast(changed)
        self.assertEqual(contrast['q1'], other['q1']);self.assertEqual(contrast['q2'], other['q2'])
        self.assertAlmostEqual(-20., other['high_minus_low'])
        self.assertEqual('tied_tertile_boundaries', tertile_contrast([dict(score=0., reference_error=1.)] * 15)['reason'])
        self.assertIsNone(tertile_contrast(rows[:5])['high_minus_low'])

    def test_screening_requires_every_image_and_fold_and_fixed_support(self):
        assets = synthetic_assets();fits = [dict(status='passed')] * 15
        rows = [dict(image=a['name'], point_id=i, fold=i % 5, score=float(i), reference_error=float(i),
                     source_count=2, status='two_sources') for a in assets for i in range(50)]
        result = summarize(assets, fits, rows, [])
        self.assertTrue(result['screening_pass'])
        for r in rows:
            if r['image'] == NAMES[2]:r['reference_error'] = 100. - r['reference_error']
        result = summarize(assets, fits, rows, [])
        self.assertFalse(result['screening_pass'])
        self.assertLess(result['images'][2]['correlation']['rho'], 0)
        for r in rows:
            if r['image'] == NAMES[0] and r['point_id'] >= 20:
                r.update(score=None, source_count=1, status='one_source_only')
        self.assertFalse(summarize(assets, fits, rows, [])['images'][0]['screening_support_pass'])


class CrossviewEntryGuards(unittest.TestCase):
    def test_frozen_plan_hash_and_contract(self):
        plan = ROOT / 'configs/leaves-crossview-prior-plan.json'
        self.assertEqual(PLAN_SHA256, sha(plan.read_bytes()))
        config = json.loads(plan.read_text())
        self.assertEqual(config['input_contract']['sha256'], sha((ROOT / config['input_contract']['path']).read_bytes()))

    def test_output_location_rejected_before_creation_and_configuration_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);roots = {k: base / k for k in ['train', 'anchors', 'priors']}
            for path in roots.values():path.mkdir()
            output = roots['train'] / 'new-output'
            with self.assertRaisesRegex(ValueError, 'overlaps'):entry.prevalidate_output(output, roots)
            self.assertFalse(output.exists())
            for p in [base / 'evaluation' / 'train' / 'config.json', base / 'metrics.json']:
                p.parent.mkdir(parents=True, exist_ok=True);p.write_text('{}')
                with self.assertRaises(ValueError):entry.config_file(p)

    def test_help_existing_output_and_no_model_imports(self):
        cmd = [sys.executable, str(ROOT / 'scripts/diagnose_crossview_priors.py')]
        p = subprocess.run(cmd + ['--help'], capture_output=True, text=True)
        self.assertEqual(0, p.returncode, p.stderr)
        with tempfile.TemporaryDirectory() as directory:
            args = sum((['--' + k, '/nonexistent'] for k in ['train-data', 'anchors', 'priors', 'input-allowlist', 'plan']), [])
            p = subprocess.run(cmd + args + ['--output', directory], capture_output=True, text=True)
            self.assertNotEqual(0, p.returncode);self.assertIn('FileExistsError', p.stderr)
            self.assertEqual([], list(Path(directory).iterdir()))
        self.assertFalse(any(m in sys.modules for m in ['torch', 'gsplat', 'depth_anything_v2']))


@unittest.skipUnless(os.environ.get('GEOSPLAT_D2_RUN'), 'Explicit completed D2 diagnostic output not supplied')
class RealCrossviewEvidence(unittest.TestCase):
    def test_complete_output_cpu_and_unknown_records(self):
        root = Path(os.environ['GEOSPLAT_D2_RUN'])
        read = lambda n: json.loads((root / n).read_text())
        s, r, access = read('summary.json'), read('resources.json'), read('input-access.json')
        self.assertEqual((15, 610, 1220), (s['fit_records'], s['target_rows'], s['direction_rows']))
        self.assertEqual(14, len(access['assets']));self.assertFalse(access['denials'])
        self.assertEqual('passed', r['status']);self.assertEqual('CPU', r['device'])
        self.assertLess(r['peak_rss_mib'], 2048);self.assertLess(r['elapsed_wall_seconds'], 600)
        self.assertFalse(r['cuda_initialized']);self.assertFalse(r['rendering']);self.assertFalse(r['model_inference'])
        self.assertEqual(0, r['optimizer_updates'])
        with (root / 'targets.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        with (root / 'directions.csv').open() as stream:
            pairs = list(csv.DictReader(stream))
        self.assertEqual((610, 1220), (len(rows), len(pairs)))
        groups = {g['point_id']: g['fold'] for g in read('point-folds.json')}
        for row in rows:
            self.assertEqual(groups[int(row['point_id'])], int(row['fold']))
            sources = [p for p in pairs if p['image'] == row['image'] and p['anchor_index'] == row['anchor_index']]
            self.assertEqual(2, len(sources))
            self.assertEqual(sum(p['status'] == 'available' for p in sources), int(row['source_count']))
            self.assertEqual(row['score'] == '', int(row['source_count']) < 2)


if __name__ == '__main__':
    unittest.main()
