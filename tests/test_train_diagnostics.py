"""CPU-only leakage, path and numerical checks for the frozen D1 diagnostic."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geosplat_r.train_diagnostics import (PLAN_SHA256, AllowlistedInputs, analyze_arrays,
    average_ranks, describe, distance_summary, fit_affine, fold_for_id,
    grouped_crossfit, nearest_distance, spearman, tile_indices, verify_full_fit)


def synthetic():
    yy, xx = np.mgrid[:378, :504]
    raw = (xx + 2 * yy).astype(np.float64) / 1024
    xy = np.array([[10 + i * 4, 10 + (i % 17) * 8] for i in range(100)], dtype=np.float64)
    samples = raw[xy[:, 1].astype(int), xy[:, 0].astype(int)]
    return ({'raw_inverse_depth': raw, 'inverse_z': raw * 2 + .1,
             'valid': np.ones(raw.shape, bool)},
            {'xy': xy, 'inverse_z': samples * 2 + .1, 'point_ids': np.arange(100, dtype=np.int64)}, samples)


class DiagnosticMath(unittest.TestCase):
    def test_full_fit_reference_drift_rejected_separately_from_crossfit(self):
        p, a, samples = synthetic()
        reference = fit_affine(samples, a['inverse_z'])
        verify_full_fit(p, a, reference)
        with self.assertRaisesRegex(ValueError, 'reproduction mismatch'):
            verify_full_fit(p, a, {**reference, 'slope': reference['slope'] + .01})

    def test_known_affine_recovery_and_sparse_error_without_dense_truth(self):
        p, a, _ = synthetic()
        s, records, tiles, _ = analyze_arrays(p, a)
        self.assertTrue(s['cv_complete'])
        self.assertLess(s['oof_error']['mean'], 1e-12)
        self.assertAlmostEqual(2, s['full_fit']['slope'])
        self.assertAlmostEqual(.1, s['full_fit']['shift'])
        self.assertEqual(100, len(records));self.assertEqual(108, len(tiles))
        self.assertEqual(504 * 378, sum(t['pixel_count'] for t in tiles))
        self.assertEqual(100, sum(t['anchor_count'] for t in tiles))
        self.assertTrue(all(t['oof_error'] is None for t in tiles if t['oof_count'] < 5))

    def test_same_point_fold_shared_across_views_and_held_targets_do_not_fit(self):
        _, a, samples = synthetic()
        folds, records = grouped_crossfit(samples, a['xy'], a['inverse_z'], a['point_ids'])
        changed = a['inverse_z'].copy()
        held = np.array([fold_for_id(p) == 0 for p in a['point_ids']])
        changed[held] *= 8
        altered, new = grouped_crossfit(samples, a['xy'], changed, a['point_ids'])
        self.assertEqual(folds[0]['parameters'], altered[0]['parameters'])
        for first, second in zip(records, new):
            if first['fold'] == 0:
                self.assertEqual(first['prediction'], second['prediction'])
                self.assertEqual(first['normalizer'], second['normalizer'])
                self.assertNotEqual(first['error'], second['error'])
                self.assertGreater(first['distance'], 0)
        reverse = a['point_ids'][::-1]
        by_point = {r['point_id']: r['fold'] for r in records}
        self.assertEqual([by_point[int(p)] for p in reverse], [fold_for_id(p) for p in reverse])

    def test_cached_full_targets_and_masks_cannot_control_crossfit_predictions(self):
        p, a, _ = synthetic()
        first, rows, _, _ = analyze_arrays(p, a)
        changed = copy.deepcopy(p)
        changed['inverse_z'][:] = 123456
        changed['valid'][:] = False
        second, altered, tiles, _ = analyze_arrays(changed, a)
        self.assertEqual(rows, altered)
        self.assertEqual(first['folds'], second['folds'])
        self.assertEqual(0, second['prior_valid_support_distance']['n'])
        self.assertIsNone(second['prior_valid_support_distance']['mean'])
        self.assertTrue(all(t['prior_valid_distance']['mean'] is None for t in tiles))

    def test_nonpositive_predictions_are_kept_as_errors(self):
        ids = np.arange(150, dtype=np.int64)
        held = np.array([fold_for_id(p) == 0 for p in ids])
        samples = np.linspace(1, 2, len(ids))
        y = 2 * samples + 1
        samples[held] = -10;y[held] = 1
        xy = np.column_stack([np.arange(150) + 5., np.full(150, 40.)])
        folds, rows = grouped_crossfit(samples, xy, y, ids)
        self.assertEqual('passed', folds[0]['status'])
        observed = [r for r in rows if r['fold'] == 0]
        self.assertEqual(int(held.sum()), len(observed))
        self.assertTrue(all(r['prediction_nonpositive'] and r['error'] > 0 for r in observed))

    def test_bad_fits_and_insufficient_folds_are_not_refolded(self):
        for x, y in [(np.ones(25), np.ones(25)), (np.arange(25.), 30 - np.arange(25.)),
                     (np.arange(19.), np.ones(19)), (np.arange(25.), np.full(25, np.nan))]:
            with self.assertRaises(ValueError):fit_affine(x, y)
        n = 19
        folds, rows = grouped_crossfit(np.arange(n), np.column_stack([np.arange(n), np.ones(n)]),
                                      np.arange(n) + 1, np.arange(n, dtype=np.int64))
        self.assertEqual(5, len(folds));self.assertEqual(n, len(rows))
        self.assertTrue(all(f['status'] in ['failed', 'unsupported'] for f in folds))
        self.assertTrue(all(r['error'] is None for r in rows))

    def test_distance_chunk_boundary_and_display_bins(self):
        query = np.zeros((4100, 2));query[:, 0] = 3;query[:, 1] = 4
        calls = []
        d = nearest_distance(query, np.array([[0., 0.]]), lambda: calls.append(1))
        np.testing.assert_allclose(d, 5 / 630, rtol=0, atol=1e-15)
        self.assertEqual(2, len(calls))
        self.assertEqual([2, 1, 1, 1, 2], distance_summary([0, .009, .01, .025, .05, .1, 1])['bin_counts'])
        self.assertIsNone(describe([])['mean'])

    def test_average_ranks_constants_and_five_pair_minimum(self):
        self.assertEqual([1.5, 1.5, 3., 4.5, 4.5], average_ranks([1, 1, 3, 4, 4]).tolist())
        self.assertAlmostEqual(-1, spearman([1, 1, 3, 4, 4], [-1, -1, -3, -4, -4])['rho'])
        for x, y in [([1] * 5, range(5)), (range(4), range(4)), ([1, 2, 3, 4, np.nan], range(5))]:
            self.assertIsNone(spearman(x, y)['rho'])

    def test_invalid_coordinates_and_duplicate_point_ids_rejected(self):
        for xy in [np.array([[504., 3.]]), np.array([[0., np.nan]])]:
            with self.assertRaises(ValueError):tile_indices(xy)
        _, a, samples = synthetic()
        a['point_ids'][1] = a['point_ids'][0]
        with self.assertRaisesRegex(ValueError, 'unique_int64'):
            grouped_crossfit(samples, a['xy'], a['inverse_z'], a['point_ids'])


class DiagnosticInputGuards(unittest.TestCase):
    def fixture(self, root):
        roots = {k: root / k for k in ['train', 'anchors', 'priors']}
        plan = json.loads((ROOT / 'configs/leaves-train-error-coverage-plan.json').read_text())
        plan = copy.deepcopy(plan);manifest = {'protocol': plan['protocol'], 'plan_sha256': PLAN_SHA256, 'files': []}
        for i, row in enumerate(plan['allowlisted_files']):
            path = roots[row['root']] / row['path'];path.parent.mkdir(parents=True, exist_ok=True)
            data = ('fixture-' + str(i)).encode();path.write_bytes(data)
            row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            manifest['files'].append({**row, 'absolute_path': str(path.resolve())})
        return roots, plan, manifest

    def test_exact_fourteen_reads_and_unknown_asset_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            roots, plan, manifest = self.fixture(Path(directory))
            loader = AllowlistedInputs(roots, plan, manifest)
            for row in plan['allowlisted_files']:loader.read(row['root'], row['path'])
            loader.read('train', 'cameras.json')
            self.assertEqual(14, len(loader.accesses))
            with self.assertRaisesRegex(ValueError, 'outside'):loader.read('train', '../test/images/image000.png')
            self.assertEqual(1, len(loader.denials))

    def test_extra_and_traversing_paths_rejected_before_asset_read(self):
        with tempfile.TemporaryDirectory() as directory:
            roots, plan, manifest = self.fixture(Path(directory))
            for wrong in ['../test/metrics.json', '/absolute/test.png']:
                bad = copy.deepcopy(manifest);bad['files'][0]['path'] = wrong
                with patch.object(Path, 'read_bytes', side_effect=AssertionError('Unexpected asset read')):
                    with self.assertRaises(ValueError):AllowlistedInputs(roots, plan, bad)
            bad = copy.deepcopy(manifest);bad['files'].append(bad['files'][0])
            with self.assertRaises(ValueError):AllowlistedInputs(roots, plan, bad)

    def test_parent_test_roots_symlink_escape_and_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory);roots, plan, manifest = self.fixture(base)
            with self.assertRaises(ValueError):AllowlistedInputs({**roots, 'train': base}, plan, manifest)
            other = base / 'evaluation' / 'train';other.mkdir(parents=True)
            with self.assertRaises(ValueError):AllowlistedInputs({**roots, 'train': other}, plan, manifest)
            victim = roots['train'] / 'cameras.json';victim.write_bytes(b'drift')
            loader = AllowlistedInputs(roots, plan, manifest)
            with self.assertRaisesRegex(ValueError, 'drift'):loader.read('train', 'cameras.json')
            # Fixture replacement is local temporary test data, never project evidence.
            victim.unlink();outside = base / 'outside.json';outside.write_text('{}');victim.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'Symlink'):AllowlistedInputs(roots, plan, manifest)

    def test_help_and_existing_output_refusal_need_no_assets(self):
        command = [sys.executable, str(ROOT / 'scripts/diagnose_train_support.py')]
        result = subprocess.run(command + ['--help'], capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            args = []
            for key in ['train-data', 'anchors', 'priors', 'input-allowlist', 'plan']:
                args.extend(['--' + key, '/nonexistent'])
            result = subprocess.run(command + args + ['--output', directory], capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode);self.assertIn('FileExistsError', result.stderr)
            self.assertEqual([], list(Path(directory).iterdir()))


@unittest.skipUnless(os.environ.get('GEOSPLAT_D1_RUN'), 'Explicit completed D1 diagnostic output not supplied')
class RealD1Evidence(unittest.TestCase):
    def test_complete_proxy_evidence_and_cpu_resource_boundary(self):
        root = Path(os.environ['GEOSPLAT_D1_RUN'])
        read = lambda n: json.loads((root / n).read_text())
        s, r, access = read('summary.json'), read('resources.json'), read('input-access.json')
        self.assertEqual((15, 324, 610), (s['fold_records'], s['tile_records'], s['anchor_observations']))
        self.assertEqual(14, len(access['assets']));self.assertFalse(access['denials'])
        self.assertEqual('passed', r['status']);self.assertEqual('CPU', r['device'])
        self.assertFalse(r['cuda_initialized']);self.assertFalse(r['rendering']);self.assertEqual(0, r['optimizer_updates'])
        self.assertLess(r['peak_rss_mib'], 2048);self.assertLess(r['elapsed_wall_seconds'], 600)
        groups = read('point-folds.json')
        self.assertEqual(len(groups), len({g['point_id'] for g in groups}))
        self.assertTrue(all(g['fold'] == fold_for_id(g['point_id']) for g in groups))


if __name__ == '__main__':
    unittest.main()
