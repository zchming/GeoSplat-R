"""Synthetic geometry, raw COLMAP semantics and exact input isolation checks."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geosplat_r.raw_track_diagnostics import (accept_model, read_cameras, read_images, read_tracks,
    source_score, triangulated_projection, evaluate_tracks, summarize, contrast, camera_arrays)
from geosplat_r.train_diagnostics import NAMES, fold_for_id
spec = importlib.util.spec_from_file_location('raw_entry', ROOT / 'scripts/diagnose_raw_tracks.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def camera(center=(0., 0., 0.), name=None):
    pose = np.eye(4); pose[:3, 3] = center
    return dict(name=name, K=[[100., 0, 250.], [0, 100., 188.], [0, 0, 1.]],
                c2w=pose.tolist(), width=504, height=378, image_sha256='synthetic')


def project(point, cam):
    k, _, p = camera_arrays(cam)
    q = k @ (p @ np.r_[point, 1.])
    return q[:2] / q[2]


def fixture():
    cams = [camera((-1, 0, 0), NAMES[0]), camera((0, 0, 0), NAMES[1]), camera((1, 0, 0), NAMES[2])]
    xyz = {11: np.array([.3, -.2, 5.]), 22: np.array([-.1, .4, 7.])}
    cb = struct.pack('<QiiQQdddd', 1, 7, 1, 504, 378, 100., 100., 250., 188.)
    ib = struct.pack('<Q', 3)
    ids = [10, 30, 50]  # IDs are not contiguous indices.
    for iid, cam in zip(ids, cams):
        center = np.asarray(cam['c2w'])[:3, 3]
        ib += struct.pack('<idddddddi', iid, 1., 0., 0., 0., *(-center), 7)
        ib += cam['name'].encode() + b'\0' + struct.pack('<Q', 3)
        for pid, point in xyz.items():
            ib += struct.pack('<ddq', *project(point, cam), pid)
        ib += struct.pack('<ddq', 1., 2., -1)
    tb = struct.pack('<Q', 2)
    for idx, pid in enumerate(xyz):
        tb += struct.pack('<QdddBBBdQ', pid, 999., 888., 777., 10, 20, 30, 444., 3)
        tb += b''.join(struct.pack('<ii', iid, idx) for iid in ids)
    manifest = dict(scene='leaves', split='train', protocol='leaves-undistorted-v1', resolution=[504, 378],
                    initial_point_count=2, train_images=[{'name': c['name'], 'sha256': c['image_sha256']} for c in cams])
    prep = dict(train_names=NAMES, intrinsics_protocol='leaves-undistorted-v1', point_count=2,
                normalization=dict(center=[0, 0, 0], scale=1., source='3 training camera centers only'))
    return dict(cameras=cams, manifest=manifest, preparation=prep, cameras_bin=cb, images_bin=ib, tracks_bin=tb)


class RawBinaryAndInputAcceptance(unittest.TestCase):
    def test_raw_measurements_are_not_fitted_xyz_projections(self):
        blobs = fixture();cams, obs, accepted = accept_model(blobs)
        self.assertTrue(accepted['accepted']);self.assertEqual(2, accepted['points'])
        self.assertEqual({'3': 2}, accepted['track_length_counts'])
        np.testing.assert_allclose(obs[11][NAMES[0]], [276., 184.])
        self.assertGreater(np.linalg.norm(obs[11][NAMES[0]] - project([999., 888., 777.], cams[0])), 10)

    def test_fitted_xyz_rgb_error_bytes_are_not_numeric_inputs(self):
        a = fixture();b = copy.deepcopy(a)
        data = bytearray(b['tracks_bin'])
        # Each point's ignored section is 35 bytes, irrespective of its numeric contents.
        offset = 8
        for _ in range(2):
            data[offset + 8:offset + 43] = b'\xff' * 35
            offset += 51 + 3 * 8
        b['tracks_bin'] = bytes(data)
        ca, oa, _ = accept_model(a);cb, ob, _ = accept_model(b)
        self.assertEqual(evaluate_tracks(ca, oa), evaluate_tracks(cb, ob))

    def test_truncation_counts_and_trailing_bytes_rejected(self):
        b = fixture()
        for key, fn in [('cameras_bin', read_cameras), ('images_bin', read_images), ('tracks_bin', read_tracks)]:
            for data in [b[key][:-1], b[key] + b'?', struct.pack('<Q', 2**63)]:
                with self.subTest(key=key), self.assertRaises(ValueError):fn(data)

    def test_test_view_and_duplicate_image_ids_rejected(self):
        data = fixture()['images_bin']
        with self.assertRaises(ValueError):read_images(data.replace(b'image001.png', b'image000.png'))
        images = read_images(data)
        self.assertEqual({10, 30, 50}, set(images))
        bad = bytearray(data)
        second_offset = 8 + 64 + len(NAMES[0]) + 1 + 8 + 3 * 24
        bad[second_offset:second_offset+4] = struct.pack('<i', 10)
        with self.assertRaises(ValueError):read_images(bytes(bad))

    def test_nonreciprocal_duplicate_and_invalid_track_reference_rejected(self):
        for replacement in [(10, 1), (999, 0), (10, 999), (10, -1), (30, 0)]:
            b = fixture();data = bytearray(b['tracks_bin'])
            data[8 + 51:8 + 59] = struct.pack('<ii', *replacement)
            b['tracks_bin'] = bytes(data)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):accept_model(b)

    def test_prepared_pose_intrinsics_normalization_and_manifest_mismatch_rejected(self):
        for part in ['K', 'pose', 'scale', 'split', 'count']:
            b = fixture()
            if part == 'K':b['cameras'][0]['K'][0][0] += 1
            elif part == 'pose':b['cameras'][0]['c2w'][0][3] += .1
            elif part == 'scale':b['preparation']['normalization']['scale'] = 2.
            elif part == 'split':b['manifest']['split'] = 'test'
            else:b['manifest']['initial_point_count'] = 3
            with self.subTest(part=part), self.assertRaises(ValueError):accept_model(b)


class SensitivityGeometry(unittest.TestCase):
    def test_nonplanar_points_and_rotation_similarity(self):
        sources = [camera((-1, 0, 0)), camera((1, 0, 0))];target = camera((0, .2, .5))
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        for point in [[.3, .4, 5.], [-1.2, -.3, 9.], [.7, -.1, 3.]]:
            xy = [project(point, c) for c in sources]
            uv, recovered = triangulated_projection(xy, sources, target)
            np.testing.assert_allclose(recovered, point, atol=1e-12)
            np.testing.assert_allclose(uv, project(point, target), atol=1e-12)
            transformed = copy.deepcopy([*sources, target])
            for cam in transformed:
                c = np.asarray(cam['c2w']);c[:3, :3] = rotation @ c[:3, :3]
                c[:3, 3] = rotation @ (c[:3, 3] * 2) + [3, 2, 1];cam['c2w'] = c.tolist()
            score = source_score(xy, sources, target);other = source_score(xy, transformed[:2], transformed[2])
            self.assertAlmostEqual(score['score'], other['score'], places=7)
            np.testing.assert_allclose(score['projection'], other['projection'], atol=1e-10)

    def test_jacobian_matches_independent_rectified_first_order_formula(self):
        sources = [camera((-1, 0, 0)), camera((1, 0, 0))];target = camera((0, 0, 1))
        point = [.3, -.2, 5.];xy = np.array([project(point, c) for c in sources])
        result = source_score(xy, sources, target)
        d = 2 - (xy[0, 0] - xy[1, 0]) / 100
        sx, sy = xy.sum(axis=0) - [500, 376]
        expected = np.array([[1/d + sx/(100*d*d), 0, 1/d - sx/(100*d*d), 0],
                             [sy/(100*d*d), 1/d, -sy/(100*d*d), 1/d]])
        np.testing.assert_allclose(result['jacobian'], expected, atol=2e-7, rtol=0)
        self.assertAlmostEqual(np.linalg.norm(expected), result['score'], places=7)

    def test_held_target_change_affects_only_label_for_its_row(self):
        c, obs, _ = accept_model(fixture());before = evaluate_tracks(c, obs)
        after_obs = copy.deepcopy(obs);after_obs[11][NAMES[0]] += [12, -7]
        after = evaluate_tracks(c, after_obs)
        a = next(r for r in before if r['point_id'] == 11 and r['image'] == NAMES[0])
        b = next(r for r in after if r['point_id'] == 11 and r['image'] == NAMES[0])
        for key in ['score', 'angle_control', 'projection', 'xyz', 'jacobian', 'status', 'fold']:
            self.assertEqual(a[key], b[key])
        self.assertGreater(b['residual_px'], 10)

    def test_projected_target_outside_image_retained(self):
        c, obs, _ = accept_model(fixture());c[0]['c2w'][0][3] = -100
        rows = evaluate_tracks(c, obs);r = next(r for r in rows if r['image'] == NAMES[0])
        self.assertGreater(r['projection'][0], 504)
        self.assertIsNotNone(r['score']);self.assertGreater(r['residual_px'], 100)

    def test_near_zero_baseline_parallel_and_behind_camera_status(self):
        point = [.1, .2, 5.];c = camera();xy = [project(point, c)] * 2
        self.assertIsNone(source_score(xy, [c, c], c)['score'])
        sources = [camera((-1, 0, 0)), camera((1, 0, 0))]
        back = camera((0, 0, 10))
        result = source_score([project(point, x) for x in sources], sources, back)
        self.assertIn('nonpositive_camera_z', result['status'])
        self.assertIsNone(result['projection'])

    def test_probe_failure_keeps_base_projection_and_error(self):
        import geosplat_r.raw_track_diagnostics as module
        original = module.triangulated_projection
        c, obs, _ = accept_model(fixture())
        calls = 0
        def fail_probe(*args):
            nonlocal calls
            calls += 1
            if calls % 2 == 0:raise ValueError('synthetic_probe_failure')
            return original(*args)
        with patch.object(module, 'triangulated_projection', side_effect=fail_probe):
            rows = evaluate_tracks(c, obs)
        self.assertTrue(all(r['score'] is None and r['residual_px'] is not None for r in rows))
        self.assertTrue(all(r['status'] == 'probe_synthetic_probe_failure' for r in rows))

    def test_incomplete_tracks_stay_as_unknown_rows(self):
        c, obs, _ = accept_model(fixture());del obs[11][NAMES[0]]
        rows = evaluate_tracks(c, obs);self.assertEqual(6, len(rows))
        missing = [r for r in rows if r['point_id'] == 11]
        self.assertTrue(all(r['status'] == 'missing_three_view_track' and r['score'] is None for r in missing))

    def test_consistent_wrong_correspondences_can_have_small_scores_and_zero_residual(self):
        c, _, _ = accept_model(fixture());wrong = np.array([0., .1, 8.]);truth = np.array([0., .1, 4.])
        obs = {11: {cam['name']: project(wrong, cam) for cam in c}}
        row = next(r for r in evaluate_tracks(c, obs) if r['image'] == NAMES[1])
        self.assertLess(row['residual_px'], 1e-10);self.assertLess(row['score'], 1.1)
        self.assertGreater(np.linalg.norm(np.array(row['xyz']) - truth), 3)


class FrozenScreening(unittest.TestCase):
    def rows(self, count=150):
        ids = []
        for fold in range(5):
            candidates = [i for i in range(10000) if fold_for_id(i) == fold]
            ids.extend(candidates[:count//5])
        rows = []
        for rank, pid in enumerate(ids):
            for name in NAMES:
                rows.append(dict(point_id=pid, fold=fold_for_id(pid), image=name, track_length=3,
                    score=float(rank), angle_control=float(rank+1), residual_px=float(rank),
                    status='available', angle_status='available'))
        return rows

    def test_positive_screen_and_same_subset_angle_no_extra_contribution(self):
        rows = self.rows();summary = summarize(rows, len(rows)//3)
        self.assertTrue(summary['all_observability_passed']);self.assertTrue(summary['all_direction_passed'])
        for image in summary['per_image'].values():
            self.assertEqual(0., image['same_subset']['rho_score_minus_angle'])

    def test_one_view_reversal_stops_and_all_negative_rows_retained(self):
        rows = self.rows()
        for row in rows:
            if row['image'] == NAMES[0]:row['residual_px'] = 200-row['residual_px']
        summary = summarize(rows, len(rows)//3)
        self.assertEqual('direction_failed_stop', summary['decision'])
        self.assertEqual(-1., summary['per_image'][NAMES[0]]['score']['correlation']['rho'])

    def test_insufficient_sample_constant_score_ties_and_missing_not_success(self):
        for mode in ['small', 'constant', 'missing']:
            rows = self.rows(25 if mode == 'small' else 150)
            if mode == 'constant':
                for row in rows:row['score'] = 1.
            if mode == 'missing':
                for row in rows:row['score'] = None;row['status'] = 'probe_failed'
            summary = summarize(rows, len(rows)//3)
            self.assertEqual('insufficient_support_stop', summary['decision'])
        tied = contrast([dict(score=1., residual_px=float(i)) for i in range(30)], 'score')
        self.assertEqual('tied_tertiles', tied['reason'])

    def test_incomplete_duplicate_records_rejected(self):
        rows = self.rows()
        for bad in [rows[:-1], rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):summarize(bad, len(rows)//3)


class IsolationAndEntry(unittest.TestCase):
    def loader(self, root):
        files = [('train', 'cameras.json'), ('train', 'input-manifest.json'),
                 ('train', 'triangulated/cameras.bin'), ('train', 'triangulated/images.bin'),
                 ('train', 'triangulated/points3D.bin'), ('preparation', 'preparation.json')]
        train = root/'train';train.mkdir();prep = root/'preparation.json'
        plan, allow = {'protocol': 'synthetic', 'inputs': []}, {'protocol': 'synthetic', 'plan_sha256': entry.PLAN_SHA256, 'files': []}
        for alias, name in files:
            path = train/name if alias == 'train' else prep
            path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(b'fixture')
            row = dict(root=alias, path=name, bytes=7, sha256=hashlib.sha256(b'fixture').hexdigest())
            plan['inputs'].append(row);allow['files'].append(dict(row, absolute_path=str(path)))
        return train, prep, plan, allow

    def test_allowlist_rejects_extra_duplicate_traversal_symlink_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);train, prep, plan, allow = self.loader(root)
            entry.Inputs(train, prep, plan, allow)
            for mode in ['extra', 'duplicate', 'traversal', 'size', 'path']:
                bad = copy.deepcopy(allow)
                if mode == 'extra':bad['files'].append(bad['files'][0])
                elif mode == 'duplicate':bad['files'][1] = bad['files'][0]
                elif mode == 'traversal':bad['files'][0]['path'] = '../test/image.png'
                elif mode == 'size':bad['files'][0]['bytes'] += 1
                else:bad['files'][0]['absolute_path'] = str(root/'test/data')
                with self.subTest(mode=mode), self.assertRaises(ValueError):entry.Inputs(train, prep, plan, bad)
            path = train/'cameras.json';path.unlink();path.symlink_to(prep)
            with self.assertRaises(ValueError):entry.Inputs(train, prep, plan, allow)

    def test_hash_drift_fails_before_decoding(self):
        with tempfile.TemporaryDirectory() as d:
            train, prep, plan, allow = self.loader(Path(d));loader = entry.Inputs(train, prep, plan, allow)
            (train/'cameras.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'drift'):loader.read_all()

    def test_boundary_exact_reads_writes_network_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);a = root/'train.bin';a.write_bytes(b'keep');out = root/'new'
            boundary = entry.ReadBoundary([a], out)
            boundary('open', (str(a), 'r', os.O_RDONLY))
            for args in [(str(a), 'w', os.O_WRONLY), (str(root/'test/image.png'), 'r', os.O_RDONLY),
                         (str(root/'old.pt'), 'r', os.O_RDONLY)]:
                with self.assertRaises(RuntimeError):boundary('open', args)
            with self.assertRaises(RuntimeError):boundary('socket.connect', ())
            link = root/'alias';link.symlink_to(root/'test')
            with self.assertRaises(RuntimeError):boundary('open', (bytes(str(link/'x'), 'utf8'), 'r', os.O_RDONLY))
            self.assertEqual({str(a)}, boundary.reads)

    def test_help_and_existing_output_need_no_heavy_imports(self):
        code = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','gsplat','pycolmap','depth_anything_v2'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        p = subprocess.run([sys.executable, '-c', code, str(ROOT/'scripts/diagnose_raw_tracks.py')], capture_output=True, text=True)
        self.assertEqual(0, p.returncode, p.stderr);self.assertIn('--validate-only', p.stdout)
        with tempfile.TemporaryDirectory() as d:
            argv = sum((['--'+key, '/nonexistent'] for key in ['train-data','preparation','plan','input-allowlist']), [])
            with self.assertRaises(FileExistsError):entry.main(argv+['--output', d])
            self.assertEqual([], list(Path(d).iterdir()))

    def test_output_overlap_rejected_and_frozen_proposal_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);train = root/'prepared/train';train.mkdir(parents=True);prep = root/'prepared/preparation.json';prep.write_text('{}')
            for out in [train/'new', ROOT/'new-output', root/'prepared', root]:
                with self.assertRaises((ValueError, FileExistsError)):entry.output_location(out, train, prep)
        self.assertEqual(entry.PLAN_SHA256, entry.digest((ROOT/'configs/leaves-raw-track-plan.json').read_bytes()))
        # Frozen JSON retains the historical proposal identity; report wording may change.
        plan = json.loads((ROOT/'configs/leaves-raw-track-plan.json').read_text())
        self.assertEqual(entry.PROPOSAL_SHA256, plan['plan_sha256'])


if __name__ == '__main__':
    unittest.main()
