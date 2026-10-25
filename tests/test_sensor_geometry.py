"""Meaningful synthetic gates for known-camera measured-track preparation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geosplat_r import sensor_geometry as g
spec = importlib.util.spec_from_file_location('sensor_entry', ROOT/'scripts/prepare_sensor_anchors_cpu.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixture(point=(.2, .1, 6)):
    views, rays = [], []
    for index, cx in enumerate([-1., 0., 1.]):
        view = dict(image_id=[0, 12, 23][index], camera_id=[0, 7, 11][index],
                    name='view'+str(index)+'.JPG', qvec_wxyz=[1, 0, 0, 0], tvec=[-cx, 0, 0],
                    camera=dict(model='THIN_PRISM_FISHEYE', width=6048, height=4032,
                                params=[3600, 3600, 3024, 2016, 0, 0, 0, 0, 0, 0, 0, 0]))
        params, r, t = g.camera_arrays(view)
        camera_point = r@np.array(point)+t
        # Projection mathematics also works for negative Z in the cheirality fixture.
        ray = camera_point[:2]/camera_point[2]
        xy = g.forward(params, np.r_[ray, 1.])-.5
        view['xy'] = [xy.tolist()]
        rays.append(ray)
        views.append(view)
    return views, rays


class SensorGeometryTests(unittest.TestCase):
    def test_half_pixel_and_native_work_inverse(self):
        xy = np.array([[0, 0], [2, 3], [1005, 669]], dtype=float)
        np.testing.assert_allclose(g.native_center_to_work(g.work_to_native_corner(xy)-.5), xy)
        np.testing.assert_allclose(g.native_center_to_work([[0, 0]]), [[-5/12, -5/12]])

    def test_full_fisheye_projection_not_pinhole(self):
        params = [600, 600, 504, 336, 0, 0, 0, 0, 0, 0, 0, 0]
        uv = g.forward(params, np.array([1., 0., 1.]))
        self.assertAlmostEqual(uv[0], 504+600*np.pi/4)
        self.assertGreater(abs(uv[0]-(504+600)), 100)
        np.testing.assert_array_equal(g.forward(params, [0, 0, 2]), [504, 336])

    def test_rounded_quaternion_normalized_and_bad_norm_rejected(self):
        r = g.rotation([.707107, 0, .707107, 0])
        np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1)
        for q in [[0, 0, 0, 0], [2, 0, 0, 0], [np.nan, 0, 0, 0]]:
            with self.assertRaises(ValueError):g.rotation(q)

    def test_two_and_three_view_DLT_recovers_world_and_axial_Z(self):
        views, rays = fixture()
        for count in [2, 3]:
            record, reason = g.triangulate([(i, 0) for i in range(count)], views, rays[:count])
            self.assertIsNone(reason)
            np.testing.assert_allclose(record['xyz_m'], [.2, .1, 6], atol=1e-10)
            for obs in record['observations']:
                self.assertAlmostEqual(obs['Z_m'], 6)
            self.assertGreater(np.linalg.norm(np.r_[rays[0], 1.])*6, 6)

    def test_negative_cheirality_rejected(self):
        views, rays = fixture((.2, .1, -6))
        record, reason = g.triangulate([(0, 0), (1, 0)], views, rays[:2])
        self.assertIsNone(record)
        self.assertEqual(reason, 'nonpositive_Z')

    def test_insufficient_angle_and_border_rejected(self):
        views, rays = fixture((.2, .1, 600))
        self.assertEqual(g.triangulate([(0, 0), (1, 0)], views, rays[:2])[1], 'angle_below_1_5deg')
        views, rays = fixture()
        views[0]['xy'][0][0] = 1
        self.assertEqual(g.triangulate([(0, 0), (1, 0)], views, rays[:2])[1], 'measurement_border')

    def test_real_measurements_retained_not_reprojections(self):
        views, rays = fixture()
        views[0]['xy'][0][1] += .3
        # Analytic inverse for zero-coefficient fisheye, independent of pycolmap.
        normalized = (np.array(views[0]['xy'][0])+.5-np.array([504, 336]))/600
        theta = np.linalg.norm(normalized)
        rays[0] = normalized*np.tan(theta)/theta
        record, reason = g.triangulate([(0, 0), (1, 0), (2, 0)], views, rays)
        self.assertIsNone(reason)
        self.assertEqual(record['observations'][0]['xy'], views[0]['xy'][0])
        self.assertGreater(record['observations'][0]['reprojection_work_px'], .01)

    def test_large_reprojection_residual_rejected(self):
        views, rays = fixture()
        views[0]['xy'][0][1] += 10
        self.assertEqual(g.triangulate([(0, 0), (1, 0)], views, rays[:2])[1], 'reprojection_above_2px')

    def test_track_union_conflict_rejects_entire_component(self):
        edges = [((0, 0), (1, 0)), ((1, 0), (2, 0))]
        accepted, rejected = g.components(edges)
        self.assertEqual(accepted, [[(0, 0), (1, 0), (2, 0)]])
        self.assertFalse(rejected)
        accepted, rejected = g.components(edges+[((1, 0), (0, 1))])
        self.assertFalse(accepted)
        self.assertEqual(len(rejected[0]), 4)
        self.assertEqual(g.components(edges), g.components(list(reversed(edges))))

    def test_anchor_collision_nearest_Z_and_support_gate(self):
        views, _ = fixture()
        tracks = []
        for i in range(60):
            tracks.append(dict(observations=[dict(image_index=j, feature_index=i, xy=[50+(i%12)*80, 50+(i//12)*100],
                Z_m=6., reprojection_work_px=0.) for j in range(3)]))
        tracks.append(dict(observations=[dict(image_index=0, feature_index=999, xy=[50.1, 50.1],
                                             Z_m=3., reprojection_work_px=.2)]))
        result = g.select_anchors(tracks, views)
        self.assertTrue(all(r['support_passed'] for r in result))
        self.assertEqual(result[0]['anchor_count'], 60)
        self.assertEqual(result[0]['collisions_removed'], 1)
        self.assertEqual(result[0]['anchors'][0]['feature_index'], 999)
        self.assertFalse(g.select_anchors(tracks[:59], views)[0]['support_passed'])
        compact = copy.deepcopy(tracks[:60])
        for i, track in enumerate(compact):
            for obs in track['observations']:obs['xy'] = [20+i*.7, 20]
        self.assertFalse(g.select_anchors(compact, views)[0]['support_passed'])

    def test_inverse_sampling_not_depth_interpolation(self):
        q = np.array([[1., .5], [1., .5]])
        sampled, valid = g.sample_inverse_z(q, [[.5, .5]])
        self.assertTrue(valid[0]);self.assertAlmostEqual(1/sampled[0], 4/3)
        self.assertNotAlmostEqual(1/sampled[0], 1.5)

    def test_invalid_contributor_and_unknown_never_filled(self):
        for bad in [0., -1., np.inf, np.nan, 1e-7]:
            q = np.ones((3, 3));q[1, 1] = bad
            sampled, valid = g.sample_inverse_z(q, [[0, 0], [-.1, 0], [np.nan, 0]])
            self.assertFalse(valid.any())
            self.assertTrue(np.isnan(sampled).all())
        sampled, valid = g.sample_inverse_z(np.ones((3, 3)), [[.5, .5], [2, 2]])
        self.assertEqual(valid.tolist(), [True, False])

    def test_inventory_hash_mapping_extra_file_and_symlink_rejection(self):
        views, _ = fixture()
        for v in views:del v['xy']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/'rgb').mkdir()
            (root/'cameras.json').write_text(json.dumps(views))
            for v in views:(root/'rgb'/v['name']).write_bytes(b'synthetic RGB placeholder')
            config = dict(names=[v['name'] for v in views], inputs={str(p.relative_to(root)):
                dict(bytes=p.stat().st_size, sha256=g.sha256(p)) for p in root.rglob('*') if p.is_file()})
            self.assertEqual(len(g.validate_predictor(root, config)[0]), 3)
            extra = root/'label.npy';extra.write_bytes(b'forbidden')
            with self.assertRaises(ValueError):g.validate_predictor(root, config)
            extra.unlink()
            rgb = root/'rgb'/views[0]['name'];original=rgb.read_bytes();rgb.write_bytes(b'changed')
            with self.assertRaises(ValueError):g.validate_predictor(root, config)
            rgb.write_bytes(original);link=root/'escape';link.symlink_to(root/'rgb', target_is_directory=True)
            with self.assertRaises(ValueError):g.validate_predictor(root, config)

    def test_supervisor_refuses_existing_output_and_changed_protocol(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);out = root/'existing';out.mkdir();(out/'keep').write_text('preserve')
            args = SimpleNamespace(plan=ROOT/'configs/terrace-sensor-diagnostic-plan.json',
                                   output=out, data=root/'data', source=ROOT)
            with self.assertRaises(ValueError):entry.run(args)
            self.assertEqual((out/'keep').read_text(), 'preserve')
            wrong=root/'wrong.json';wrong.write_text('{}');args.plan=wrong
            with self.assertRaises(ValueError):entry.run(args)


if __name__ == '__main__':
    unittest.main()
