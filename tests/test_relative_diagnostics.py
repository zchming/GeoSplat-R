"""Synthetic-only ordinal metric, deterministic pairing and sealing checks."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geosplat_r import relative_diagnostics as rd


def arrays(index=0):
    y, x = np.indices((96, 144))
    raw = ((x//3+y//3+index) % 3+1).astype(np.float32)/8
    gt = (1/cv2.resize(raw, (864, 576), interpolation=cv2.INTER_LINEAR)).astype(np.float32)
    gt[100:120, 200:230] = np.inf
    return gt, raw


def tiny_nodes():
    return dict(indices=np.array([0, 1, 2]), cell=np.array([0, 0, 4]), z=np.array([1., 2., 4.]))


class RelativeDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt, cls.raw = arrays()
        cls.report, cls.nodes, cls.groups = rd.evaluate_arrays(cls.gt, cls.raw, 0)

    def test_correct_wrong_constant_and_invalid_closed_forms(self):
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        for q, expected in [([1., .5, .25], 0.), ([.25, .5, 1.], 1.), ([0, 0, 0], .5), ([1, 1, 1], .5)]:
            loss, invalid, ties = rd.relation_loss(q, [1, 2, 4], pairs)
            np.testing.assert_array_equal(loss, np.full(3, expected));self.assertFalse(invalid.any())
            self.assertEqual(ties.all(), expected == .5)
        loss, invalid, _ = rd.relation_loss([np.nan, 0, 0], [1, 2, 4], pairs)
        np.testing.assert_array_equal(loss, [1, 1, .5]);self.assertEqual(invalid.tolist(), [True, True, False])

    def test_monotone_distortion_is_not_metric_accuracy(self):
        z = np.array([1., 2., 4.]);q = (1/z)**2
        self.assertFalse(np.array_equal(1/q, z))
        self.assertFalse(rd.relation_loss(q, z, [[0, 1], [1, 2], [0, 2]])[0].any())

    def test_raw_zero_and_tiny_values_not_masked(self):
        q = np.zeros((4, 4));q[1, 1] = 1e-12
        samples, valid = rd.sample_raw(q, [[0, 0], [.5, .5]])
        self.assertTrue(valid.all());self.assertEqual(samples[0], 0.)
        self.assertGreater(samples[1], 0.)

    def test_four_contributors_even_zero_weight_and_no_edge_filling(self):
        for bad in [np.nan, np.inf, -1.]:
            q = np.ones((3, 3));q[1, 1] = bad
            values, valid = rd.sample_raw(q, [[0, 0], [2, 2], [-.1, 0], [1e300, 0]])
            self.assertFalse(valid.any());self.assertTrue(np.isnan(values).all())

    def test_half_pixel_interpolation_independent_opencv(self):
        gt, q = arrays()
        indices = self.nodes['indices'];y, x = np.divmod(indices, gt.shape[1])
        xy = np.column_stack([(x+.5)/6-.5, (y+.5)/6-.5])
        values, valid = rd.sample_raw(q, xy)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(values, 1/gt.flat[indices].astype(np.float64), rtol=2e-6, atol=1e-7)
        # Native pixel (0,0) center maps outside the first work center, never clamp it.
        self.assertAlmostEqual((.5)/6-.5, -5/12)

    def test_GT_ties_and_contrast_boundary(self):
        z = np.array([1., 1., np.nextafter(1.1, 0), 1.1, np.nextafter(1.1, 2)])
        mask = rd.contrast_mask(z, [[0, 1], [0, 2], [0, 3], [0, 4]])
        self.assertEqual(mask.tolist(), [False, False, True, True])
        with self.assertRaises(ValueError):rd.relation_loss([1, 1], [1, 1], [[0, 1]])
        with self.assertRaises(ValueError):rd.contrast_mask([0, 1], [[0, 1]])

    def test_nodes_pairing_independent_of_predictions_and_deterministic(self):
        nodes = rd.select_nodes(self.gt, 0);groups = rd.make_pairs(nodes, 0)
        self.assertEqual(rd.pair_fingerprint(nodes, groups), self.report['pair_fingerprint'])
        np.testing.assert_array_equal(nodes['indices'], np.sort(nodes['indices']))
        self.assertLessEqual(len(nodes['indices']), 6912)
        changed = rd.score(nodes, groups, np.zeros(len(nodes['indices'])))
        self.assertEqual(changed['loss'], .5)
        self.assertEqual(rd.pair_fingerprint(nodes, groups), self.report['pair_fingerprint'])

    def test_cross_distance_dedup_and_local_same_cell(self):
        nodes = self.nodes
        for kind, groups in self.groups.items():
            for group in groups:
                candidates = group['candidates']
                self.assertEqual(len(set(map(tuple, candidates.tolist()))), len(candidates))
                if not len(candidates):continue
                self.assertTrue((candidates[:, 0] < candidates[:, 1]).all())
                ca, cb = nodes['cell'][candidates].T
                if kind == 'local':np.testing.assert_array_equal(ca, cb)
                else:self.assertTrue((np.maximum(abs(ca//12-cb//12), abs(ca%12-cb%12)) >= 3).all())
                self.assertLessEqual(group['draws'], 256)

    def test_full_synthetic_oracle_constants_and_support(self):
        report = self.report
        self.assertTrue(report['model']['support_passed'])
        self.assertEqual(report['model']['loss'], 0.)
        self.assertEqual(report['controls']['oracle']['loss'], 0.)
        self.assertEqual(report['controls']['constant']['loss'], .5)
        self.assertGreater(report['controls']['shuffle_mean'], .4)
        self.assertEqual(len(report['controls']['shuffled']), 20)

    def test_cell_macro_not_pooled_pair_mean_and_invalid_not_dropped(self):
        nodes = dict(z=np.array([1., 2., 4.]), indices=np.arange(3), cell=np.array([0, 1, 1]))
        def cell(c, pair, n):
            pairs=np.tile(pair, (n,1));return dict(cell=c, pool_count=n, draws=n, duplicate_draws=0,
                candidates=pairs, eligible=np.ones(n,dtype=bool), pairs=pairs, included=True)
        # Cell0: correct 16 times; cell1: wrong 256 times. Cell-balanced mean=.5.
        groups={k:[cell(0,[0,1],16),cell(1,[1,2],256)] for k in ['local','cross']}
        scored=rd.score(nodes,groups,[3.,1.,2.])
        self.assertEqual(scored['loss'], .5)
        invalid=rd.score(nodes,groups,[np.nan,1.,2.])
        self.assertEqual(invalid['loss'], 1.)
        self.assertEqual(invalid['strata']['local']['main_pairs'],272)
        self.assertEqual(invalid['strata']['local']['invalid_main_pairs'],16)

    def test_empty_low_contrast_support_and_unknown_reference(self):
        gt=np.full((72,108),np.inf,dtype=np.float32)
        nodes=rd.select_nodes(gt,0);groups=rd.make_pairs(nodes,0);scored=rd.score(nodes,groups,[])
        self.assertIsNone(scored['loss']);self.assertFalse(scored['support_passed'])
        gt=np.ones((72,108),dtype=np.float32)
        gt[0,:6]=[np.inf,-np.inf,np.nan,0,-1,2]
        nodes=rd.select_nodes(gt,0)
        self.assertEqual(nodes['counts']['nan'],1);self.assertEqual(nodes['counts']['zero'],1)
        self.assertEqual(nodes['counts']['finite_negative'],1)
        groups=rd.make_pairs(nodes,0);self.assertFalse(rd.score(nodes,groups,np.ones(len(nodes['z'])))['support_passed'])

    def test_one_shuffle_per_node_invalid_mask_and_ties_preserved(self):
        q=np.array([0., 0., 1., 2., np.nan, 4.]);valid=np.array([True,True,True,True,False,True])
        a=rd.shuffled_nodes(q,valid,42,0);b=rd.shuffled_nodes(q,valid,42,0)
        np.testing.assert_array_equal(a,b);self.assertTrue(np.isnan(a[4]))
        np.testing.assert_array_equal(np.sort(a[valid]),np.sort(q[valid]))
        pair=[[0,2],[0,2]];loss=rd.relation_loss(a,[1,2,3,4,5,6],pair,valid)[0]
        self.assertEqual(loss[0],loss[1])

    def test_average_tie_ranks_secondary_constant_unknown(self):
        np.testing.assert_array_equal(rd.average_ranks([3,1,1,2]),[4,1.5,1.5,3])
        self.assertIsNone(rd.spearman(np.zeros(3),np.array([1,2,4]),np.ones(3,dtype=bool))['value'])
        self.assertAlmostEqual(rd.spearman(np.array([1.,.5,.25]),np.array([1,2,4]),np.ones(3,dtype=bool))['value'],1.)

    def test_each_stratum_gate_cannot_hide_behind_good_cross_order(self):
        reports=[]
        for i in range(3):
            report=copy.deepcopy(self.report);report['image_index']=i;reports.append(report)
        self.assertEqual(rd.assess(reports)['status'],'passed_ordinal_screen')
        reports[0]['model']['strata']['local']['loss']=.6
        reports[0]['model']['loss']=.3
        self.assertEqual(rd.assess(reports)['status'],'failed_ordinal_gate')
        reports[0]['model']['support_passed']=False
        self.assertEqual(rd.assess(reports)['status'],'blocked_relative_support')
        with self.assertRaises(ValueError):rd.assess(reports[:2])

    def test_affine_positive_order_invariance_after_sampling(self):
        q=np.array([0.,1.,2.,4.]);z=np.array([4.,3.,2.,1.]);pairs=[[0,1],[0,3],[2,3]]
        np.testing.assert_array_equal(rd.relation_loss(q,z,pairs)[0],rd.relation_loss(3*q+7,z,pairs)[0])

    def test_all_three_prediction_hashes_before_any_label_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(3):
                np.save(root/('%d.npy'%i),np.zeros((2,3),dtype=np.float32))
                rows.append(dict(name=str(i),file='%d.npy'%i,sha256=rd.digest(root/('%d.npy'%i))))
            seal=dict(status='sealed_all_predictions',plan_sha256=rd.PLAN_SHA,
                      source_sha256={'cpu':'accepted'},input_sha256={'input':'accepted'},predictions=rows)
            rd.write_json(root/'seal.json',seal)
            self.assertEqual(len(rd.verify_prediction_seal(root,['0','1','2'],{'cpu':'accepted'},{'input':'accepted'})['predictions']),3)
            (root/'2.npy').write_bytes(b'corrupt third prediction')
            called=[]
            def gated_read():
                rd.verify_prediction_seal(root,['0','1','2'],{'cpu':'accepted'},{'input':'accepted'})
                called.append('GT')
            with self.assertRaises(ValueError):gated_read()
            self.assertFalse(called)
            with self.assertRaises(FileExistsError):rd.write_json(root/'seal.json',{})


class SyntheticEntryAndSealTests(unittest.TestCase):
    def test_synthetic_only_manifest_and_output_refusal(self):
        spec=importlib.util.spec_from_file_location('relative_entry',ROOT/'scripts/check_relative_evaluator_cpu.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifest.json').write_text(json.dumps(dict(kind='real-data')))
            with self.assertRaises(ValueError):entry.synthetic_config(root,{})
            from types import SimpleNamespace
            out=root/'existing';out.mkdir();(out/'keep').write_text('unchanged')
            args=SimpleNamespace(plan=ROOT/'configs/terrace-relative-diagnostic-plan.json',
                output=out,source=ROOT,fixtures=root)
            with self.assertRaises(ValueError):entry.run(args)
            self.assertEqual((out/'keep').read_text(),'unchanged')
            args.plan=root/'wrong.json';args.plan.write_text('{}')
            with self.assertRaises(ValueError):entry.run(args)

    def test_invalid_pair_reference_rejected(self):
        for pairs in [[[-1,0]],[[0,9]]]:
            with self.assertRaises(ValueError):rd.relation_loss([1,2],[1,2],pairs)
        with self.assertRaises(ValueError):rd.make_pairs(tiny_nodes(),3)

    def test_seal_status_source_split_extra_file_and_link_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(3):
                np.save(root/('%d.npy'%i),np.zeros((2,3),dtype=np.float32))
                rows.append(dict(name=str(i),file='%d.npy'%i,sha256=rd.digest(root/('%d.npy'%i))))
            original=dict(status='sealed_all_predictions',plan_sha256=rd.PLAN_SHA,
                source_sha256={'cpu':'accepted'},input_sha256={'input':'accepted'},predictions=rows)
            for kind in ['status','source','names','path','extra','symlink']:
                seal=copy.deepcopy(original)
                if kind=='status':seal['status']='partial'
                elif kind=='source':seal['source_sha256']={}
                elif kind=='names':seal['predictions'][1]['name']='other-view'
                elif kind=='path':seal['predictions'][1]['file']='../1.npy'
                elif kind=='extra':(root/'cache.npy').write_bytes(b'foreign')
                elif kind=='symlink':(root/'extra-link').symlink_to(root/'0.npy')
                (root/'seal.json').write_text(json.dumps(seal))
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    rd.verify_prediction_seal(root,['0','1','2'],{'cpu':'accepted'},{'input':'accepted'})
                if kind=='extra':(root/'cache.npy').unlink()
                if kind=='symlink':(root/'extra-link').unlink()


class CrossSourceMembershipTests(unittest.TestCase):
    def test_duplicate_pair_keeps_different_source_cell_memberships(self):
        nodes=tiny_nodes();groups=rd.make_pairs(nodes,0)
        left=groups['cross'][0]['pairs'];right=groups['cross'][4]['pairs']
        self.assertEqual(set(map(tuple,left.tolist())),set(map(tuple,right.tolist())))
        self.assertEqual(len(left),2)
        scored=rd.score(nodes,groups,np.array([1.,.5,.25]))
        self.assertEqual(scored['strata']['cross']['canonical_unique_eligible_pairs'],2)
        self.assertEqual(scored['strata']['cross']['repeated_eligible_pair_memberships'],2)


class DirectWorkerSchemaTests(unittest.TestCase):
    def test_worker_cannot_bypass_synthetic_schema_before_any_input(self):
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('direct_relative_entry',ROOT/'scripts/check_relative_evaluator_cpu.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);conf=root/'conf.json';conf.write_text(json.dumps(dict(kind='real-data')))
            args=SimpleNamespace(output=root/'result',config=conf,mode='evaluate',predictions=root/'never-open')
            with self.assertRaises(ValueError):entry.worker(args)
            audit=json.loads((root/'result/reference-access.json').read_text())
            self.assertEqual(audit['synthetic_label_attempts'],[])
