"""Read-only completed-run checks; no raw-model decoding or diagnostic execution."""
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('GEOSPLAT_RAW_TRACK_RUN'), 'Explicit completed raw-track run not supplied')
class CompletedRawTrackEvidence(unittest.TestCase):
    def test_input_sources_and_single_frozen_execution(self):
        root = Path(os.environ['GEOSPLAT_RAW_TRACK_RUN'])
        freeze = json.loads((root/'frozen-execution.json').read_text())
        for name, expected in freeze['source_hashes'].items():
            self.assertEqual(expected, hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
        self.assertEqual({'attempt': 1, 'returncode': 0}, json.loads((root/'formal-attempt.json').read_text()))
        for stage in ['input-acceptance', 'diagnostic']:
            status = json.loads((root/stage/'status.json').read_text())
            self.assertEqual('complete', status['status']);self.assertFalse(status['torch_imported'])
            access = json.loads((root/stage/'access.json').read_text())
            self.assertEqual(6, len(access['read_files']));self.assertEqual([], access['denials'])
            self.assertEqual(6, len(access['verified_inputs']))
            resources = json.loads((root/stage/'resources.json').read_text())
            self.assertEqual(0, resources['optimizer_updates']);self.assertFalse(resources['cuda_initialized'])
        allow = json.loads((root/'input-allowlist.json').read_text())
        for row in allow['files']:
            data = Path(row['absolute_path']).read_bytes()
            self.assertEqual(row['bytes'], len(data));self.assertEqual(row['sha256'], hashlib.sha256(data).hexdigest())

    def test_complete_records_grouping_nulls_and_declared_gates(self):
        from scipy.stats import spearmanr
        root = Path(os.environ['GEOSPLAT_RAW_TRACK_RUN'])/'diagnostic'
        rows = json.loads((root/'records.json').read_text())
        inv = json.loads((root/'input-acceptance.json').read_text())
        summary = json.loads((root/'summary.json').read_text())
        self.assertEqual(inv['points']*3, len(rows));self.assertEqual(len(rows), len({(r['point_id'],r['image']) for r in rows}))
        for pid in {r['point_id'] for r in rows}:
            self.assertEqual(1, len({r['fold'] for r in rows if r['point_id']==pid}))
        all_support, all_direction = True, True
        for name, report in summary['per_image'].items():
            own = [r for r in rows if r['image']==name]
            full = [r for r in own if r['track_length']==3]
            paired = [r for r in full if r['score'] is not None and r['residual_px'] is not None]
            unknown = [r for r in own if r['track_length']!=3]
            self.assertTrue(all(r['score'] is None and r['residual_px'] is None for r in unknown))
            rho = float(spearmanr([r['score'] for r in paired],[r['residual_px'] for r in paired]).statistic)
            self.assertAlmostEqual(report['score']['correlation']['rho'], rho, places=14)
            counts = [sum(r['fold']==f for r in paired) for f in range(5)]
            folds = [float(spearmanr([r['score'] for r in paired if r['fold']==f],
                                     [r['residual_px'] for r in paired if r['fold']==f]).statistic) for f in range(5)]
            q1,q2 = np.quantile([r['score'] for r in paired],[1/3,2/3],method='linear')
            lo = [r['residual_px'] for r in paired if r['score']<=q1]
            hi = [r['residual_px'] for r in paired if r['score']>=q2]
            gap = float(np.mean(hi)-np.mean(lo));self.assertAlmostEqual(gap,report['score']['contrast']['high_minus_low_mean'],places=14)
            support = len(paired)>=60 and len(paired)/len(full)>=.5 and min(counts)>=10 and q1<q2 and min(len(lo),len(hi))>=10
            direction = rho>=.2 and sum(x>0 for x in folds)>=4 and gap>0
            self.assertEqual(support,report['observability_passed']);self.assertEqual(direction,report['direction_passed'])
            all_support &= support;all_direction &= direction
        self.assertEqual(all_support,summary['all_observability_passed']);self.assertEqual(all_direction,summary['all_direction_passed'])


if __name__=='__main__':unittest.main()
