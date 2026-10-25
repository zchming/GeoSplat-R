"""Prospective gate boundaries and complete failure reporting on synthetic scores."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geosplat_r.robust_pilot_summary import ARMS, METRICS, VIEWS, summarize


def fixture():
    result = {}
    for arm, score in zip(ARMS, [10., 11., 14., 12.]):
        record = {'metric_protocol': {'frozen': True}, 'optimizer_updates': 30000}
        for split, names in VIEWS.items():
            values = dict(psnr=score, ssim=score / 20, lpips_alex=1 - score / 20)
            record[split] = {'mean': values.copy(), 'per_image': [dict(image=n, **values) for n in names]}
        result[arm] = record
    return result


class PilotSummary(unittest.TestCase):
    def test_strict_equality_fails_and_all_six_pairs_are_retained(self):
        result = summarize(fixture())
        self.assertFalse(result['all_three_metrics_passed'])
        self.assertEqual(0., result['prospective_test_direction_gate']['psnr']['margin_over_range'])
        self.assertEqual(6, len(result['all_pairwise_contrasts']))
        self.assertEqual(42, sum(len(s['per_image']) for c in result['all_pairwise_contrasts'].values() for s in c.values()))

    def test_three_metrics_required_and_lpips_direction_is_inverted(self):
        records = fixture()
        for split in VIEWS:
            for row in records['C_cauchy'][split]['per_image']:
                row.update(psnr=16., ssim=.8, lpips_alex=.2)
            records['C_cauchy'][split]['mean'].update(psnr=16., ssim=.8, lpips_alex=.2)
        result = summarize(records);self.assertTrue(result['all_three_metrics_passed'])
        for row in records['C_cauchy']['test']['per_image']:row['lpips_alex']=.9
        records['C_cauchy']['test']['mean']['lpips_alex']=.9
        result = summarize(records);self.assertFalse(result['all_three_metrics_passed'])
        self.assertLess(result['prospective_test_direction_gate']['lpips_alex']['margin_over_range'],0)

    def test_a_bad_view_is_preserved_even_when_average_improves(self):
        records = fixture();rows=records['C_cauchy']['test']['per_image'];rows[0]['psnr']=9.
        records['C_cauchy']['test']['mean']['psnr']=sum(r['psnr'] for r in rows)/4
        result=summarize(records)
        bad=result['all_pairwise_contrasts']['C_cauchy_minus_A1_l1']['test']['per_image'][0]
        self.assertIn('psnr',bad['regressed_metrics']);self.assertEqual(-1.,bad['delta']['psnr'])

    def test_missing_duplicate_nonfinite_protocol_or_incomplete_runs_rejected(self):
        for mutation in ['missing','duplicate','nonfinite','protocol','updates','mean']:
            r=fixture()
            if mutation=='missing':r.pop('A2_l1_repeat')
            elif mutation=='duplicate':r['C_cauchy']['test']['per_image'][1]['image']='image000.png'
            elif mutation=='nonfinite':r['C_cauchy']['test']['mean']['psnr']=float('nan')
            elif mutation=='protocol':r['C_cauchy']['metric_protocol']={'changed':True}
            elif mutation=='updates':r['C_cauchy']['optimizer_updates']=1000
            elif mutation=='mean':r['C_cauchy']['test']['mean']['psnr']=999.
            with self.assertRaises(ValueError,msg=mutation):summarize(r)


if __name__=='__main__':unittest.main()
