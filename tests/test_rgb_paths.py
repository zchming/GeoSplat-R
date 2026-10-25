"""Guard matched path deltas, view identities, and preservation of regressions."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from geosplat_r.rgb_path_summary import summarize_paths


def fixture():
    rows=[]
    for seed,a,b,c in [(42,10.,11.,12.),(43,12.,11.,10.),(44,14.,15.,16.)]:
        row={'seed':seed}
        for arm,value in [('rgb_only',a),('rgb_ed_zero',b),('uniform',c)]:
            row[arm]={}
            for split,names in [('train',['image001.png','image010.png','image019.png']),('test',['image000.png','image008.png','image016.png'])]:
                row[arm][split]={'per_image':[{'image':n,'psnr':value,'ssim':value/20,'lpips_alex':1-value/20} for n in names],
                    'mean':{'psnr':value,'ssim':value/20,'lpips_alex':1-value/20}}
        rows.append(row)
    return rows


class PathStatistics(unittest.TestCase):
    def test_contrast_direction_sample_sd_and_negative_seed_preserved(self):
        result=summarize_paths(fixture());p=result['contrasts']['rgb_ed_zero_minus_rgb_only']
        metric=p['splits']['test']['psnr']
        self.assertAlmostEqual(1/3,metric['delta']['mean'])
        self.assertAlmostEqual(math.sqrt(4/3),metric['delta']['sample_sd'])
        self.assertEqual(1,metric['degraded_seeds']);self.assertEqual(2,metric['improved_seeds'])
        self.assertEqual(3,len(p['degradations']));self.assertTrue(all(r['seed']==43 for r in p['degradations']))
        self.assertEqual(2,p['splits']['test']['lpips_alex']['improved_seeds'])
        self.assertEqual(3,result['arm_aggregates']['test']['rgb_only']['psnr']['n_seeds'])

    def test_one_view_ssim_regression_is_not_hidden_by_improved_mean(self):
        rows=fixture();result=rows[0]['rgb_ed_zero']['test']
        result['per_image'][0]['ssim']=.49;result['mean']['ssim']=(.49+.55+.55)/3
        summary=summarize_paths(rows);damaged=summary['contrasts']['rgb_ed_zero_minus_rgb_only']['degradations']
        row=next(r for r in damaged if r['seed']==42 and r['image']=='image000.png')
        self.assertEqual(['ssim'],row['degraded_metrics'])

    def test_missing_seed_or_wrong_train_view_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'seed42/43/44'):summarize_paths(fixture()[:2])
        rows=fixture();rows[0]['rgb_only']['train']['per_image'][0]['image']='image000.png'
        with self.assertRaisesRegex(ValueError,'view names'):summarize_paths(rows)

    def test_nonfinite_or_false_mean_is_rejected(self):
        rows=fixture();rows[1]['rgb_only']['test']['mean']['psnr']=0
        with self.assertRaisesRegex(ValueError,'mean mismatch'):summarize_paths(rows)
        rows=fixture();rows[0]['uniform']['test']['per_image'][0]['psnr']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):summarize_paths(rows)


@unittest.skipUnless(os.environ.get('GEOSPLAT_RGB_PATH_MANIFEST'),'Explicit completed RGB-only path manifest not supplied')
class RealPathEvidence(unittest.TestCase):
    def test_all_nine_runs_and_matched_initialization_have_valid_provenance(self):
        spec=importlib.util.spec_from_file_location('rgb_path_entry',ROOT/'scripts/summarize_rgb_paths.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        result=module.collect_paths(json.loads(Path(os.environ['GEOSPLAT_RGB_PATH_MANIFEST']).read_text()))
        self.assertEqual([42,43,44],[r['seed'] for r in result['rows']])
        self.assertEqual(3,len(result['test_gt_png_sha256']))
        self.assertEqual(3,len(result['contrasts']))
        for contrast in result['contrasts'].values():self.assertEqual(9,len(contrast['paired_test_images']))


if __name__=='__main__':unittest.main()
