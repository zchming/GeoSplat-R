"""Guard descriptive seed statistics against pooling views or hiding degradation."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r.seed_summary import describe,summarize_pairs


def fixture():
    pairs=[]
    for seed,a,b in [(42,10.,11.),(43,12.,11.),(44,14.,15.)]:
        p={'seed':seed}
        for arm,value in [('zero',a),('uniform',b)]:
            result={'per_image':[{'image':n,'psnr':value,'ssim':value/20,'lpips_alex':1-value/20}
                for n in ['image000.png','image008.png','image016.png']],
                'mean':{'psnr':value,'ssim':value/20,'lpips_alex':1-value/20}}
            p[arm]={'train':copy.deepcopy(result),'test':copy.deepcopy(result)}
        pairs.append(p)
    return pairs


class SeedStatistics(unittest.TestCase):
    def test_paired_sample_variation_and_degradations_are_preserved(self):
        r=summarize_pairs(fixture());m=r['aggregates']['test']['psnr']
        self.assertAlmostEqual(1/3,m['delta_uniform_minus_zero']['mean'])
        self.assertAlmostEqual(math.sqrt(4/3),m['delta_uniform_minus_zero']['sample_sd'])
        self.assertEqual(3,m['zero']['n_seeds']);self.assertEqual(2,m['improved_seeds']);self.assertEqual(1,m['degraded_seeds'])
        self.assertEqual(3,len(r['degradations']))
        self.assertTrue(all(x['seed']==43 for x in r['degradations']))
        self.assertEqual(2,r['aggregates']['test']['lpips_alex']['improved_seeds'])

    def test_missing_or_duplicated_seed_cannot_be_silently_aggregated(self):
        p=fixture()
        for invalid in [p[:2],[p[0],p[0],p[2]]]:
            with self.assertRaisesRegex(ValueError,'Exactly one pair'):summarize_pairs(invalid)

    def test_wrong_test_split_or_reported_mean_is_rejected(self):
        p=fixture();p[0]['uniform']['test']['per_image'][0]['image']='image001.png'
        with self.assertRaisesRegex(ValueError,'test-image'):summarize_pairs(p)
        p=fixture();p[1]['zero']['test']['mean']['psnr']=99
        with self.assertRaisesRegex(ValueError,'mean metrics'):summarize_pairs(p)

    def test_nonfinite_or_single_value_is_not_a_sample_sd(self):
        for values in [[1.],[1.,float('nan')],[float('inf'),2.]]:
            with self.assertRaises(ValueError):describe(values)


@unittest.skipUnless(os.environ.get('GEOSPLAT_SEED_MANIFEST'),'Explicit completed seed-pair manifest not supplied')
class RealSeedSummary(unittest.TestCase):
    def test_all_six_run_provenance_and_same_gt_are_validated(self):
        spec=importlib.util.spec_from_file_location('summary_entry',ROOT/'scripts/summarize_depth_seeds.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        manifest=json.loads(Path(os.environ['GEOSPLAT_SEED_MANIFEST']).read_text())
        result=module.collect(manifest)
        self.assertEqual([42,43,44],[p['seed'] for p in result['pairs']])
        self.assertEqual(9,len(result['paired_test_images']))
        self.assertEqual(3,result['aggregates']['test']['psnr']['zero']['n_seeds'])
        self.assertEqual(3,len(result['test_gt_png_sha256']))


if __name__=='__main__':unittest.main()
