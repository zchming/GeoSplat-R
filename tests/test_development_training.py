"""Artifact-backed checks for the predeclared 18-run matrix; no training is executed."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import unittest


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(os.environ.get('GEOSPLAT_DEV_TRAINING'),'Explicit completed 18-run training root not supplied')
class RealDevelopmentTraining(unittest.TestCase):
    def test_all_eighteen_runs_match_budget_inputs_sources_and_paired_initialization(self):
        root=Path(os.environ['GEOSPLAT_DEV_TRAINING'])
        completion=read(root/'training-completion.json');self.assertEqual(18,completion['completed'])
        freeze=read(root/completion.get('active_execution_manifest','frozen-execution.json'))
        ledger=read(root/'training-attempts.json');attempts=[x for x in ledger if x['status']=='passed']
        self.assertEqual(18,len(attempts))
        self.assertTrue(all(x['returncode']==0 for x in attempts))
        self.assertTrue(all(x['status'] in ['passed','interrupted'] for x in ledger))
        for interrupted in [x for x in ledger if x['status']=='interrupted']:
            self.assertTrue(Path(interrupted['run']).is_dir())
            self.assertFalse((Path(interrupted['run'])/'checkpoint.pt').exists())
        self.assertEqual({(s,k,a) for s in ['flower','leaves'] for k in [42,43,44] for a in ['rgb_only','rgb_ed_zero','uniform']},
            {(x['scene'],x['seed'],x['arm']) for x in attempts})
        initials={}
        for item in freeze['commands']:
            run=Path(item['run']);c=read(run/'config.json');r=read(run/'resources.json')
            self.assertEqual(30000,r['optimizer_updates']);self.assertEqual(30000,c['executed_steps'])
            self.assertEqual('passed',r['status']);self.assertEqual('float32',c['precision']);self.assertEqual(1,c['batch_size'])
            self.assertEqual(freeze['source_hashes']['src/geosplat_r/baseline.py'],sha(run/'source/baseline.py'))
            self.assertEqual(freeze['source_hashes']['src/geosplat_r/scene_input.py'],sha(run/'source/scene_input.py'))
            self.assertLess(r['whole_board_sampled_peak_mib'],12282);self.assertIsNone(r['sampler_error'])
            self.assertGreater(r['samples'],0)
            expected={'cameras.json','points.npz','input-manifest.json',*[f"images/{v['name']}" for v in c['train_images']]}
            self.assertEqual(expected,set(read(run/'training-access.json')))
            key=(item['scene'],item['seed']);initials.setdefault(key,set()).add(r['initial_parameters_sha256'])
            if item['arm']=='rgb_only':self.assertFalse(c['depth_loss']);self.assertNotIn('depth_protocol',c)
            else:
                weight=.01 if item['arm']=='uniform' else 0.
                self.assertEqual(weight,c['depth_protocol']['lambda']);self.assertEqual(1000,c['depth_protocol']['start_zero_based_step'])
                self.assertEqual(freeze['source_hashes']['src/geosplat_r/depth.py'],sha(run/'source/depth.py'))
                for row in read(run/'training-curve.json'):
                    self.assertEqual(weight if row['updates']>1000 else 0.,row['depth_weight'])
                    self.assertTrue(all(math.isfinite(v) for v in row.values() if isinstance(v,(float,int))))
        self.assertEqual(6,len(initials));self.assertTrue(all(len(v)==1 for v in initials.values()))


@unittest.skipUnless(os.environ.get('GEOSPLAT_DEV_SUMMARY'),'Explicit completed two-scene evaluation summary not supplied')
class RealDevelopmentEvaluation(unittest.TestCase):
    def test_scene_means_equal_weight_and_every_regression_are_preserved(self):
        root=Path(os.environ['GEOSPLAT_DEV_SUMMARY']);summary=read(root/'summary.json')
        self.assertEqual({'flower','leaves'},set(summary['scenes']))
        for contrast,metrics in summary['macro_contrasts'].items():
            cases=summary['all_paired_test_images'][contrast];self.assertEqual(27,len(cases))
            for metric,value in metrics.items():
                expected=statistics.mean(s['contrasts'][contrast]['splits']['test'][metric]['delta']['mean'] for s in summary['scenes'].values())
                self.assertAlmostEqual(expected,value['equal_scene_mean'],places=12)
            damaged=[]
            for row in cases:
                degraded=[m for m in ['psnr','ssim','lpips_alex'] if (row['candidate'][m]>row['reference'][m] if m=='lpips_alex' else row['candidate'][m]<row['reference'][m])]
                self.assertEqual(degraded,row['degraded_metrics'])
                if degraded:damaged.append(row)
            self.assertEqual(damaged,summary['degradations'][contrast])
        for scene,result in summary['scenes'].items():
            self.assertEqual(5 if scene=='flower' else 4,len(result['test_gt_png_sha256']))
            for row in result['rows']:
                for arm in ['rgb_only','rgb_ed_zero','uniform']:
                    for split in ['train','test']:
                        for metric in ['psnr','ssim','lpips_alex']:
                            self.assertAlmostEqual(statistics.mean(v[metric] for v in row[arm][split]['per_image']),row[arm][split]['mean'][metric],places=10)


if __name__=='__main__':unittest.main()
