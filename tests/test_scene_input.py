"""Approved scene guards, immutable training math and real preparation provenance."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from geosplat_r.scene_input import validate_scene_input,same_colored_point_multiset,SCENE_TRAINS,SCENE_TESTS
from geosplat_r.rgb_path_summary import summarize_paths
from geosplat_r.seed_summary import summarize_pairs
from fetch_llff_subset import read_range


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SceneInputChecks(unittest.TestCase):
    def test_full_response_and_wrong_content_range_rejected_before_read(self):
        class Response:
            status=200
            headers={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):raise AssertionError('Unexpected body read')
        response=Response()
        with patch('urllib.request.urlopen',return_value=response):
            with self.assertRaisesRegex(RuntimeError,'full archive'):read_range('https://example.invalid/a',0,29,100)
            response.status=206;response.headers={'Content-Range':'bytes 0-28/100'}
            with self.assertRaisesRegex(RuntimeError,'Content-Range'):read_range('https://example.invalid/a',0,29,100)

    def test_short_range_body_rejected(self):
        class Response:
            status=206
            headers={'Content-Range':'bytes 0-29/100'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return b'short'
        with patch('urllib.request.urlopen',return_value=Response()):
            with self.assertRaisesRegex(RuntimeError,'length'):read_range('https://example.invalid/a',0,29,100)

    def test_manifest_rejects_test_views_extra_fields_outside_path_and_changed_points(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'train';data.mkdir()
            views=[{'name':n,'image_sha256':str(i),'width':504,'height':378} for i,n in enumerate(SCENE_TRAINS['flower'])]
            (data/'cameras.json').write_text(json.dumps(views))
            np.savez(data/'points.npz',points=np.ones((4,3),np.float32),rgb=np.ones((4,3),np.uint8))
            manifest={'scene':'flower','protocol':'flower-undistorted-v1','split':'train',
                'train_images':[{'name':v['name'],'sha256':v['image_sha256']} for v in views],
                'resolution':[504,378],'cameras_sha256':sha(data/'cameras.json'),
                'points_sha256':sha(data/'points.npz'),'initial_point_count':4}
            path=data/'input-manifest.json';path.write_text(json.dumps(manifest))
            self.assertEqual('flower',validate_scene_input(data,views,path)['scene'])
            with self.assertRaisesRegex(ValueError,'inside'):validate_scene_input(data,views,data.parent/'input-manifest.json')
            bad=copy.deepcopy(views);bad[0]['name']='image000.png'
            with self.assertRaisesRegex(ValueError,'training views'):validate_scene_input(data,bad,path)
            path.write_text(json.dumps({**manifest,'test_images':['image000.png']}))
            with self.assertRaisesRegex(ValueError,'train-only'):validate_scene_input(data,views,path)
            path.write_text(json.dumps(manifest));(data/'points.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):validate_scene_input(data,views,path)

    def test_colmap_point_order_is_irrelevant_but_multiplicity_and_color_matter(self):
        points=np.array([[1.,2.,3.],[4.,5.,6.],[1.,2.,3.]],np.float32)
        colors=np.array([[1,2,3],[4,5,6],[1,2,3]],np.uint8)
        self.assertTrue(same_colored_point_multiset(points,colors,points[[1,0,2]],colors[[1,0,2]]))
        self.assertFalse(same_colored_point_multiset(points,colors,points[:2],colors[:2]))
        changed=colors.copy();changed[0,0]=99
        self.assertFalse(same_colored_point_multiset(points,colors,points,changed))

    def test_new_scene_view_counts_and_local_regressions_preserved(self):
        for scene in ['flower','leaves']:
            rows=[];names={'train':SCENE_TRAINS[scene],'test':SCENE_TESTS[scene]}
            for seed in [42,43,44]:
                row={'seed':seed}
                for arm,score in [('rgb_only',10.),('rgb_ed_zero',9.),('uniform',11.)]:
                    row[arm]={split:{'per_image':[{'image':n,'psnr':score,'ssim':score/20,'lpips_alex':1-score/20} for n in ns],
                        'mean':{'psnr':score,'ssim':score/20,'lpips_alex':1-score/20}} for split,ns in names.items()}
                rows.append(row)
            pairs=[{'seed':r['seed'],'zero':r['rgb_ed_zero'],'uniform':r['uniform']} for r in rows]
            self.assertEqual(3*len(names['test']),len(summarize_pairs(pairs,names)['paired_test_images']))
            result=rows[0]['uniform']['test'];result['per_image'][0]['psnr']=8.
            result['mean']['psnr']=np.mean([v['psnr'] for v in result['per_image']])
            summary=summarize_paths(rows,names);contrast=summary['contrasts']['uniform_minus_rgb_only']
            self.assertEqual(3*len(names['test']),len(contrast['paired_test_images']))
            self.assertEqual(1,len(contrast['degradations']))
            rows[0]['uniform']['test']['per_image'][0]['image']=names['train'][0]
            with self.assertRaisesRegex(ValueError,'view names'):summarize_paths(rows,names)

    def test_frozen_training_loop_render_and_depth_math_unchanged(self):
        reference=subprocess.check_output(['git','show','42e297e:src/geosplat_r/baseline.py'],cwd=ROOT,text=True)
        current=(ROOT/'src/geosplat_r/baseline.py').read_text()
        def loop(source):
            return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.For)
                and isinstance(n.target,ast.Name) and n.target.id=='step')
        # Only erase the three explicitly checked opt-in additions. Default L1
        # must still reconstruct the exact historical loop, not an approximation.
        expected_dispatch=ast.parse("""
if depth_objective_kind=='cauchy' and step>=depth_start:
    depth_term,depth_fraction=cauchy_inverse_depth_loss(rgb_depth[...,3],alpha,
        prior['target'],prior['mask'],prior['normalizer'],depth_cauchy_scale)
else:
    depth_term,depth_fraction=uniform_inverse_depth_loss(rgb_depth[...,3],alpha,
        prior['target'],prior['mask'],prior['normalizer'])
""").body[0]
        expected_order=ast.parse("""
if depth_diagnostics_enabled:
    view_order_digest.update(int(view_index).to_bytes(4,'big'))
""").body[0]
        expected_log=ast.parse("""
if depth_diagnostics_enabled:
    row.update(depth_objective_diagnostics(rgb_depth[...,3],alpha,prior['target'],prior['mask'],
        prior['normalizer'],depth_objective_kind if step>=depth_start else 'l1',weight,depth_cauchy_scale))
    row.update({'depth_objective':depth_term.item(),'depth_requested_objective':depth_objective_kind,
                'view_order_sha256':view_order_digest.hexdigest()})
""").body[0]
        extra_log=ast.parse('depth_diagnostics_enabled and step==depth_start',mode='eval').body
        case=self
        class LegacyDefault(ast.NodeTransformer):
            def visit_If(self,node):
                if ast.dump(node.test)==ast.dump(expected_dispatch.test):
                    case.assertEqual(ast.dump(node),ast.dump(expected_dispatch))
                    return node.orelse
                if ast.dump(node.test)==ast.dump(expected_order.test):
                    case.assertIn(ast.dump(node),[ast.dump(expected_order),ast.dump(expected_log)])
                    return None
                if isinstance(node.test,ast.BoolOp) and isinstance(node.test.op,ast.Or) and len(node.test.values)==4:
                    case.assertEqual(ast.dump(node.test.values[-1]),ast.dump(extra_log))
                    node.test.values.pop()
                return self.generic_visit(node)
        self.assertEqual(ast.dump(loop(reference)),ast.dump(LegacyDefault().visit(loop(current))))
        for name in ['render','load_views']:
            old=next(n for n in ast.parse(reference).body if isinstance(n,ast.FunctionDef) and n.name==name)
            new=next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(old),ast.dump(new))
        old_depth=subprocess.check_output(['git','show','42e297e:src/geosplat_r/depth.py'],cwd=ROOT)
        self.assertEqual(old_depth,(ROOT/'src/geosplat_r/depth.py').read_bytes())


@unittest.skipUnless(os.environ.get('GEOSPLAT_DEV_PREPARATION'),'Explicit completed development preparation not supplied')
class RealDevelopmentPreparation(unittest.TestCase):
    def test_prepared_inputs_splits_and_scene_geometry_have_auditable_hashes(self):
        root=Path(os.environ['GEOSPLAT_DEV_PREPARATION']);total=0
        for scene in ['flower','leaves']:
            raw=json.loads((root/'raw'/scene/'download.json').read_text());total+=raw['downloaded_compressed_bytes']
            self.assertEqual('passed',raw['status'])
            self.assertFalse((root/'raw'/scene/'sparse/0/images.bin').exists())
            data=root/scene/'prepared/train';views=json.loads((data/'cameras.json').read_text())
            manifest=validate_scene_input(data,views,data/'input-manifest.json')
            audit=root/scene/('input-audit-v2' if scene=='flower' else 'input-audit')/'audit.json'
            report=json.loads(audit.read_text())
            self.assertTrue(report['all_point_tracks_training_only'])
            self.assertTrue(report['database_images_training_only'])
            self.assertEqual(manifest['initial_point_count'],report['point_count'])
            self.assertEqual(report['input_manifest_sha256'],sha(data/'input-manifest.json'))
            self.assertEqual(SCENE_TESTS[scene],report['test_names'])
        self.assertEqual(11049045,total)

    def test_train_priors_and_gpu_preflight_use_no_test_assets_or_updates(self):
        root=Path(os.environ['GEOSPLAT_DEV_PREPARATION'])
        for scene in ['flower','leaves']:
            data=root/scene/'prepared/train';priors=root/scene/'priors'
            manifest=json.loads((priors/'manifest.json').read_text())
            self.assertEqual(SCENE_TRAINS[scene],[v['image'] for v in manifest['records']])
            self.assertEqual(sha(data/'cameras.json'),manifest['cameras_sha256'])
            self.assertEqual(sha(data/'points.npz'),manifest['points_sha256'])
            for record in manifest['records']:
                self.assertGreaterEqual(record['alignment']['anchors'],20)
                self.assertGreater(record['alignment']['slope'],0)
                self.assertEqual(sha(priors/record['file']),record['sha256'])
            expected={'cameras.json','points.npz','input-manifest.json',*[f'images/{n}' for n in SCENE_TRAINS[scene]]}
            self.assertEqual(expected,set(json.loads((priors/'training-access.json').read_text())))
            preflight=json.loads((root/scene/'gpu-input-preflight/preflight.json').read_text())
            self.assertEqual(expected,set(preflight['actual_training_asset_reads']))
            self.assertFalse(preflight['training_performed']);self.assertEqual(0,preflight['optimizer_updates'])


if __name__=='__main__':
    unittest.main()
