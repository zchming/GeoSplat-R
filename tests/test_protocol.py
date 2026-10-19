"""Checks for leakage, camera conventions, immutable inputs and baseline provenance.

Set GEOSPLAT_FERN_DATA, GEOSPLAT_FERN_RAW, GEOSPLAT_RUN for real artifact checks.
Missing optional artifacts are explicitly skipped, never treated as a pass.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r.baseline import load_views,sha256
spec=importlib.util.spec_from_file_location('fern_fetch',ROOT/'scripts/fetch_fern.py')
fetch=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetch)
DATA=os.environ.get('GEOSPLAT_FERN_DATA')
RAW=os.environ.get('GEOSPLAT_FERN_RAW')
RUN=os.environ.get('GEOSPLAT_RUN')


class InputGuards(unittest.TestCase):
    def test_server_cannot_fall_back_to_full_download(self):
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):raise AssertionError('A full archive must not be read')
        with patch('urllib.request.urlopen',return_value=Response()):
            with self.assertRaisesRegex(RuntimeError,'full archive'):
                fetch.fetch_range('https://example.invalid/file.zip',0,29)

    def test_image_hash_mismatch_rejected_before_gpu_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'images').mkdir();(p/'images/a.png').write_bytes(b'changed')
            (p/'cameras.json').write_text(json.dumps([{'name':'a.png','image_sha256':'wrong'}]))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_views(p)

    def test_image_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'cameras.json').write_text(json.dumps([{'name':'../test.png'}]))
            with self.assertRaisesRegex(ValueError,'basenames'):load_views(p)


@unittest.skipUnless(DATA and RAW,'Explicit fern prepared/raw locations not provided')
class RealSplit(unittest.TestCase):
    def test_frozen_split_and_hashes(self):
        data=Path(DATA);raw=Path(RAW)
        downloaded=json.loads((raw/'download.json').read_text())
        train=json.loads((data/'train/cameras.json').read_text())
        test=json.loads((data/'test/cameras.json').read_text())
        remaining=[i for i in range(20) if i%8!=0]
        chosen=[remaining[i] for i in np.rint(np.linspace(0,len(remaining)-1,3)).astype(int)]
        self.assertEqual(chosen,[r['source_index'] for r in train])
        self.assertEqual([0,8,16],[r['source_index'] for r in test])
        self.assertTrue(set(r['name'] for r in train).isdisjoint(r['name'] for r in test))
        for split,records in [('train',train),('test',test)]:
            self.assertEqual(sorted(x.name for x in (data/split/'images').iterdir()),sorted(r['name'] for r in records))
            for r in records:self.assertEqual(sha256(data/split/'images'/r['name']),r['image_sha256'])
        self.assertFalse(any('points3D' in f['archive_member'] or 'database' in f['archive_member'] for f in downloaded['files']))
        self.assertEqual(7,len(downloaded['files']))

    def test_camera_projection_invariant_under_train_only_normalization(self):
        data=Path(DATA);raw=Path(RAW)
        poses=np.load(raw/'poses_bounds.npy')[:,:15].reshape(-1,3,5)
        train=json.loads((data/'train/cameras.json').read_text())
        original_centers=poses[[r['source_index'] for r in train],:,3]
        center=original_centers.mean(0);scale=1/np.linalg.norm(original_centers-center,axis=1).max()
        summary=json.loads((data/'preparation.json').read_text())
        np.testing.assert_allclose(center,summary['normalization']['center'])
        self.assertAlmostEqual(scale,summary['normalization']['scale'])
        normalized=np.load(data/'train/points.npz')['points'].astype(float)
        original=normalized/scale+center
        for split in ['train','test']:
            for r in json.loads((data/split/'cameras.json').read_text()):
                p=poses[r['source_index']]
                old=np.eye(4);old[:3,:3]=p[:,[1,0,2]]*np.array([1,1,-1]);old[:3,3]=p[:,3]
                new=np.array(r['c2w']);K=np.array(r['K'])
                old_xyz=(original-old[:3,3])@old[:3,:3]
                new_xyz=(normalized-new[:3,3])@new[:3,:3]
                np.testing.assert_allclose(new_xyz,old_xyz*scale,rtol=1e-6,atol=1e-6)
                a=old_xyz@K.T;b=new_xyz@K.T
                np.testing.assert_allclose(a[:,:2]/a[:,2:],b[:,:2]/b[:,2:],rtol=1e-5,atol=1e-4)
                self.assertEqual([504,378],[r['width'],r['height']])
                self.assertAlmostEqual(np.linalg.det(new[:3,:3]),1.,places=6)


@unittest.skipUnless(RUN,'Explicit completed training location not provided')
class RealTraining(unittest.TestCase):
    def test_no_test_asset_access(self):
        p=Path(RUN)
        accessed=json.loads((p/'training-access.json').read_text())
        self.assertEqual(set(accessed),{'cameras.json','points.npz','images/image001.png',
            'images/image010.png','images/image019.png'})
        resources=json.loads((p/'resources.json').read_text())
        self.assertEqual('passed',resources['status'])
        self.assertEqual(30000,resources['optimizer_updates'])
        self.assertGreater(resources['samples'],0)
        self.assertLess(resources['whole_board_sampled_peak_mib'],12282)
        config=json.loads((p/'config.json').read_text())
        source=Path(os.environ['GEOSPLAT_TRAINING_SOURCE']) if os.environ.get('GEOSPLAT_TRAINING_SOURCE') else p/'source/baseline.py'
        if not source.exists():source=ROOT/'src/geosplat_r/baseline.py'
        self.assertEqual(sha256(source),config['training_implementation_sha256'])
        self.assertFalse(config['depth_loss'])
        self.assertFalse(config['camera_optimization'])


if __name__=='__main__':unittest.main()
