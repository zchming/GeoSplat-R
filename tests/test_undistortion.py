"""Camera-ray, remap validity, source isolation and corrected-run checks."""
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r.camera import forward_simple_radial,pinhole_remap
from geosplat_r.baseline import MemorySampler,sha256
SOURCE=os.environ.get('GEOSPLAT_UNDISTORT_SOURCE')
DATA=os.environ.get('GEOSPLAT_UNDISTORT_DATA')
RUN=os.environ.get('GEOSPLAT_UNDISTORT_RUN')
CAMERA={'model_id':2,'width':4032,'height':3024,'params':[3260.5263328805895,2016.,1512.,.019750463822896944]}


class RayGeometry(unittest.TestCase):
    def test_all_rays_match_radial_formula_and_invert_to_pinhole(self):
        K,D,N,x,y,_=pinhole_remap(CAMERA,(504,378))
        gy,gx=np.mgrid[0:378,0:504];pixels=np.stack([gx,gy],-1)
        predicted=forward_simple_radial(pixels,K,N,D[0])
        np.testing.assert_allclose(predicted,np.stack([x,y],-1),atol=3e-5,rtol=0)
        sparse=np.stack([x[::31,::29],y[::31,::29]],-1).reshape(-1,1,2).astype(np.float64)
        restored=cv2.undistortPoints(sparse,K,D,P=N).reshape(-1,2)
        np.testing.assert_allclose(restored,pixels[::31,::29].reshape(-1,2),atol=3e-5,rtol=0)
        # Valid bilinear support for every destination pixel, without a test-derived mask.
        self.assertGreaterEqual(x.min(),0);self.assertLess(x.max(),503)
        self.assertGreaterEqual(y.min(),0);self.assertLess(y.max(),377)
        marker=np.full((378,504),255,np.uint8)
        self.assertTrue(np.all(cv2.remap(marker,x,y,cv2.INTER_LINEAR)==255))

    def test_wrong_camera_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'SIMPLE_RADIAL'):
            pinhole_remap({**CAMERA,'model_id':1},(504,378))

    def test_memory_sampler_uses_independent_gpu_clock_and_utilization(self):
        class Process:
            stdout=io.StringIO('2026/10/19 08:00:00.000, 1000, 80\n2026/10/19 08:00:01.000, 1200, 100\n')
            def terminate(self):pass
        with tempfile.TemporaryDirectory() as d,patch('subprocess.Popen',return_value=Process()):
            sampler=MemorySampler(Path(d)/'gpu.csv');sampler.start();result=sampler.stop()
        self.assertEqual(1200,result['whole_board_sampled_peak_mib'])
        self.assertEqual(1.,result['gpu_timestamp_sampling_span_seconds'])
        self.assertEqual(90.,result['gpu_utilization_mean_percent'])
        self.assertIsNone(result['sampler_error'])


@unittest.skipUnless(SOURCE and DATA,'Explicit corrected source/prepared data not supplied')
class RealUndistortedData(unittest.TestCase):
    def test_shared_intrinsics_frozen_splits_and_training_only_database(self):
        source=Path(SOURCE);data=Path(DATA)
        protocol=json.loads((source/'protocol.json').read_text())
        self.assertEqual('fern-undistorted-v1',protocol['name'])
        self.assertEqual(0,protocol['padding_pixels']);self.assertFalse(protocol['evaluation_mask'])
        self.assertLess(protocol['analytic_forward_map_max_abs_error_px'],5e-5)
        for split,expected in [('train',[1,10,19]),('test',[0,8,16])]:
            views=json.loads((data/split/'cameras.json').read_text())
            self.assertEqual(expected,[r['source_index'] for r in views])
            for r in views:
                np.testing.assert_array_equal(r['K'],protocol['new_K'])
                self.assertEqual(sha256(data/split/'images'/r['name']),r['image_sha256'])
                if os.environ.get('GEOSPLAT_FERN_DATA'):
                    old=json.loads((Path(os.environ['GEOSPLAT_FERN_DATA'])/split/'cameras.json').read_text())
                    legacy=next(x for x in old if x['name']==r['name'])
                    np.testing.assert_array_equal(r['c2w'],legacy['c2w'])
        with sqlite3.connect(f'file:{data}/train/database.db?mode=ro',uri=True) as db:
            names={x[0] for x in db.execute('SELECT name FROM images')}
        self.assertEqual({'image001.png','image010.png','image019.png'},names)

    def test_derived_inputs_are_content_hashed_and_originals_unchanged(self):
        source=Path(SOURCE)
        manifest=json.loads((source/'download.json').read_text())
        self.assertEqual(7,len(manifest['files']))
        for r in manifest['files']:
            self.assertEqual(sha256(source/r['local']),r['sha256'])
            if 'source_sha256' in r:self.assertNotEqual(r['sha256'],r['source_sha256'])


@unittest.skipUnless(RUN,'Completed corrected training run not supplied')
class RealCorrectedTraining(unittest.TestCase):
    def test_run_is_complete_isolated_and_records_all_clocks(self):
        root=Path(RUN)
        r=json.loads((root/'resources.json').read_text())
        config=json.loads((root/'config.json').read_text())
        self.assertEqual('passed',r['status']);self.assertEqual(30000,r['optimizer_updates'])
        for field in ['training_perf_counter_seconds','training_wall_clock_seconds','cuda_stream_elapsed_seconds','gpu_timestamp_sampling_span_seconds']:
            self.assertGreater(r[field],0)
        self.assertLess(r['whole_board_sampled_peak_mib'],12282)
        self.assertEqual('3 CUDA float32 RGBs',config['image_cache'])
        self.assertEqual(sha256(root/'source/baseline.py'),config['training_implementation_sha256'])
        self.assertEqual({'cameras.json','points.npz','images/image001.png','images/image010.png','images/image019.png'},
                         set(json.loads((root/'training-access.json').read_text())))


if __name__=='__main__':unittest.main()
