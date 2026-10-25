"""Independent binary-camera conventions and the completed diagnostic evidence."""
import importlib.util
import io
import json
import os
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_fern import parse_images
AUDIT=os.environ.get('GEOSPLAT_CAMERA_AUDIT')
REFERENCE=os.environ.get('GEOSPLAT_REFERENCE')
ORIGINAL=os.environ.get('GEOSPLAT_RUN')


class CameraBinary(unittest.TestCase):
    def test_quaternion_order_inverse_translation_and_skipped_observations(self):
        # W2C: +90deg about Z, translation [1,2,3]; C2W center [-2,1,-3].
        q=np.sqrt(.5)
        payload=struct.pack('<Q',1)+struct.pack('<i7di',7,q,0,0,q,1,2,3,1)
        payload+=b'IMG_0042.JPG\0'+struct.pack('<Q',1)+struct.pack('<ddq',91,17,9999)
        result=parse_images(payload)
        self.assertEqual('IMG_0042.JPG',result[0]['name'])
        self.assertEqual({'id','name','camera_id','c2w'},set(result[0]))
        np.testing.assert_allclose(result[0]['c2w'][:3,3],[-2,1,-3],atol=1e-12)
        np.testing.assert_allclose(result[0]['c2w'][:3,:3],[[0,1,0],[-1,0,0],[0,0,1]],atol=1e-12)

    def test_unterminated_name_rejected(self):
        payload=struct.pack('<Q',1)+struct.pack('<i7di',1,1,0,0,0,0,0,0,1)+b'not-terminated'
        with self.assertRaisesRegex(ValueError,'Unterminated'):parse_images(payload)


@unittest.skipUnless(AUDIT,'Explicit camera audit result not supplied')
class RealCameraAudit(unittest.TestCase):
    def test_official_mapping_and_radial_distortion_are_reported(self):
        j=json.loads(Path(AUDIT).read_text())
        self.assertEqual(20,len(j['mapping']))
        self.assertLess(j['mapping_max_abs_error'],1e-8)
        self.assertLess(j['intrinsics_max_abs_error'],1e-8)
        self.assertEqual(2,j['camera_source']['model_id'])
        d=j['camera_source']['distortion']['p50_p95_max_shift_px']
        self.assertGreater(d[-1],3)
        self.assertLess(d[-1],4)
        self.assertEqual(373,j['initial_point_count'])
        rows={r['image']:r for r in j['projection_diagnostics']}
        self.assertGreater(rows['image016.png']['nearest_training_camera_distance_normalized'],
                           rows['image000.png']['nearest_training_camera_distance_normalized'])


@unittest.skipUnless(REFERENCE and ORIGINAL,'Completed upstream/original runs not supplied')
class RealReference(unittest.TestCase):
    def test_identical_inputs_and_initialization_without_test_reads(self):
        reference=Path(REFERENCE);original=Path(ORIGINAL)
        a=json.loads((reference/'config.json').read_text());b=json.loads((original/'config.json').read_text())
        self.assertEqual(a['points_sha256'],b['points_sha256'])
        self.assertEqual(a['cameras_sha256'],b['cameras_sha256'])
        x=json.loads((reference/'resources.json').read_text());y=json.loads((original/'resources.json').read_text())
        self.assertEqual('passed',x['status'])
        self.assertEqual(30000,x['optimizer_updates'])
        self.assertEqual(x['initial_parameters_sha256'],y['initial_parameters_sha256'])
        expected={'cameras.json','points.npz','images/image001.png','images/image010.png','images/image019.png'}
        self.assertEqual(expected,set(json.loads((reference/'training-access.json').read_text())))
        self.assertLess(x['whole_board_sampled_peak_mib'],12282)


if __name__=='__main__':unittest.main()
