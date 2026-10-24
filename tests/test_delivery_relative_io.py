"""Synthetic-only delivery scene admission/sealing and legacy regression boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r import delivery_relative_io as io


def contexts():
    paths = {n:io.rd.digest(ROOT/'src/geosplat_r'/n) for n in ['relative_io.py','relative_diagnostics.py','delivery_relative_io.py']}
    return io.load_contexts(ROOT/'configs/delivery-area-relative-adapter-plan.json',
        ROOT/'configs/delivery-area-relative-inputs.json',ROOT/'configs/terrace-relative-diagnostic-plan.json',
        ROOT/'configs/second-scene-ordinal-review.json',paths)


def fixture(root):
    _,cfg = contexts()
    cfg['native_wh']=[36,36]
    cfg['work_wh']=[6,6]
    pred = root/'pred'
    pred.mkdir()
    for i in range(3):
        np.save(pred/('%d.npy'%i),np.ones((6,6),dtype=np.float32))
    io.seal_predictions(pred,cfg)
    return pred,cfg


class DeliveryIOTests(unittest.TestCase):
    def test_context_roles_and_separate_identity(self):
        p,e=contexts()
        self.assertEqual(p['names'],io.NAMES)
        self.assertNotEqual(p['plan_sha256'],io.rd.PLAN_SHA)
        self.assertNotIn('labels',p)
        self.assertNotIn('camera_records',e)
        self.assertNotIn('model',e)
        self.assertEqual(len(p['inputs']),4)
        self.assertEqual(p['model'],json.loads((ROOT/'configs/terrace-relative-diagnostic-plan.json').read_text())['prediction']['model'])

    def test_core_and_legacy_IO_byte_identity(self):
        plan=json.loads((ROOT/'configs/delivery-area-relative-adapter-plan.json').read_text())
        self.assertEqual(io.rd.digest(ROOT/'src/geosplat_r/relative_diagnostics.py'),plan['numerical_core_sha256'])
        self.assertEqual(io.rd.digest(ROOT/'src/geosplat_r/relative_io.py'),plan['legacy_IO_sha256'])

    def test_plan_manifest_base_review_drift_rejected(self):
        paths=[ROOT/'configs'/n for n in ['delivery-area-relative-adapter-plan.json','delivery-area-relative-inputs.json','terrace-relative-diagnostic-plan.json','second-scene-ordinal-review.json']]
        with tempfile.TemporaryDirectory() as tmp:
            wrong=Path(tmp)/'wrong.json';wrong.write_text('{}')
            for i in range(4):
                changed=paths.copy();changed[i]=wrong
                with self.assertRaises(ValueError):io.load_contexts(*changed,{})

    def test_wrong_numerical_source_rejected(self):
        with self.assertRaises(ValueError):
            io.load_contexts(ROOT/'configs/delivery-area-relative-adapter-plan.json',ROOT/'configs/delivery-area-relative-inputs.json',ROOT/'configs/terrace-relative-diagnostic-plan.json',ROOT/'configs/second-scene-ordinal-review.json',{'relative_diagnostics.py':'wrong'})

    def test_exact_camera_records_and_four_asset_inventory(self):
        p,_=contexts()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in io.NAMES:
                f=root/'rgb'/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'synthetic RGB')
            (root/'cameras.json').write_text(json.dumps(p['camera_records']))
            p['inputs']={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=io.rd.digest(f)) for f in root.rglob('*') if f.is_file()}
            self.assertEqual(len(io.admit_predictor(root,p)),3)
            bad=copy.deepcopy(p);bad['camera_records'][2]['tvec'][0]+=1
            with self.assertRaises(ValueError):io.admit_predictor(root,bad)
            (root/'labels').mkdir();(root/'labels/GT').write_bytes(b'forbidden')
            with self.assertRaises(ValueError):io.admit_predictor(root,p)

    def test_wrong_scene_admission(self):
        p,_=contexts();p['names'][0]='terrace.JPG'
        with self.assertRaises(ValueError):io.admit_predictor('/not-read',p)

    def test_seal_complete_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred,cfg=fixture(Path(tmp));self.assertEqual(len(io.sealed_predictions(pred,cfg)),3)
            before=(pred/'seal.json').read_bytes()
            with self.assertRaises(ValueError):io.seal_predictions(pred,cfg)
            self.assertEqual((pred/'seal.json').read_bytes(),before)

    def test_incomplete_outputs_never_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred=Path(tmp);_,cfg=contexts();np.save(pred/'0.npy',np.ones((6,6),dtype=np.float32))
            with self.assertRaises(ValueError):io.seal_predictions(pred,cfg)
            self.assertFalse((pred/'seal.json').exists())

    def test_corrupt_third_has_zero_GT_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred,cfg=fixture(root);(pred/'2.npy').write_bytes(b'corrupt');access=[]
            with patch.object(io.legacy,'native_reference') as loader,self.assertRaises(ValueError):
                io.evaluate_sealed(pred,root/'labels',cfg,root/'out',access)
            loader.assert_not_called();self.assertEqual(access,[])

    def test_digest_consistent_wrong_dtype_third_has_zero_GT_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred,cfg=fixture(root);np.save(pred/'2.npy',np.ones((6,6),dtype=np.float64))
            seal=json.loads((pred/'seal.json').read_text());seal['predictions'][2]['sha256']=io.rd.digest(pred/'2.npy');(pred/'seal.json').write_text(json.dumps(seal))
            with patch.object(io.legacy,'native_reference') as loader,self.assertRaises(ValueError):io.evaluate_sealed(pred,root/'labels',cfg,root/'out',[])
            loader.assert_not_called()

    def test_wrong_scene_and_every_provenance_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred,cfg=fixture(Path(tmp));path=pred/'seal.json';original=json.loads(path.read_text())
            for key in ['protocol','plan_sha256','input_manifest_sha256','base_plan_sha256','source_sha256','input_sha256']:
                bad=copy.deepcopy(original);bad[key]='wrong';path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):io.sealed_predictions(pred,cfg)

    def test_terrace_verifier_rejects_delivery_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred,cfg=fixture(Path(tmp))
            with self.assertRaises(ValueError):io.rd.verify_prediction_seal(pred,cfg['names'],cfg['source_sha256'],cfg['input_sha256'])

    def test_paths_symlinks_and_extra_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred,cfg=fixture(root);(pred/'cache').write_bytes(b'x')
            with self.assertRaises(ValueError):io.sealed_predictions(pred,cfg)
            (pred/'cache').unlink();(root/'alias').symlink_to(pred,target_is_directory=True)
            with self.assertRaises(ValueError):io.sealed_predictions(root/'alias',cfg)
            seal=json.loads((pred/'seal.json').read_text());seal['predictions'][2]['file']='../2.npy';(pred/'seal.json').write_text(json.dumps(seal))
            with self.assertRaises(ValueError):io.sealed_predictions(pred,cfg)

    def test_labels_mapping_before_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred,cfg=fixture(root);cfg['labels'][2]['image']='wrong.JPG'
            with patch.object(io.legacy,'native_reference') as loader,self.assertRaises(ValueError):io.evaluate_sealed(pred,root/'labels',cfg,root/'out',[])
            loader.assert_not_called()

    def test_symlink_seal_refused_before_json_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred,cfg=fixture(root)
            saved=root/'outside.json';saved.write_bytes((pred/'seal.json').read_bytes())
            (pred/'seal.json').unlink();(pred/'seal.json').symlink_to(saved)
            with patch.object(io.json,'loads') as parser,self.assertRaises(ValueError):io.sealed_predictions(pred,cfg)
            parser.assert_not_called()

    def test_evaluation_output_cannot_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred,cfg=fixture(root)
            for output in [pred/'child',root/'labels/child',root]:
                with patch.object(io.legacy,'native_reference') as loader,self.assertRaises(ValueError):io.evaluate_sealed(pred,root/'labels',cfg,output,[])
                loader.assert_not_called()

    def test_CPU_entry_preserves_existing_output(self):
        spec=importlib.util.spec_from_file_location('delivery_entry',ROOT/'scripts/check_delivery_adapter_cpu.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'keep').write_text('keep')
            args=SimpleNamespace(source=ROOT,data=Path('/unmounted/data'),output=root,
                plan=ROOT/'configs/delivery-area-relative-adapter-plan.json',input_manifest=ROOT/'configs/delivery-area-relative-inputs.json',
                base_plan=ROOT/'configs/terrace-relative-diagnostic-plan.json',review_plan=ROOT/'configs/second-scene-ordinal-review.json')
            with self.assertRaises(ValueError):entry.run(args)
            self.assertEqual((root/'keep').read_text(),'keep')

    def test_synthetic_raw_IO_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred,cfg=fixture(root);labels=root/'labels';labels.mkdir()
            for row in cfg['labels']:
                p=labels/row['image'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(np.ones((36,36),dtype='<f4').tobytes());row.update(bytes=p.stat().st_size,sha256=io.rd.digest(p))
            access=[];result=io.evaluate_sealed(pred,labels,cfg,root/'out',access)
            self.assertEqual(result['status'],'blocked_relative_support');self.assertEqual(access,io.NAMES)
            self.assertTrue((root/'out/image-2-evaluator-only-pairs.npz').exists())
            with patch.object(io.legacy,'native_reference') as loader,self.assertRaises(FileExistsError):io.evaluate_sealed(pred,labels,cfg,root/'out',[])
            loader.assert_not_called()


if __name__=='__main__':unittest.main()
