"""CPU-only admission/sealing/raw-label and preflight-command checks; no real assets."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r import relative_io as io
spec=importlib.util.spec_from_file_location('real_relative_entry',ROOT/'scripts/run_relative_diagnostic.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
PLAN=json.loads((ROOT/'configs/terrace-relative-diagnostic-plan.json').read_text())


def predictions(root, shape=(2,3), dtype=np.float32):
    root.mkdir();rows=[]
    for i in range(3):
        p=root/('%d.npy'%i);np.save(p,np.zeros(shape,dtype=dtype))
        rows.append(dict(name='image'+str(i),file=p.name,sha256=io.rd.digest(p)))
    config=dict(names=['image0','image1','image2'],work_wh=[shape[1],shape[0]],
        native_wh=[shape[1]*6,shape[0]*6],source_sha256={'code':'fixed'},input_sha256={'RGB':'fixed'},
        labels=[dict(image='image'+str(i),bytes=0,sha256='unused') for i in range(3)])
    io.rd.write_json(root/'seal.json',dict(status='sealed_all_predictions',plan_sha256=io.rd.PLAN_SHA,
        source_sha256=config['source_sha256'],input_sha256=config['input_sha256'],predictions=rows))
    return config


class RelativeIOTests(unittest.TestCase):
    def test_predictor_config_excludes_labels_and_evaluator_excludes_weights(self):
        p,e=io.configs(PLAN,{'adapter':'fixed'})
        self.assertNotIn('labels',p);self.assertEqual(len(p['inputs']),4)
        self.assertNotIn('model',e);self.assertNotIn('inputs',e);self.assertEqual(len(e['labels']),3)
        io.validate_context(p);io.validate_context(e)
        wrong=copy.deepcopy(p);wrong['names'][0]='reserved.JPG'
        with self.assertRaises(ValueError):io.validate_context(wrong)

    def test_four_input_inventory_hash_and_camera_mapping(self):
        p,_=io.configs(PLAN,{})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i,n in enumerate(p['names']):
                target=root/'rgb'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'synthetic RGB')
                rows.append(dict(name=n,image_id=i,camera={'width':6048,'height':4032}))
            (root/'cameras.json').write_text(json.dumps(rows))
            p['inputs']={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=io.rd.digest(f)) for f in root.rglob('*') if f.is_file()}
            self.assertEqual(len(io.admit_predictor(root,p)),3)
            (root/'GT.npy').write_bytes(b'forbidden')
            with self.assertRaises(ValueError):io.admit_predictor(root,p)
            (root/'GT.npy').unlink();(root/'cameras.json').write_text('changed')
            with self.assertRaises(ValueError):io.admit_predictor(root,p)

    def test_corrupt_third_prediction_refused_before_any_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=predictions(root/'pred')
            (root/'pred/2.npy').write_bytes(b'corrupt')
            with patch.object(io,'native_reference') as loader,self.assertRaises(ValueError):
                io.evaluate_sealed(root/'pred',root/'labels',config,root,[])
            loader.assert_not_called()

    def test_third_schema_mismatch_before_any_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=predictions(root/'pred')
            np.save(root/'pred/2.npy',np.ones((2,3),dtype=np.float64))
            seal=json.loads((root/'pred/seal.json').read_text());seal['predictions'][2]['sha256']=io.rd.digest(root/'pred/2.npy')
            (root/'pred/seal.json').write_text(json.dumps(seal))
            with patch.object(io,'native_reference') as loader,self.assertRaises(ValueError):
                io.evaluate_sealed(root/'pred',root/'labels',config,root,[])
            loader.assert_not_called()

    def test_all_label_mapping_metadata_checked_before_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=predictions(root/'pred');config['labels']=config['labels'][:2]
            with patch.object(io,'native_reference') as loader,self.assertRaises(ValueError):
                io.evaluate_sealed(root/'pred',root/'labels',config,root,[])
            loader.assert_not_called()

    def test_native_raw_float32_JPG_mmap_readonly_and_length_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.JPG';data=np.array([[1,2,np.inf],[3,4,5]],dtype='<f4');path.write_bytes(data.tobytes())
            spec=dict(bytes=24,sha256=io.rd.digest(path));gt=io.native_reference(path,spec,[3,2])
            np.testing.assert_array_equal(gt,data);self.assertFalse(gt.flags.writeable)
            for change in [dict(bytes=20,sha256=spec['sha256']),dict(bytes=24,sha256='wrong')]:
                with self.assertRaises(ValueError):io.native_reference(path,change,[3,2])
            link=Path(tmp)/'link.JPG';link.symlink_to(path)
            with self.assertRaises(ValueError):io.native_reference(link,spec,[3,2])
            del gt;path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):io.native_reference(path,spec,[3,2])

    def test_complete_synthetic_raw_IO_evaluation_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=predictions(root/'pred',shape=(6,6));config['native_wh']=[36,36]
            labels=root/'labels';labels.mkdir();out=root/'out';out.mkdir()
            for i,row in enumerate(config['labels']):
                p=labels/row['image'];z=np.ones((36,36),dtype='<f4');p.write_bytes(z.tobytes())
                row['bytes']=p.stat().st_size;row['sha256']=io.rd.digest(p)
            access=[];result=io.evaluate_sealed(root/'pred',labels,config,out,access)
            self.assertEqual(access,config['names']);self.assertEqual(result['status'],'blocked_relative_support')
            self.assertTrue((out/'image-2-evaluator-only-pairs.npz').exists())

    def test_existing_output_and_plan_drift_refused_without_GPU(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'old';out.mkdir();(out/'keep').write_text('preserve')
            args=SimpleNamespace(plan=ROOT/'configs/terrace-relative-diagnostic-plan.json',output=out,source=ROOT,
                data=None,labels=None,predictions=None,model_source=None)
            with self.assertRaises(ValueError):entry.run(args)
            self.assertEqual((out/'keep').read_text(),'preserve')
            wrong=root/'wrong.json';wrong.write_text('{}');args.plan=wrong
            with self.assertRaises(ValueError):entry.run(args)

    def test_WSL_device_mount_list_never_adds_data_or_history_root(self):
        with patch.object(Path,'is_char_device',return_value=True),patch.object(Path,'is_file',return_value=True):
            mounts=entry.device_mounts()
        self.assertIn('/dev/dxg',mounts);self.assertIn('/sys',mounts)
        self.assertNotIn('/home',mounts);self.assertNotIn('/mnt',mounts)
        with patch.object(Path,'is_char_device',return_value=False),self.assertRaises(RuntimeError):entry.device_mounts()

    def test_direct_forward_probe_counts_once_and_refuses_wrong_shape(self):
        class Model:
            def forward(self,x):return 'one-forward'
            def infer_image(self,x):return self.forward(x)
        model=Model();torch=SimpleNamespace(float32='float32',is_grad_enabled=lambda:False)
        calls,original=entry.install_forward_probe(model,torch)
        tensor=SimpleNamespace(shape=(1,3,518,784),dtype='float32',device=SimpleNamespace(type='cuda'))
        self.assertEqual(model.infer_image(tensor),'one-forward');self.assertEqual(len(calls),1)
        tensor.shape=(1,3,504,378)
        with self.assertRaises(ValueError):model.infer_image(tensor)
        self.assertEqual(len(calls),1);model.forward=original

    def test_derived_model_provenance_matches_every_fixed_source_digest(self):
        p,_=io.configs(PLAN,{'entry':'fixed'})
        for record in PLAN['prediction']['model']['source_files']:
            self.assertEqual(p['source_sha256']['model/'+record['path']],record['sha256'])


if __name__=='__main__':unittest.main()
