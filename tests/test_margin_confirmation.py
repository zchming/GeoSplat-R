"""Synthetic confirmation gates and descriptor integrity; no real GT or model."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r import ordinal_margin as margin
spec=importlib.util.spec_from_file_location('confirmation_entry',ROOT/'scripts/run_margin_confirmation.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class ConfirmationTests(unittest.TestCase):
    def context(self):
        return dict(confirmation_plan_sha256=entry.PLAN_SHA,names=['dslr_images/A.JPG','dslr_images/B.JPG','dslr_images/C.JPG'],native_wh=[6048,4032],work_wh=[1008,672],input_sha256={'RGB':'fixed'},inputs={'RGB':'identity'},model={'model':'frozen'},labels=[{'secret':'GT'}])

    def rows(self):
        return [dict(image_index=i,support={'passed':True},margin={'risk':{'0.25':.01,'0.5':.02,'0.75':.05,'1.0':.1},'strata':{'local':{'risk':{'0.5':0.,'1.0':0.}},'cross':{'risk':{'0.5':.04,'1.0':.2}}}},shuffle_mean={'0.5':.1}) for i in range(3)]

    def test_descriptor_context_independent_of_labels_and_model(self):
        a=self.context();b=copy.deepcopy(a);b['labels']=[{'secret':'changed'}];b['model']={'changed':True}
        cfg=entry.role_config(a,'describe',{'source':'fixed'})
        self.assertEqual(cfg,entry.role_config(b,'describe',{'source':'fixed'}))
        self.assertTrue(set(cfg).isdisjoint({'labels','model','inputs'}))

    def test_predictor_never_receives_labels(self):
        for stage in ['preflight','predict']:
            cfg=entry.role_config(self.context(),stage,{})
            self.assertNotIn('labels',cfg)
            self.assertEqual('inputs' in cfg,stage=='predict')

    def test_evaluator_has_no_model_or_input_mount_configuration(self):
        cfg=entry.role_config(self.context(),'evaluate',{})
        self.assertNotIn('model',cfg);self.assertNotIn('inputs',cfg);self.assertIn('labels',cfg)

    def test_wrong_scene_grid_path_and_plan_refused(self):
        for field,value in [('confirmation_plan_sha256','changed'),('native_wh',[6000,4000]),('names',['/bad','B','C']),('names',['dslr_images/A.JPG']*3)]:
            cfg=self.context();cfg[field]=value
            with self.assertRaises(ValueError):entry.role_config(cfg,'describe',{})

    def test_domain_excludes_border_and_keeps_zero(self):
        raw=np.full((672,1008),-100,dtype=np.float32);raw[2:670,2:1006]=0
        values,c=entry.domain_reference(raw,margin)
        self.assertEqual(len(values),670672);self.assertEqual(c['invalid_pixels'],0);self.assertEqual(c['valid_fraction'],1.)
        self.assertTrue((values==0).all())

    def test_domain_invalid_support_not_filled(self):
        raw=np.ones((672,1008),dtype=np.float32);raw[2:10,2:1006]=np.nan
        values,c=entry.domain_reference(raw,margin)
        self.assertEqual(c['invalid_pixels'],8032);self.assertLess(c['valid_fraction'],.99)
        self.assertEqual(len(values)+c['invalid_pixels'],670672)

    def test_three_image_positive_with_zero_error_stratum(self):
        result=entry.assess(self.rows());self.assertEqual(result['status'],'passed_unopened_scene_margin_screen')
        self.assertEqual(result['image_gates'],[True]*3)

    def test_single_failed_image_stops_confirmation(self):
        rows=self.rows();rows[1]['margin']['risk']['0.5']=.081
        self.assertEqual(entry.assess(rows)['status'],'failed_confirmation_margin_gate')

    def test_single_no_harm_failure_stops_confirmation(self):
        rows=self.rows();rows[2]['margin']['strata']['local']['risk']['0.5']=.0001
        self.assertEqual(entry.assess(rows)['status'],'failed_confirmation_margin_gate')

    def test_scene_gate_independent_of_image_gate(self):
        rows=self.rows()
        for row in rows:row['margin']['risk']['0.5']=.078
        result=entry.assess(rows);self.assertTrue(all(result['image_gates']));self.assertFalse(result['scene_gate'])

    def test_support_block_distinct_from_negative(self):
        rows=self.rows();rows[0]['support']['passed']=False
        self.assertEqual(entry.assess(rows)['status'],'blocked_error_support')
        with self.assertRaises(ValueError):entry.assess(rows[:2])

    def test_all_descriptors_validate_and_tamper_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred=root/'pred';pred.mkdir();desc=root/'desc';desc.mkdir()
            raw=np.ones((672,1008),dtype=np.float32);arrays=[raw]*3
            cfg=entry.role_config(self.context(),'describe',{'source':'fixed'});records=[]
            for i in range(3):
                name=f'{i}.npy';np.save(pred/name,raw);values,counts=entry.domain_reference(raw,margin);np.save(desc/name,values)
                records.append(dict(file=name,sha256=margin.digest(desc/name),raw_sha256=margin.digest(pred/name),counts=counts,passed=True))
            seal={k:cfg[k] for k in ['names','source_sha256','input_sha256','confirmation_plan_sha256']};seal['records']=records
            (desc/'seal.json').write_text(json.dumps(seal))
            refs,passed=entry.verify_descriptors(desc,cfg,arrays,margin,pred);self.assertTrue(passed);self.assertEqual(len(refs),3)
            records[2]['raw_sha256']='bad';(desc/'seal.json').write_text(json.dumps(seal))
            with self.assertRaises(ValueError):entry.verify_descriptors(desc,cfg,arrays,margin,pred)
