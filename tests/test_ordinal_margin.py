"""Generated-data tests only; never load actual prediction/GT-cache values."""
import copy
import importlib.util
import json
import itertools
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r import ordinal_margin as m
spec=importlib.util.spec_from_file_location('margin_entry',ROOT/'scripts/run_ordinal_margin_cpu.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def fixture_row(path,key,role):
    return dict(path=str(path),key=key,role=role,bytes=path.stat().st_size,sha256=m.digest(path))


class MarginMathTests(unittest.TestCase):
    def test_midrank_CDF_between_values_zero_and_invalid(self):
        got=m.empirical_cdf([0,1,1,3],[-1,np.nan,0,.5,1,2,3,4])
        self.assertTrue(np.isnan(got[:2]).all())
        np.testing.assert_array_equal(got[2:],[.125,.25,.5,.75,.875,1])

    def test_reference_mask_keeps_zero_and_excludes_nonfinite_negative(self):
        a,n=m.sorted_reference([2,0,-1,np.inf,np.nan])
        np.testing.assert_array_equal(a,[0,2]);self.assertEqual(n,3)

    def test_bad_reference_rejected(self):
        for ref in [[],[2,1],[-1,0],[0,np.inf]]:
            with self.assertRaises(ValueError):m.empirical_cdf(ref,[1])

    def test_constant_descriptor_is_uninformative(self):
        f=m.empirical_cdf(np.ones(20),[1,1,1])
        np.testing.assert_array_equal(f,[.5,.5,.5])
        np.testing.assert_array_equal(m.pair_risk(f,[[0,1],[1,2]],np.ones(3,dtype=bool)),[1,1])

    def test_large_gap_can_be_confidently_wrong(self):
        q=np.array([1.,2.,3.]);z=np.array([1.,2.,4.]);p=np.array([[0,2]])
        self.assertAlmostEqual(m.pair_risk(m.empirical_cdf(q,q),p,np.ones(3,dtype=bool))[0],1/3)
        self.assertEqual(m.rd.relation_loss(q,z,p)[0][0],1)

    def test_invalid_endpoint_worst_and_target_retained(self):
        np.testing.assert_allclose(m.pair_risk(np.array([.1,np.nan,.9]),[[0,1],[0,2]],np.array([True,False,True])),[1,.2],rtol=0,atol=1e-15)

    def test_fractional_boundary_tie_order_invariant(self):
        loss=np.array([0,1,0,1]);risk=np.array([.1,.2,.2,.9])
        self.assertEqual(m.fractional_risk(loss,risk,.5),.25)
        perm=[0,2,1,3];self.assertEqual(m.fractional_risk(loss[perm],risk[perm],.5),.25)

    def test_all_ties_constant_and_full_coverage_identity(self):
        loss=np.array([0,.5,1,1,0])
        for c in m.COVERAGES:self.assertAlmostEqual(m.fractional_risk(loss,np.ones(5),c),loss.mean())
        self.assertEqual(m.fractional_risk(loss,[.4,.9,.1,.6,.5],1),loss.mean())

    def test_fractional_ties_agree_with_all_stable_orderings(self):
        loss=np.array([0,.5,1,0,1]);score=np.array([0,0,1,1,1])
        for coverage in m.COVERAGES:
            risks=[]
            for permutation in itertools.permutations(range(5)):
                ordered=sorted(permutation,key=lambda i:score[i]);remaining=5*coverage;total=0.
                for i in ordered:
                    taken=min(1.,remaining);total+=taken*loss[i];remaining-=taken
                    if remaining<=0:break
                risks.append(total/(5*coverage))
            self.assertAlmostEqual(m.fractional_risk(loss,score,coverage),sum(risks)/len(risks))

    def test_oracle_lower_bound_keeps_full_coverage_errors(self):
        loss=np.array([0,0,.5,1])
        self.assertEqual(m.fractional_risk(loss,loss,.5),0)
        self.assertEqual(m.fractional_risk(loss,loss,1),.375)

    def test_invalid_risk_or_loss_not_silently_dropped(self):
        for loss,risk,c in [([],[],.5),([1],[np.nan],.5),([1],[-1],.5),([.1],[1],.5),([1],[1],0)]:
            with self.assertRaises(ValueError):m.fractional_risk(loss,risk,c)

    def test_shuffle_features_preserves_invalid_mask_and_original_target(self):
        f=np.array([.1,.2,np.nan,.8,.9]);valid=np.isfinite(f);before=f.copy()
        a=m.shuffled_features(f,valid,42,1);b=m.shuffled_features(f,valid,42,1)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(f,before)
        self.assertTrue(np.isnan(a[2]));np.testing.assert_array_equal(np.sort(a[valid]),np.sort(f[valid]))

    def test_constant_oracle_macro_and_zero_error_stratum(self):
        pairs=np.array([[0,1],[0,2],[1,2],[0,3]])
        loss=np.array([0,1,.5,0]);groups={}
        for kind in ['local','cross']:
            groups[kind]=[dict(cell=0,included=True,pairs=pairs,target_loss=loss,invalid_pairs=np.zeros(4,dtype=bool),loss=.375),dict(cell=1,included=False)]
        image=dict(groups=groups,valid=np.ones(4,dtype=bool))
        for mode in ['constant','oracle','margin']:
            out=m.risk_curves(image,np.array([.1,.2,.6,.9]),mode)
            self.assertEqual(out['risk']['1.0'],.375)
            np.testing.assert_array_equal(loss,[0,1,.5,0])
        for c in m.COVERAGES:self.assertEqual(m.risk_curves(image,np.ones(4)*.5,'constant')['risk'][str(c)],.375)
        for kind in groups:groups[kind][0].update(target_loss=np.zeros(4),loss=0.)
        out=m.risk_curves(image,np.ones(4)*.5)
        self.assertEqual(out['strata']['local']['discrimination'],'unknown_no_error')

    def test_assessment_no_image_or_scene_rescue(self):
        def rows(r50):
            return [dict(scene=s,image_index=i,support=dict(passed=True),margin=dict(risk={'0.25':.03,'0.5':r50,'0.75':.08,'1.0':.1},strata={k:dict(risk={'0.5':r50,'1.0':.1}) for k in ['local','cross']}),shuffle_mean={'0.5':.1}) for s in m.SCENES for i in range(3)]
        self.assertEqual(m.assess(rows(.05))['status'],'passed_development_margin_screen')
        self.assertEqual(m.assess(rows(.078))['status'],'failed_margin_direction_gate')
        a=rows(.05);a[0]['margin']['risk']['0.5']=.09
        self.assertEqual(m.assess(a)['status'],'failed_margin_direction_gate')
        a=rows(.05);a[0]['margin']['strata']['local']['risk']['0.5']=.11
        self.assertEqual(m.assess(a)['status'],'failed_margin_direction_gate')
        a=rows(.05);a[0]['support']['passed']=False
        self.assertEqual(m.assess(a)['status'],'blocked_error_support')
        with self.assertRaises(ValueError):m.assess(a[:-1])


class MarginIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.rows=[]
        cls.raw=np.broadcast_to(np.arange(1008,dtype=np.float32)+1,(672,1008)).copy()
        for scene in m.SCENES:
            prefix=cls.root/scene/'prediction/stage/result/predictions';prefix.mkdir(parents=True)
            records=[]
            for i in range(3):
                path=prefix/('%d.npy'%i);np.save(path,cls.raw)
                cls.rows.append(fixture_row(path,scene+'/prediction/stage/result/predictions/'+path.name,'scorer'))
                records.append(dict(name=m.NAMES[scene][i],file=path.name,sha256=m.digest(path)))
            path=prefix/'seal.json';path.write_text(json.dumps(dict(status='sealed_all_predictions',predictions=records)))
            cls.rows.append(fixture_row(path,scene+'/prediction/stage/result/predictions/seal.json','scorer'))
        cls.scorer_rows=cls.rows.copy();cls.sources={'test':'generated-only'}
        cls.descriptor_root=cls.root/'descriptors'
        m.descriptors(cls.scorer_rows,cls.descriptor_root,cls.sources)
        ids=[];cell=[]
        for c in range(108):
            cy,cx=divmod(c,12)
            for j in range(64):ids.append((cy*448+20)*6048+cx*504+20+j);cell.append(c)
        cls.nodes=dict(indices=np.array(ids,dtype=np.int64),cell=np.array(cell,dtype=np.int64),z=np.tile([1.,2.],3456))
        y,x=np.divmod(cls.nodes['indices'],6048);q,valid=m.rd.sample_raw(cls.raw,np.column_stack([(x+.5)/6-.5,(y+.5)/6-.5]))
        for scene in m.SCENES:
            root=cls.root/scene/'eval';root.mkdir()
            for i in range(3):
                groups=m.rd.make_pairs(cls.nodes,i);model=m.rd.score(cls.nodes,groups,q,valid)
                baseline=dict(image_index=i,model=model,pair_fingerprint=m.rd.pair_fingerprint(cls.nodes,groups))
                path=root/('image-%d.json'%i);path.write_text(json.dumps(baseline));cls.rows.append(fixture_row(path,scene+'/eval/'+path.name,'evaluator'))
                payload=dict(native_indices=cls.nodes['indices'],cell=cls.nodes['cell'],reference_Z=cls.nodes['z'])
                for kind in ['local','cross']:
                    payload[kind+'_candidate_pairs']=np.concatenate([c['candidates'] for c in groups[kind]])
                    payload[kind+'_eligible']=np.concatenate([c['eligible'] for c in groups[kind]])
                    payload[kind+'_source_cell']=np.concatenate([np.full(len(c['candidates']),c['cell'],dtype=np.int64) for c in groups[kind]])
                path=root/('image-%d-evaluator-only-pairs.npz'%i);np.savez(path,**payload);cls.rows.append(fixture_row(path,scene+'/eval/'+path.name,'evaluator'))
            path=root/'assessment.json';path.write_text('{}');cls.rows.append(fixture_row(path,scene+'/eval/'+path.name,'evaluator'))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_checked_plan_identity_and_drift(self):
        self.assertEqual(m.checked_plan(ROOT/'configs/ordinal-error-margin-plan.json')['protocol'],'ordinal-error-margin-hypothesis-v1')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wrong.json';path.write_text('{}')
            with self.assertRaises(ValueError):m.checked_plan(path)

    def test_metadata_gate_never_loads_arrays_or_features(self):
        with patch.object(np,'load',side_effect=AssertionError('array loading forbidden')),patch.object(m,'sorted_reference',side_effect=AssertionError('feature forbidden')):
            result=m.inspect_assets(self.rows)
        self.assertEqual(result['files'],22);self.assertEqual(len(result['prediction_headers']),6)
        self.assertFalse(result['evaluator_payload_deserialized'])

    def test_metadata_opaque_evaluator_blob_not_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'not-a-zip.npz';path.write_bytes(b'opaque fixture, deliberately not NPZ')
            result=m.inspect_assets([fixture_row(path,'terrace/opaque.npz','evaluator')])
            self.assertFalse(result['descriptor_computed'])

    def test_header_dtype_shape_fortran_and_length_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw.npy'
            for data in [np.ones((2,3),dtype=np.float32),np.ones((672,1008),dtype=np.float64),np.asfortranarray(self.raw)]:
                np.save(path,data)
                with self.assertRaises(ValueError):m.prediction_header(path)
            np.save(path,self.raw)
            with path.open('ab') as f:f.write(b'extra')
            with self.assertRaises(ValueError):m.prediction_header(path)

    def test_descriptor_role_rejects_GT_before_any_load(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(np,'load') as loader,self.assertRaises(ValueError):m.descriptors(self.rows,Path(tmp)/'out',self.sources)
        loader.assert_not_called()

    def test_six_descriptors_seal_source_and_raw_identity(self):
        seal,refs=m.verified_descriptors(self.descriptor_root,self.rows,self.sources)
        self.assertEqual(len(refs),6);self.assertEqual(refs[0].shape,(670672,))
        self.assertTrue(all(r['status']=='passed' for r in seal['records']))
        with self.assertRaises(ValueError):m.verified_descriptors(self.descriptor_root,self.rows,{'test':'changed'})

    def test_no_overwrite_and_partial_seal_refused(self):
        with self.assertRaises(FileExistsError):m.descriptors(self.scorer_rows,self.descriptor_root,self.sources)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);seal=json.loads((self.descriptor_root/'seal.json').read_text());seal['records']=seal['records'][:-1];(root/'seal.json').write_text(json.dumps(seal))
            with patch.object(m,'prepare_image') as loader,self.assertRaises(ValueError):m.evaluate(self.rows,root,root/'out',self.sources,[])
            loader.assert_not_called()

    def test_corrupt_sixth_refused_before_evaluator_asset_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'copy';shutil.copytree(self.descriptor_root,root);(root/'delivery_area-2.npy').write_bytes(b'corrupt')
            with patch.object(m,'inspect_assets') as assets,patch.object(m,'prepare_image') as loader,self.assertRaises(ValueError):m.evaluate(self.rows,root,Path(tmp)/'out',self.sources,[])
            assets.assert_not_called();loader.assert_not_called()

    def test_frozen_pair_fingerprint_and_baseline_reproduction(self):
        cache=next(r for r in self.rows if r['key']=='terrace/eval/image-0-evaluator-only-pairs.npz')
        baseline=next(r for r in self.rows if r['key']=='terrace/eval/image-0.json')
        image=m.prepare_image(self.raw,cache['path'],baseline['path'])
        self.assertTrue(image['support']['passed'])
        self.assertGreaterEqual(image['support']['unique_wrong_pairs'],50)
        with tempfile.TemporaryDirectory() as tmp:
            bad=json.loads(Path(baseline['path']).read_text());bad['model']['loss']+=.1;p=Path(tmp)/'bad.json';p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):m.prepare_image(self.raw,cache['path'],p)

    def test_generated_six_image_evaluation_and_target_fidelity(self):
        with tempfile.TemporaryDirectory() as tmp:
            access=[];out=Path(tmp)/'evaluation';result=m.evaluate(self.rows,self.descriptor_root,out,self.sources,access)
            self.assertIn(result['status'],['passed_development_margin_screen','failed_margin_direction_gate'])
            self.assertEqual(len(access),12)
            row=json.loads((out/'terrace-0.json').read_text());self.assertEqual(len(row['shuffled']),20)
            for control in [row['margin'],row['constant'],row['oracle'],*row['shuffled']]:
                self.assertAlmostEqual(control['risk']['1.0'],row['margin']['risk']['1.0'])
            for c in m.COVERAGES:self.assertAlmostEqual(row['constant']['risk'][str(c)],row['constant']['risk']['1.0'])

    def test_role_manifest_does_not_expose_evaluator_hashes_to_scorer(self):
        plan=m.checked_plan(ROOT/'configs/ordinal-error-margin-plan.json')
        manifest=dict(protocol='ordinal-margin-asset-locations-v1',plan_sha256=m.PLAN_SHA,files=[dict(key=m.asset_key(row),path=str(self.root/('location%d'%i))) for i,row in enumerate(plan['input_assets'])])
        p=entry.role_rows(plan,manifest,'describe',m.PLAN_SHA);self.assertEqual(len(p),8)
        self.assertTrue(all(x['role']=='scorer' for x in p))
        changed=copy.deepcopy(plan)
        for row in changed['input_assets']:
            if row['role']=='evaluator':row['sha256']='different-label-content'
        self.assertEqual(p,entry.role_rows(changed,manifest,'describe',m.PLAN_SHA))
        manifest['files'].append(manifest['files'][0])
        with self.assertRaises(ValueError):entry.role_rows(plan,manifest,'describe',m.PLAN_SHA)

    def test_entry_refuses_existing_output_without_computation(self):
        plan=m.checked_plan(ROOT/'configs/ordinal-error-margin-plan.json')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'locations.json';manifest.write_text(json.dumps(dict(protocol='ordinal-margin-asset-locations-v1',plan_sha256=m.PLAN_SHA,files=[dict(key=m.asset_key(row),path=str(root/('asset%d'%i))) for i,row in enumerate(plan['input_assets'])])))
            out=root/'old';out.mkdir();(out/'keep').write_text('keep')
            args=SimpleNamespace(stage='validate',source=ROOT,plan=ROOT/'configs/ordinal-error-margin-plan.json',asset_manifest=manifest,output=out,descriptors=None)
            with patch.object(m,'descriptors') as describe,self.assertRaises(ValueError):entry.run(args)
            describe.assert_not_called();self.assertEqual((out/'keep').read_text(),'keep')


if __name__=='__main__':unittest.main()
