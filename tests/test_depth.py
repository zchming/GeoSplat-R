"""Alignment, masking, gradient and provenance checks for the uniform control."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geosplat_r.depth import align_inverse_depth,uniform_inverse_depth_loss,load_depth_priors
from geosplat_r.baseline import sha256


class DepthMath(unittest.TestCase):
    def test_affine_inverse_z_recovery_and_orientation(self):
        y,x=np.mgrid[:30,:40];raw=(x+2*y).astype(np.float32)
        xy=np.stack([x[3:-3:3,3:-3:3].ravel(),y[3:-3:3,3:-3:3].ravel()],axis=1)
        inverse_z=.03*raw[xy[:,1],xy[:,0]]+.5
        target,mask,fit=align_inverse_depth(raw,xy,inverse_z)
        np.testing.assert_allclose(target,.03*raw+.5,atol=1e-6)
        self.assertAlmostEqual(.03,fit['slope'],places=6)
        self.assertFalse(mask[:2].any());self.assertTrue(mask[3:-3,3:-3].all())
        with self.assertRaisesRegex(ValueError,'Nonpositive'):
            align_inverse_depth(raw,xy,4-.03*raw[xy[:,1],xy[:,0]])

    def test_unsupported_anchor_fit_fails_instead_of_fallback(self):
        xy=np.ones((25,2))*5;values=np.ones(25)
        with self.assertRaisesRegex(ValueError,'constant'):
            align_inverse_depth(np.ones((20,20)),xy,values)
        with self.assertRaisesRegex(ValueError,'20'):
            align_inverse_depth(np.ones((20,20)),xy[:5],values[:5])

    def test_uniform_loss_mean_mask_and_expected_z_derivative(self):
        z=torch.tensor([[2.,4.],[8.,16.]],requires_grad=True)
        alpha=torch.tensor([[1.,1.],[.1,1.]])
        target=torch.zeros_like(z);mask=torch.tensor([[True,True],[True,False]])
        loss,fraction=uniform_inverse_depth_loss(z,alpha,target,mask,.5)
        self.assertAlmostEqual(loss.item(),.75);self.assertAlmostEqual(fraction.item(),.5)
        loss.backward()
        torch.testing.assert_close(z.grad,torch.tensor([[-.25,-.0625],[0.,0.]]))

    def test_empty_invalid_render_mask_produces_finite_zero_loss(self):
        z=torch.tensor([[float('nan'),0.]],requires_grad=True)
        loss,fraction=uniform_inverse_depth_loss(z,torch.zeros_like(z),torch.ones_like(z),torch.ones_like(z,dtype=torch.bool),1.)
        self.assertEqual(0.,loss.item());self.assertEqual(0.,fraction.item())
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_prior_provenance_mismatch_rejected_before_cuda(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text(json.dumps({'cameras_sha256':'wrong','points_sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'provenance'):
                load_depth_priors(p,[],'camera','points',sha256)

    def test_audit_rejects_test_sibling_in_separate_process(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'train').mkdir();(p/'test').mkdir();(p/'test/rgb.txt').write_text('held out')
            code="from geosplat_r.isolation import audit_training_reads; from pathlib import Path; import sys; audit_training_reads(Path(sys.argv[1])); Path(sys.argv[2]).read_text()"
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
            r=subprocess.run([sys.executable,'-c',code,str(p/'train'),str(p/'test/rgb.txt')],env=env,capture_output=True,text=True)
            self.assertNotEqual(0,r.returncode);self.assertIn('Non-training asset',r.stderr)


@unittest.skipUnless(os.environ.get('GEOSPLAT_CUDA_CHECK'),'Explicit CUDA parity check not requested')
class RasterChannelParity(unittest.TestCase):
    def test_rgb_and_rgb_ed_forward_and_rgb_only_gradients_match(self):
        from gsplat.rendering import rasterization
        torch.manual_seed(42)
        device='cuda';n=128
        means=torch.rand(n,3,device=device);means[:,:2]-=.5;means[:,2]+=2
        source={'means':means,'quats':torch.randn(n,4,device=device),
            'scales':torch.full((n,3),.08,device=device),'opacities':torch.full((n,),.6,device=device),
            'colors':torch.rand(n,16,3,device=device)*.2}
        camera=torch.eye(4,device=device)[None]
        K=torch.tensor([[[64.,0.,40.],[0.,64.,32.],[0.,0.,1.]]],device=device)
        results=[];gradients=[]
        for mode in ['RGB','RGB+ED']:
            params={k:v.detach().clone().requires_grad_() for k,v in source.items()}
            image,_,_=rasterization(**params,viewmats=camera,Ks=K,width=80,height=64,packed=True,
                sh_degree=3,render_mode=mode,rasterize_mode='classic',near_plane=.01,far_plane=1e10)
            rgb=image[...,:3];(rgb-.3).square().mean().backward()
            results.append(rgb.detach());gradients.append({k:p.grad.detach() for k,p in params.items()})
        torch.testing.assert_close(results[0],results[1],atol=2e-6,rtol=2e-5)
        for key in source:torch.testing.assert_close(gradients[0][key],gradients[1][key],atol=2e-6,rtol=2e-4)


@unittest.skipUnless(os.environ.get('GEOSPLAT_DEPTH_PRIORS'),'Explicit completed training-prior bundle not supplied')
class RealPriors(unittest.TestCase):
    def test_three_training_priors_have_frozen_hashes_and_positive_alignment(self):
        root=Path(os.environ['GEOSPLAT_DEPTH_PRIORS']);j=json.loads((root/'manifest.json').read_text())
        self.assertEqual(['image001.png','image010.png','image019.png'],[r['image'] for r in j['records']])
        self.assertEqual('715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378',j['weights_sha256'])
        for r in j['records']:
            self.assertEqual(r['sha256'],sha256(root/r['file']))
            self.assertGreater(r['alignment']['slope'],0);self.assertGreaterEqual(r['alignment']['anchors'],20)
        access=json.loads((root/'training-access.json').read_text())
        self.assertEqual({'cameras.json','points.npz','images/image001.png','images/image010.png','images/image019.png'},set(access))


@unittest.skipUnless(os.environ.get('GEOSPLAT_DEPTH_PAIR'),'Explicit completed zero/uniform comparison not supplied')
class RealDepthPair(unittest.TestCase):
    def test_pair_preserves_inputs_initialization_and_training_budget(self):
        root=Path(os.environ['GEOSPLAT_DEPTH_PAIR'])
        configs=[];resources=[]
        for name,weight in [('control-zero-30k',0.),('uniform-30k',.01)]:
            run=root/name;c=json.loads((run/'config.json').read_text());r=json.loads((run/'resources.json').read_text())
            self.assertEqual('passed',r['status']);self.assertEqual(30000,r['optimizer_updates'])
            self.assertEqual(weight,c['depth_protocol']['lambda']);self.assertEqual(1000,c['depth_protocol']['start_zero_based_step'])
            self.assertEqual(sha256(run/'source/baseline.py'),c['training_implementation_sha256'])
            self.assertEqual(sha256(run/'source/depth.py'),c['depth_protocol']['depth_implementation_sha256'])
            self.assertLess(r['whole_board_sampled_peak_mib'],12282)
            self.assertEqual({'cameras.json','points.npz','images/image001.png','images/image010.png','images/image019.png'},
                set(json.loads((run/'training-access.json').read_text())))
            rows=json.loads((run/'training-curve.json').read_text())
            self.assertEqual(0.,rows[0]['depth_weight']);self.assertEqual(weight,rows[-1]['depth_weight'])
            self.assertGreater(rows[-1]['depth_valid_fraction'],.1)
            configs.append(c);resources.append(r)
        for field in ['train_images','cameras_sha256','points_sha256','max_steps','resolution','seed','strategy',
                'scene_scale','precision','packed','batch_size','sh_degree','sh_degree_interval','rgb_loss','training_implementation_sha256']:
            self.assertEqual(configs[0][field],configs[1][field],field)
        self.assertEqual(configs[0]['depth_protocol']['priors_manifest_sha256'],configs[1]['depth_protocol']['priors_manifest_sha256'])
        self.assertEqual(resources[0]['initial_parameters_sha256'],resources[1]['initial_parameters_sha256'])
        self.assertEqual('b600c3076269ef9eca9066c9081f163a805cd880a2a1be8d65f2892ebe2d5c17',resources[0]['initial_parameters_sha256'])


if __name__=='__main__':unittest.main()
