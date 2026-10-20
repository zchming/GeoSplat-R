"""CPU checks for preflight boundaries, immutable inputs and forbidden updates."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('depth_gpu_entry', ROOT/'scripts/check_depth_objective_gpu.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def contract_fixture(root):
    data, priors, source, output = [root/n for n in ['train', 'priors', 'source', 'new-output']]
    args = SimpleNamespace(data=data, scene_manifest=data/'input-manifest.json',
                           priors=priors, source=source, output=output)
    expected = [data/'cameras.json', data/'points.npz', args.scene_manifest, priors/'manifest.json']
    for n in ['image001', 'image012', 'image025']:
        expected += [data/'images'/(n+'.png'), priors/(n+'.npz')]
    contract = dict(protocol='leaves-robust-depth-zero-update-v1', seed=44, scale=.1,
                    environment={'FROZEN_TEST': '1'}, assets=[{'absolute_path':str(p)} for p in expected],
                    **{k:str(getattr(args,k)) for k in ['data','scene_manifest','priors','source','output']})
    return args, contract


class PreflightEntry(unittest.TestCase):
    def test_help_does_not_import_heavy_stack(self):
        code = """
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','gsplat','numpy','fused_ssim'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result=subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts/check_depth_objective_gpu.py')],capture_output=True,text=True)
        self.assertEqual(0,result.returncode,result.stderr)
        self.assertIn('--contract',result.stdout)

    def test_existing_output_refused_before_contract_or_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'keep.txt';marker.write_text('keep')
            argv=[]
            for field in ['data','scene-manifest','priors','source','contract']:
                argv += ['--'+field,'/nonexistent']
            with self.assertRaises(FileExistsError):entry.main(argv+['--output',directory])
            self.assertEqual('keep',marker.read_text())
            self.assertEqual([marker],list(Path(directory).iterdir()))

    def test_exact_contract_locations_seed_scale_and_ten_assets(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'FROZEN_TEST':'1'},clear=True):
            args, contract=contract_fixture(Path(directory));entry.validate_contract(args,contract)
            for key,value in [('seed',43),('scale',.2),('source','/wrong')]:
                bad=copy.deepcopy(contract);bad[key]=value
                with self.assertRaises(ValueError):entry.validate_contract(args,bad)
            for bad in [contract['assets'][:-1],contract['assets']+[{'absolute_path':directory+'/test/image000.png'}]]:
                altered=copy.deepcopy(contract);altered['assets']=bad
                with self.assertRaises(ValueError):entry.validate_contract(args,altered)

    def test_unexpected_environment_or_wrong_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'FROZEN_TEST':'1'},clear=True):
            args, contract=contract_fixture(Path(directory))
            os.environ['PYTHONPATH']='/wrong'
            with self.assertRaises(ValueError):entry.validate_contract(args,contract)
            del os.environ['PYTHONPATH'];args.scene_manifest=Path(directory)/'test/input-manifest.json'
            contract['scene_manifest']=str(args.scene_manifest)
            with self.assertRaises(ValueError):entry.validate_contract(args,contract)

    def test_read_boundary_rejects_test_anchor_weights_checkpoint_write_and_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);allowed=root/'prepared/train/image001.png';weights=root.parent/'fixture-weights'
            boundary=entry.ReadBoundary(root,[allowed],[weights])
            boundary('open',(str(allowed),'r',os.O_RDONLY))
            self.assertEqual({str(allowed.resolve())},boundary.reads)
            paths=[root/'prepared/test/image000.png',root/'anchors/x.npz',weights/'alex.pth',root.parent/'old.pt']
            for path in paths:
                with self.assertRaises(RuntimeError):boundary('open',(str(path),'r',os.O_RDONLY))
            with self.assertRaises(RuntimeError):boundary('open',(str(allowed),'w',os.O_WRONLY))
            with self.assertRaises(RuntimeError):boundary('socket.connect',(None,('example.invalid',443)))
            self.assertEqual(6,len(boundary.rejections))

    def test_boundary_resolves_symlink_and_byte_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);test=root/'test';test.mkdir();outside=root.parent/('fixture-alias-'+root.name)
            try:
                outside.symlink_to(test,target_is_directory=True)
                boundary=entry.ReadBoundary(root,[],[])
                with self.assertRaises(RuntimeError):boundary('open',(bytes(str(outside/'image.png'),'utf8'),'r',os.O_RDONLY))
            finally:outside.unlink()

    def test_actual_audit_hook_rejects_fixture_test_file_in_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'test').mkdir();target=root/'test/image000.png';target.write_text('fixture')
            code="""
import importlib.util,pathlib,sys
s=importlib.util.spec_from_file_location('entry',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=m.ReadBoundary(sys.argv[2],[],[]);sys.addaudithook(b)
try:pathlib.Path(sys.argv[3]).read_bytes()
except RuntimeError:assert len(b.rejections)==1
else:raise AssertionError('Test read allowed')
"""
            p=subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts/check_depth_objective_gpu.py'),str(root),str(target)],capture_output=True,text=True)
            self.assertEqual(0,p.returncode,p.stderr)

    def test_changed_file_is_rejected_before_version_or_git_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source.py';p.write_text('original')
            contract=dict(files={str(p):entry.sha(p)},assets=[],versions={},upstream_heads={})
            entry.verify_files(contract);p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source/package'):entry.verify_files(contract)

    def test_optimizer_scheduler_strategy_and_checkpoint_calls_are_blocked(self):
        import torch
        class Strategy:
            def step_pre_backward(self):pass
            def step_post_backward(self):pass
        attempts=[]
        with contextlib.ExitStack() as stack:
            entry.forbid_operations(stack,torch,Strategy,attempts)
            operations=[lambda:torch.optim.Adam.step(None),lambda:torch.optim.lr_scheduler.LRScheduler.step(None),
                        lambda:Strategy().step_post_backward(),lambda:torch.save({},'unused.pt'),lambda:torch.load('unused.pt')]
            for operation in operations:
                with self.assertRaises(RuntimeError):operation()
        self.assertEqual(5,len(attempts));self.assertFalse(torch.cuda.is_initialized())

    def test_sampling_hashes_are_repeatable_and_do_not_consume_global_rng(self):
        import hashlib
        import torch
        state=torch.random.get_rng_state().clone()
        one=entry.sampling_hashes(torch,6);two=entry.sampling_hashes(torch,6)
        self.assertEqual(one,two);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        generator=torch.Generator().manual_seed(44)
        expected=torch.randperm(3,generator=generator).tolist()+torch.randperm(3,generator=generator).tolist()
        digest=hashlib.sha256()
        for value in expected:digest.update(value.to_bytes(4,'big'))
        self.assertEqual(digest.hexdigest(),one['6'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_source_has_no_training_or_step_invocations(self):
        import ast
        tree=ast.parse((ROOT/'scripts/check_depth_objective_gpu.py').read_text())
        forbidden={'step','step_pre_backward','step_post_backward','train','evaluate','save'}
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls & forbidden)
        self.assertNotIn('geosplat_r.baseline.train',(ROOT/'scripts/check_depth_objective_gpu.py').read_text())


if __name__=='__main__':unittest.main()
