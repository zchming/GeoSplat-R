"""CPU-only staged-entry contracts; GPU calls are mocked, never executed."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery_entry',ROOT/'scripts/run_delivery_relative_diagnostic.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def args(root,stage='preflight'):
    return SimpleNamespace(stage=stage,source=ROOT,plan=ROOT/'configs/delivery-area-relative-adapter-plan.json',
        input_manifest=ROOT/'configs/delivery-area-relative-inputs.json',base_plan=ROOT/'configs/terrace-relative-diagnostic-plan.json',
        review_plan=ROOT/'configs/second-scene-ordinal-review.json',python=Path(sys.executable),
        model_source=root/'vendor',weights=root/'weight',data=None,labels=None,predictions=None,output=root/'out')


def contexts():
    io=entry.bridge();sources={n:io.rd.digest(ROOT/'src/geosplat_r'/n) for n in ['relative_io.py','relative_diagnostics.py']}
    a=args(Path('/fixture'))
    return io.load_contexts(a.plan,a.input_manifest,a.base_plan,a.review_plan,sources)


class DeliveryExecutionTests(unittest.TestCase):
    def test_three_stage_required_and_forbidden_locations(self):
        for stage in ['preflight','predict','evaluate']:
            a=args(Path('/fixture'),stage)
            if stage=='predict':a.data=Path('/RGB')
            if stage=='evaluate':a.model_source=a.weights=None;a.predictions=Path('/sealed');a.labels=Path('/GT')
            entry.validate_stage_args(a)
            bad=copy.copy(a)
            if stage=='evaluate':bad.weights=Path('/forbidden')
            else:bad.labels=Path('/forbidden')
            with self.assertRaises(ValueError):entry.validate_stage_args(bad)
        a=args(Path('/fixture'));a.data=Path('/realRGB')
        with self.assertRaises(ValueError):entry.validate_stage_args(a)

    def test_internal_probe_cannot_be_public_stage(self):
        with self.assertRaises(ValueError):entry.validate_stage_args(args(Path('/fixture'),'probe'))

    def test_config_scene_shapes_and_roles_fail_closed(self):
        p,e=contexts();entry.validate_context(p,'preflight');entry.validate_context(p,'predict');entry.validate_context(e,'evaluate')
        for key,value in [('names',['terrace']),('plan_sha256','wrong'),('work_wh',[504,378]),('native_wh',[36,36])]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):entry.validate_context(bad,'preflight')
        with self.assertRaises(ValueError):entry.validate_context(e,'predict')
        with self.assertRaises(ValueError):entry.validate_context(p,'evaluate')

    def test_unchanged_forward_shape_preprocessing_and_import_observers(self):
        original=ast.parse((ROOT/'scripts/run_relative_diagnostic.py').read_text());new=ast.parse((ROOT/'scripts/run_delivery_relative_diagnostic.py').read_text())
        old={n.name:n for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};current={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name in ['make_dummy','install_forward_probe','verify_imported_model_files','device_mounts','denied_reads','BoardSampler']:
            self.assertEqual(ast.dump(old[name]),ast.dump(current[name]),name)
        source=ast.get_source_segment((ROOT/'scripts/run_relative_diagnostic.py').read_text(),old['gpu_worker'])
        source=source.replace("rd.write_json(pred_root/'seal.json',dict(status='sealed_all_predictions',plan_sha256=rd.PLAN_SHA,\n            source_sha256=config['source_sha256'],input_sha256=config['input_sha256'],predictions=records))","io.seal_predictions(pred_root,config)")
        self.assertEqual(ast.dump(ast.parse(source).body[0]),ast.dump(current['gpu_worker']))

    def test_CPU_sandbox_never_queries_devices(self):
        rt=Mock();rt.sandbox.return_value=['python','/code/prepare_sensor_anchors_cpu.py']
        with patch.object(entry,'runtime',return_value=rt),patch.object(entry,'device_mounts') as device:
            cmd=entry.sandbox_command(Path('/python'),['white'],['probe'])
            device.assert_not_called();self.assertIn('/code/run_delivery_relative_diagnostic.py',cmd)

    def test_missing_WSL_GPU_fails_without_fallback(self):
        with patch.object(Path,'is_char_device',return_value=False),self.assertRaises(RuntimeError):entry.device_mounts()

    def test_existing_output_preserved_before_any_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=args(Path(tmp));a.output.mkdir();(a.output/'keep').write_text('keep')
            with patch.object(entry,'runtime') as runtime,self.assertRaises(ValueError):entry.run(a)
            runtime.return_value.execute.assert_not_called();self.assertEqual((a.output/'keep').read_text(),'keep')

    def test_invalid_stage_location_rejected_before_source_load(self):
        a=args(Path('/fixture'));a.labels=Path('/GT')
        with patch.object(entry,'bridge') as bridge,self.assertRaises(ValueError):entry.run(a)
        bridge.assert_not_called()

    def test_preflight_staging_never_mounts_real_RGB_GT_or_data_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=args(root);a.model_source.mkdir();a.weights.write_bytes(b'fake')
            io=entry.bridge();p,e=contexts();p['model']=dict(source_files=[],weights_bytes=4,weights_sha256=io.rd.digest(a.weights))
            rt=Mock();rt.sandbox.side_effect=lambda py,mounts,argv:['python','/code/prepare_sensor_anchors_cpu.py']+mounts+argv
            commands=[]
            def execute(cmd,*unused,**kwargs):
                commands.append(cmd)
                if len(commands)==2:raise RuntimeError('CPU test stops before any GPU execution')
            rt.execute.side_effect=execute
            with patch.object(io,'load_contexts',return_value=(p,e)),patch.object(entry,'bridge',return_value=io),patch.object(entry,'runtime',return_value=rt),patch.object(entry,'device_mounts',return_value=['--dev-bind','/dev/dxg','/dev/dxg']),patch.object(entry,'BoardSampler') as sampler:
                sampler.return_value.stop.return_value=dict(samples=1,sampled_whole_board_peak_MiB=1)
                with self.assertRaisesRegex(RuntimeError,'CPU test stops'):entry.run(a)
            self.assertEqual(len(commands),2)
            for cmd in commands:
                self.assertNotIn('/labels',cmd);self.assertNotIn('/inputs',cmd);self.assertNotIn('/home',cmd)
                self.assertNotIn(str(root),cmd)
            self.assertNotIn('/dev/dxg',commands[0]);self.assertIn('/dev/dxg',commands[1])

    def test_unsealed_evaluation_has_no_label_stat_or_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=args(root,'evaluate');a.model_source=a.weights=None;a.predictions=root/'unsealed';a.labels=root/'forbidden-labels'
            io=entry.bridge();rt=Mock();rt.sandbox.side_effect=lambda py,mounts,argv:['python','/code/prepare_sensor_anchors_cpu.py']+mounts+argv
            with patch.object(io,'sealed_predictions',side_effect=ValueError('unsealed')),patch.object(entry,'bridge',return_value=io),patch.object(entry,'runtime',return_value=rt),patch.object(Path,'is_symlink') as stat:
                with self.assertRaisesRegex(ValueError,'unsealed'):entry.run(a)
                stat.assert_not_called()
            self.assertEqual(rt.execute.call_count,1)


if __name__=='__main__':unittest.main()
