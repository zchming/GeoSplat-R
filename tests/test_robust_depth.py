"""CPU-only value, gradient, CLI and actual trainer-fragment checks."""
import ast
import copy
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from geosplat_r.depth import uniform_inverse_depth_loss
from geosplat_r.depth_objectives import (cauchy_penalty, cauchy_inverse_depth_loss,
                                         depth_objective_diagnostics)
from geosplat_r.depth_options import validate_depth_options
from train_baseline import parse_args


def fixture():
    z = torch.tensor([[1., 2., 4.], [8., 16., 32.]], dtype=torch.float64, requires_grad=True)
    target = torch.full_like(z, .3)
    alpha = torch.tensor([[1., .5, .49], [1., 1., 1.]], dtype=z.dtype)
    mask = torch.tensor([[True, True, True], [False, True, True]])
    return z, alpha, target, mask


def analytic_gradient(r):
    t = r / .1
    return 2 * t / (1 + t * t)


class CauchyMath(unittest.TestCase):
    def test_value_sign_symmetry_zero_and_analytic_derivative(self):
        vals = [-10., -.2, -.1, -.01, 0., .01, .1, .2, 10.]
        r = torch.tensor(vals, dtype=torch.float64, requires_grad=True)
        loss = cauchy_penalty(r)
        expected = torch.tensor([.1 * math.log1p((v / .1) ** 2) for v in vals], dtype=r.dtype)
        torch.testing.assert_close(loss, expected, rtol=1e-13, atol=1e-15)
        gradient, = torch.autograd.grad(loss.sum(), r)
        torch.testing.assert_close(gradient, torch.tensor([analytic_gradient(v) for v in vals], dtype=r.dtype), rtol=1e-12, atol=1e-14)
        self.assertEqual(0., loss[4].item());self.assertTrue((loss >= 0).all())
        self.assertLessEqual(gradient.abs().max().item(), 1.)
        self.assertLess(abs(gradient[-1]), abs(gradient[-2]))

    def test_gradcheck_crosses_piecewise_boundary_and_zero(self):
        r = torch.tensor([-.2, -.1, -.001, 0., .001, .1, .2], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: cauchy_penalty(x).sum(), (r,), eps=1e-6, atol=2e-6, rtol=1e-5))

    def test_large_float32_residual_does_not_square_overflow(self):
        vals = [0., 1e-4, .1, 1e10, 1e30, torch.finfo(torch.float32).max]
        r = torch.tensor(vals, dtype=torch.float32, requires_grad=True)
        loss = cauchy_penalty(r)
        self.assertTrue(torch.isfinite(loss).all())
        expected = torch.tensor([.1 * math.log1p((v / .1) ** 2) for v in vals], dtype=r.dtype)
        torch.testing.assert_close(loss, expected, rtol=2e-6, atol=1e-7)
        loss.sum().backward();self.assertTrue(torch.isfinite(r.grad).all())
        self.assertLessEqual(r.grad.abs().max().item(), 1.)

    def test_mask_valid_mean_and_expected_z_chain_rule(self):
        z, alpha, target, mask = fixture()
        loss, fraction = cauchy_inverse_depth_loss(z, alpha, target, mask, .5)
        valid = mask & (alpha >= .5)
        residual = ((1 / z.detach() - target) / .5)[valid]
        expected = sum(.1 * math.log1p((r.item() / .1) ** 2) for r in residual) / 4
        self.assertAlmostEqual(expected, loss.item(), places=13)
        self.assertAlmostEqual(4 / 6, fraction.item())
        loss.backward()
        gradient = torch.zeros_like(z)
        for row, col in torch.nonzero(valid).tolist():
            r = (1 / z.detach()[row, col].item() - .3) / .5
            gradient[row, col] = -analytic_gradient(r) / 4 / .5 / z.detach()[row, col].item() ** 2
        torch.testing.assert_close(z.grad, gradient, rtol=1e-12, atol=1e-14)

    def test_finite_difference_through_expected_z(self):
        z, alpha, target, mask = fixture()
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: cauchy_inverse_depth_loss(value, alpha, target, mask, .5)[0],
            (z,), eps=1e-6, atol=2e-7, rtol=1e-5))

    def test_eligibility_matches_l1_and_weaker_l1_gradient_is_exactly_half(self):
        z, alpha, target, mask = fixture()
        l1, f1 = uniform_inverse_depth_loss(z, alpha, target, mask, .5)
        cauchy, f2 = cauchy_inverse_depth_loss(z, alpha, target, mask, .5)
        self.assertEqual(f1.item(), f2.item())
        g1, = torch.autograd.grad(.01 * l1, z, retain_graph=True)
        g2, = torch.autograd.grad(.005 * l1, z, retain_graph=True)
        self.assertTrue(torch.equal(g2, g1 / 2))
        gc, = torch.autograd.grad(cauchy, z)
        self.assertTrue(torch.equal(gc != 0, g1 != 0))

    def test_all_invalid_or_masked_pixels_have_finite_zero_gradient(self):
        z = torch.tensor([[float('nan'), 0., float('inf'), -1., 1e-6]], dtype=torch.float64, requires_grad=True)
        target = torch.full_like(z, float('nan'))
        loss, fraction = cauchy_inverse_depth_loss(z, torch.ones_like(z), target, torch.ones_like(z, dtype=torch.bool), .5)
        self.assertEqual(0., loss.item());self.assertEqual(0., fraction.item())
        loss.backward();self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))
        d = depth_objective_diagnostics(z, torch.ones_like(z), target, torch.ones_like(z, dtype=torch.bool), .5, 'cauchy', .01)
        self.assertEqual(0, d['depth_valid_count']);self.assertIsNone(d['depth_abs_residual_p50'])
        self.assertIsNone(d['depth_influence_mean']);self.assertEqual(0., d['depth_weighted_abs_dloss_dED_sum'])
        json.dumps(d, allow_nan=False)

    def test_partial_invalid_ed_does_not_leak_nan_or_change_denominator(self):
        z = torch.tensor([[2., float('nan'), 0., 1e-6, float('inf')]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[.1, float('nan'), float('nan'), float('nan'), float('nan')]], dtype=z.dtype)
        loss, fraction = cauchy_inverse_depth_loss(z, torch.ones_like(z), target, torch.ones_like(z, dtype=torch.bool), .5)
        self.assertAlmostEqual(.1 * math.log1p(8 ** 2), loss.item())
        self.assertAlmostEqual(.2, fraction.item());loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue(torch.equal(z.grad[0, 1:], torch.zeros(4, dtype=z.dtype)))

    def test_diagnostics_report_raw_l1_and_lambda_weighted_ed_gradient(self):
        z, alpha, target, mask = fixture()
        for kind, coefficient in [('l1', .01), ('l1', .005), ('cauchy', .01)]:
            loss, _ = (uniform_inverse_depth_loss(z, alpha, target, mask, .5) if kind == 'l1'
                       else cauchy_inverse_depth_loss(z, alpha, target, mask, .5))
            grad, = torch.autograd.grad(coefficient * loss, z, retain_graph=True)
            rng = torch.random.get_rng_state().clone()
            d = depth_objective_diagnostics(z, alpha, target, mask, .5, kind, coefficient)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertAlmostEqual(grad.abs().sum().item(), d['depth_weighted_abs_dloss_dED_sum'], places=13)
            self.assertAlmostEqual(grad.abs().sum().item() / 4, d['depth_weighted_abs_dloss_dED_mean'], places=13)
            self.assertAlmostEqual(loss.item(), d['depth_objective'], places=13)
            raw, _ = uniform_inverse_depth_loss(z, alpha, target, mask, .5)
            self.assertAlmostEqual(raw.item(), d['depth_normalized_l1'], places=13)
            self.assertEqual(4, d['depth_valid_count']);json.dumps(d, allow_nan=False)

    def test_invalid_scalar_parameters_rejected_without_epsilon_fallback(self):
        for scale in [0., -1., float('nan'), float('inf'), 1e-100]:
            with self.assertRaises(ValueError):cauchy_penalty(torch.ones(2, dtype=torch.float32), scale)
        for normalizer in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):cauchy_inverse_depth_loss(*fixture(), normalizer)


class DepthEntryAndWiring(unittest.TestCase):
    def args(self):
        return ['--data', '/unused/train', '--source', '/unused/source', '--output', '/unused/new-output']

    def test_default_and_old_fern_arguments_remain_l1(self):
        args = parse_args(self.args() + ['--seed', '42', '--max-steps', '30000'])
        self.assertEqual('l1', args.depth_loss);self.assertEqual(0., args.depth_lambda)
        self.assertEqual(1000, args.depth_start);self.assertFalse(args.depth_diagnostics)
        a = parse_args(self.args() + ['--depth-priors', '/unused/priors', '--depth-lambda', '.01'])
        self.assertEqual('l1', a.depth_loss)

    def test_invalid_cli_options_fail_before_importing_training_stack(self):
        for extra in [['--depth-loss', 'other'], ['--depth-loss', 'cauchy'],
                      ['--depth-lambda', 'nan'], ['--depth-lambda', '-.1'],
                      ['--depth-cauchy-scale', 'inf'], ['--depth-cauchy-scale', '0'],
                      ['--depth-cauchy-scale', '1e-100'], ['--depth-cauchy-scale', '1e100'],
                      ['--depth-diagnostics'], ['--depth-start', '-1']]:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):parse_args(self.args() + extra)
        code = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','gsplat','fused_ssim'}:
            raise AssertionError('Training stack imported before argument rejection')
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--data','/unused','--source','/unused','--output','/unused','--depth-lambda','nan']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        p = subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts/train_baseline.py')], capture_output=True, text=True)
        self.assertEqual(2, p.returncode);self.assertNotIn('AssertionError', p.stderr)

    def test_help_and_pure_argument_parser_need_no_torch_or_gpu(self):
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/train_baseline.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(0, p.returncode);self.assertIn('--depth-loss', p.stdout)
        code = "import sys;sys.path.insert(0,sys.argv[1]);from train_baseline import parse_args;a=parse_args(['--data','/unused','--source','/unused','--output','/unused']);assert not any(m in sys.modules for m in ['torch','gsplat','fused_ssim'])"
        p = subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts')], capture_output=True, text=True)
        self.assertEqual(0, p.returncode, p.stderr)

    def test_existing_output_is_refused_before_cuda_or_input_reads(self):
        from geosplat_r.baseline import train
        for kind in ['l1', 'cauchy']:
            with tempfile.TemporaryDirectory() as directory:
                marker = Path(directory) / 'historical-marker.txt'
                marker.write_text('preserve')
                args = SimpleNamespace(output=Path(directory), depth_loss=kind,
                    depth_cauchy_scale=.1, depth_lambda=.01,
                    depth_priors=Path('/nonexistent/priors'), depth_diagnostics=False)
                with self.assertRaises(FileExistsError):train(args)
                self.assertEqual('preserve', marker.read_text())
                self.assertEqual([marker], list(Path(directory).iterdir()))

    def test_direct_option_validator_does_not_accept_unknown_or_nonfinite_profiles(self):
        validate_depth_options('cauchy', .1, .01, True)
        for args in [('unknown', .1, .01, True), ('cauchy', .1, .01, False),
                     ('l1', .1, float('inf'), True), ('l1', -1., .01, True)]:
            with self.assertRaises(ValueError):validate_depth_options(*args)

    def test_actual_trainer_depth_fragment_dispatches_at_step1000(self):
        tree = ast.parse((ROOT / 'src/geosplat_r/baseline.py').read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
        node = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'depth_priors is not None')
        fragment = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])), '<actual-depth-fragment>', 'exec')
        for kind, coefficient in [('l1', .01), ('l1', .005), ('cauchy', .01)]:
            for step in [999, 1000]:
                z, alpha, target, mask = fixture()
                rgb_depth = torch.cat([torch.zeros(*z.shape, 3, dtype=z.dtype), z[..., None]], dim=-1)[None]
                env = dict(depth_priors=[dict(target=target, mask=mask, normalizer=.5)], view_index=0,
                    rgb_depth=rgb_depth, alpha=alpha, depth_objective_kind=kind, depth_cauchy_scale=.1,
                    step=step, depth_start=1000, depth_lambda=coefficient, loss=torch.tensor(.3, dtype=z.dtype),
                    uniform_inverse_depth_loss=uniform_inverse_depth_loss, cauchy_inverse_depth_loss=cauchy_inverse_depth_loss)
                exec(fragment, env)
                expected, _ = (cauchy_inverse_depth_loss(z, alpha, target, mask, .5) if kind == 'cauchy' and step >= 1000
                               else uniform_inverse_depth_loss(z, alpha, target, mask, .5))
                self.assertTrue(torch.equal(env['depth_term'], expected))
                self.assertEqual(coefficient if step >= 1000 else 0., env['weight'])
                self.assertAlmostEqual(.3 + env['weight'] * expected.item(), env['loss'].item(), places=13)

    def test_default_metadata_is_exactly_historical_and_cauchy_is_named_truthfully(self):
        old = subprocess.check_output(['git', 'show', '9987bf4:src/geosplat_r/baseline.py'], cwd=ROOT, text=True)
        new = (ROOT / 'src/geosplat_r/baseline.py').read_text()
        def metadata(source, kind, enabled):
            function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            block = next(n for n in function.body if isinstance(n, ast.If) and any(isinstance(v, ast.Assign)
                and any(isinstance(t, ast.Subscript) and ast.unparse(t) == "config['depth_protocol']" for t in v.targets)
                for v in n.body))
            env = dict(depth_priors=object(), config={}, depth_lambda=.01, depth_start=1000,
                depth_root=Path('/unused/priors'), Path=Path, __file__=str(ROOT / 'src/geosplat_r/baseline.py'),
                sha256=lambda path: 'source-hash', depth_diagnostics_enabled=enabled,
                depth_objective_kind=kind, depth_cauchy_scale=.1)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<metadata>', 'exec'), env)
            return env['config']['depth_protocol']
        self.assertEqual(metadata(old, 'l1', False), metadata(new, 'l1', False))
        c = metadata(new, 'cauchy', True)
        self.assertEqual('cauchy', c['objective_kind']);self.assertIn('log', c['loss'])
        self.assertIn('no geometry weights', c['weighting']);self.assertTrue(c['objective_diagnostics'])

    def test_actual_logging_boundary_and_raw_l1_field(self):
        tree = ast.parse((ROOT / 'src/geosplat_r/baseline.py').read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
        logging = next(n for n in loop.body if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp) and len(n.test.values) == 4)
        condition = compile(ast.Expression(logging.test), '<logging-boundary>', 'eval')
        for enabled in [False, True]:
            for step in [0, 999, 1000, 1001, 1999, 29999]:
                expected = step == 0 or (step + 1) % 1000 == 0 or step + 1 == 30000 or (enabled and step == 1000)
                self.assertEqual(expected, eval(condition, dict(step=step, iteration_count=30000, depth_start=1000, depth_diagnostics_enabled=enabled)))
        block = next(n for n in logging.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'depth_diagnostics_enabled')
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<actual-logging-fragment>', 'exec')
        for step in [999, 1000]:
            z, alpha, target, mask = fixture()
            rgb_depth = torch.cat([torch.zeros(*z.shape, 3, dtype=z.dtype), z[..., None]], dim=-1)[None]
            actual, _ = (cauchy_inverse_depth_loss(z, alpha, target, mask, .5) if step >= 1000
                         else uniform_inverse_depth_loss(z, alpha, target, mask, .5))
            env = dict(depth_diagnostics_enabled=True, rgb_depth=rgb_depth, alpha=alpha,
                prior=dict(target=target, mask=mask, normalizer=.5), row={}, step=step, depth_start=1000,
                depth_objective_kind='cauchy', depth_cauchy_scale=.1, weight=.01 if step >= 1000 else 0.,
                depth_objective_diagnostics=depth_objective_diagnostics, view_order_digest=hashlib.sha256(b'order'),
                depth_term=actual)
            exec(compiled, env)
            raw, _ = uniform_inverse_depth_loss(z, alpha, target, mask, .5)
            row = env['row'];self.assertAlmostEqual(raw.item(), row['depth_normalized_l1'])
            self.assertEqual('cauchy' if step >= 1000 else 'l1', row['depth_objective_kind'])
            self.assertEqual('cauchy', row['depth_requested_objective'])
            self.assertEqual(actual.item(), row['depth_objective'])
            self.assertEqual(hashlib.sha256(b'order').hexdigest(), row['view_order_sha256'])

    def test_new_dependency_is_archived_even_for_legacy_default(self):
        tree = ast.parse((ROOT / 'src/geosplat_r/baseline.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'source_dir' for t in n.targets))
        end = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Expr)
                   and ast.unparse(n.value) == 'torch.cuda.reset_peak_memory_stats()')
        block = compile(ast.fix_missing_locations(ast.Module(body=function.body[start:end], type_ignores=[])), '<archive-fragment>', 'exec')
        for enabled in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                env = dict(args=SimpleNamespace(output=Path(directory)), Path=Path,
                    __file__=str(ROOT / 'src/geosplat_r/baseline.py'), scene_input=None,
                    depth_priors=object() if enabled else None, depth_diagnostics_enabled=enabled)
                exec(block, env)
                expected = {'baseline.py', 'depth_options.py'} | ({'depth.py', 'depth_objectives.py'} if enabled else set())
                self.assertEqual(expected, {p.name for p in (Path(directory) / 'source').iterdir()})
                for name in expected:
                    self.assertEqual((ROOT / 'src/geosplat_r' / name).read_bytes(), (Path(directory) / 'source' / name).read_bytes())

    def test_cuda_context_is_not_initialized_by_this_cpu_suite(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
