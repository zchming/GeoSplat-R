"""One train-only CUDA forward/backward preflight; zero optimizer updates."""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'scene-manifest', 'priors', 'source', 'contract', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    return parser.parse_args(argv)


class ReadBoundary:
    """Audit explicit asset reads; this is an error detector, not a sandbox."""
    def __init__(self, protected_root, allowed, denied_roots):
        self.protected = Path(protected_root).resolve()
        self.allowed = {Path(p).resolve() for p in allowed}
        self.denied = [Path(p).resolve() for p in denied_roots]
        self.reads = set()
        self.rejections = []

    def __call__(self, event, arguments):
        if event == 'socket.connect':
            self.rejections.append('network connection')
            raise RuntimeError('Network connections forbidden in preflight')
        if event != 'open' or not isinstance(arguments[0], (str, bytes)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        protected = path == self.protected or self.protected in path.parents
        denied = any(path == root or root in path.parents for root in self.denied)
        mode, flags = arguments[1:3]
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                   or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR)))
        if (denied or path.suffix.lower() in {'.pt', '.pth', '.ckpt'}
                or (protected and (path not in self.allowed or writing))):
            self.rejections.append(str(path))
            raise RuntimeError('Forbidden asset/checkpoint access: ' + str(path))
        if protected:
            self.reads.add(str(path))


def validate_contract(args, contract):
    if contract['protocol'] != 'leaves-robust-depth-zero-update-v1':
        raise ValueError('Wrong preflight contract')
    if contract['seed'] != 44 or contract['scale'] != .1:
        raise ValueError('Frozen seed/scale changed')
    for field in ['data', 'scene_manifest', 'priors', 'source', 'output']:
        if getattr(args, field).resolve() != Path(contract[field]).resolve():
            raise ValueError('Contract location differs: ' + field)
    if args.scene_manifest.resolve() != args.data.resolve() / 'input-manifest.json':
        raise ValueError('Manifest must be inside explicit train directory')
    names = ['image001', 'image012', 'image025']
    expected = {args.data / 'cameras.json', args.data / 'points.npz', args.scene_manifest,
                args.priors / 'manifest.json',
                *(args.data / 'images' / (n + '.png') for n in names),
                *(args.priors / (n + '.npz') for n in names)}
    if {Path(v['absolute_path']).resolve() for v in contract['assets']} != {p.resolve() for p in expected}:
        raise ValueError('Contract must contain exactly ten frozen training assets')
    for key, value in contract['environment'].items():
        if os.environ.get(key) != value:
            raise ValueError('Frozen environment mismatch: ' + key)
    if set(os.environ) != set(contract['environment']):
        raise ValueError('Unexpected inherited environment variables')


def verify_files(contract):
    for path, digest in contract['files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source/package mismatch: ' + path)
    for asset in contract['assets']:
        path = Path(asset['absolute_path'])
        if path.stat().st_size != asset['bytes'] or sha(path) != asset['sha256']:
            raise ValueError('Frozen training input mismatch: ' + str(path))
    for name, expected in contract['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Installed version changed: ' + name)
    for source, head in contract['upstream_heads'].items():
        actual = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True).strip()
        if actual != head or dirty:
            raise ValueError('Upstream changed or dirty: ' + source)


def gpu_identity():
    raw = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=uuid,name,driver_version,memory.total,memory.free,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    row = next(csv.reader(raw.splitlines()))
    return dict(zip(['uuid', 'name', 'driver', 'total_mib', 'free_mib', 'used_mib'],
                    [v.strip() for v in row]))


def parameter_digest(params):
    digest = hashlib.sha256()
    for key, value in params.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def sampling_hashes(torch, count=30000):
    generator = torch.Generator().manual_seed(44)
    digest = hashlib.sha256()
    order = []
    result = {}
    for index in range(count):
        if not order:
            order = torch.randperm(3, generator=generator).tolist()
        digest.update(int(order.pop(0)).to_bytes(4, 'big'))
        if index + 1 in {1, 1000, 1001, count}:
            result[str(index + 1)] = digest.hexdigest()
    return result


def synthetic_loss_checks(torch, uniform, cauchy, diagnostics, tolerances):
    records = []
    for empty in [False, True]:
        for kind, coefficient in [('l1', .01), ('l1', .005), ('cauchy', .01)]:
            results = []
            for device in ['cpu', 'cuda']:
                z = torch.tensor([[1., 2., 4.], [8., 16., 32.]], device=device, requires_grad=True)
                alpha = torch.tensor([[1., .5, .49], [1., 1., 1.]], device=device)
                target = torch.full_like(z, .3)
                mask = torch.tensor([[True, True, True], [False, True, True]], device=device)
                if empty:
                    mask.zero_()
                loss, fraction = (uniform(z, alpha, target, mask, .5) if kind == 'l1'
                                  else cauchy(z, alpha, target, mask, .5, .1))
                (coefficient * loss).backward()
                rng = torch.random.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
                record = diagnostics(z, alpha, target, mask, .5, kind, coefficient, .1)
                assert torch.equal(rng, torch.random.get_rng_state()), 'CPU RNG changed by diagnostics'
                if cuda_rng is not None:
                    assert torch.equal(cuda_rng, torch.cuda.get_rng_state()), 'CUDA RNG changed by diagnostics'
                assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
                if empty:
                    assert loss.item() == 0. and torch.count_nonzero(z.grad).item() == 0
                assert record['depth_valid_count'] == (0 if empty else 4)
                assert abs(z.grad.abs().sum().item() - record['depth_weighted_abs_dloss_dED_sum']) <= 1e-6
                results.append((loss.detach().cpu(), z.grad.detach().cpu(), fraction.item()))
            torch.testing.assert_close(results[0][0], results[1][0],
                rtol=tolerances['synthetic_loss_cpu_gpu_rtol'], atol=tolerances['synthetic_loss_cpu_gpu_atol'])
            torch.testing.assert_close(results[0][1], results[1][1],
                rtol=tolerances['synthetic_ED_gradient_cpu_gpu_rtol'], atol=tolerances['synthetic_ED_gradient_cpu_gpu_atol'])
            assert results[0][2] == results[1][2]
            records.append({'kind': kind, 'lambda': coefficient, 'empty': empty,
                'loss': results[1][0].item(), 'max_ED_gradient_cpu_gpu_abs_error':
                (results[0][1] - results[1][1]).abs().max().item()})
    return records


def known_depth_check(torch, rasterization, uniform, cauchy, tolerances):
    means = torch.tensor([[0., 0., 2.]], device='cuda', requires_grad=True)
    quat = torch.tensor([[1., 0., 0., 0.]], device='cuda')
    scales = torch.full((1, 3), .1, device='cuda')
    opacity = torch.tensor([.8], device='cuda')
    color = torch.full((1, 3), .4, device='cuda')
    view = torch.eye(4, device='cuda')[None]
    K = torch.tensor([[[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]], device='cuda')
    def render(z):
        return rasterization(z, quat, scales, opacity, color, view, K, 64, 64,
                             packed=True, render_mode='RGB+ED')[:2]
    image, alpha = render(means)
    valid = alpha[..., 0].detach() > .01
    assert valid.any(), 'No visible synthetic Gaussian'
    error = (image[..., 3][valid] - 2.).abs().max().item()
    image[..., 3][valid].mean().backward()
    gradient = means.grad[0, 2].item()
    with torch.no_grad():
        delta = torch.tensor([[0., 0., 1e-3]], device='cuda')
        difference = ((render(means + delta)[0][0, 32, 32, 3]
                       - render(means - delta)[0][0, 32, 32, 3]) / .002).item()
    assert error < tolerances['known_Z_absolute_error_max']
    assert abs(gradient - 1.) < tolerances['known_Z_gradient_absolute_error_max']
    assert abs(gradient - difference) < tolerances['known_Z_finite_difference_absolute_error_max']
    chains = []
    for kind, coefficient in [('l1', .01), ('l1', .005), ('cauchy', .01)]:
        means.grad = None
        output, opacity_map = render(means)
        z = output[..., 3]
        target = torch.full((64, 64), .3, device='cuda')
        mask = torch.ones((64, 64), dtype=torch.bool, device='cuda')
        objective, _ = (uniform(z, opacity_map, target, mask, .5) if kind == 'l1'
                        else cauchy(z, opacity_map, target, mask, .5, .1))
        assert (opacity_map.detach() >= .5).any()
        (coefficient * objective).backward()
        expected = -coefficient * .5 * (1. if kind == 'l1' else 8. / 17.)
        actual = means.grad[0, 2].item()
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected),
            rtol=tolerances['synthetic_ED_gradient_cpu_gpu_rtol'],
            atol=tolerances['synthetic_ED_gradient_cpu_gpu_atol'])
        assert torch.isfinite(means.grad).all()
        chains.append({'kind': kind, 'lambda': coefficient, 'analytic_dloss_dmean_Z': expected,
                       'actual_dloss_dmean_Z': actual, 'abs_error': abs(actual - expected)})
    return {'synthetic_resolution': [64, 64], 'expected_Z_max_abs_error': error,
            'Z_derivative': gradient, 'finite_difference': difference, 'loss_raster_chain_checks': chains}


def forbid_operations(stack, torch, strategy, attempts):
    def reject(label):
        def blocked(*args, **kwargs):
            attempts.append(label)
            raise RuntimeError('Forbidden preflight operation: ' + label)
        return blocked
    for name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer) and 'step' in cls.__dict__:
            stack.enter_context(patch.object(cls, 'step', reject(name + '.step')))
    for cls, name in [(torch.optim.lr_scheduler.LRScheduler, 'step'),
                      (strategy, 'step_pre_backward'), (strategy, 'step_post_backward'),
                      (torch, 'save'), (torch, 'load')]:
        stack.enter_context(patch.object(cls, name, reject(getattr(cls, '__name__', 'torch') + '.' + name)))


def run(args, contract, boundary, report):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'src'))
    from fused_ssim import fused_ssim
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
    from geosplat_r.baseline import MemorySampler, load_upstream, load_views, render
    from geosplat_r.depth import load_depth_priors, uniform_inverse_depth_loss
    from geosplat_r.depth_objectives import cauchy_inverse_depth_loss, depth_objective_diagnostics
    from geosplat_r.scene_input import validate_scene_input
    torch.set_num_threads(4)
    sampler = MemorySampler(args.output / 'gpu-memory.csv')
    sampler.start()
    attempts = report['forbidden_operation_attempts']
    try:
        with ExitStack() as stack:
            forbid_operations(stack, torch, DefaultStrategy, attempts)
            total = torch.cuda.get_device_properties(0).total_memory
            free, _ = torch.cuda.mem_get_info()
            assert total <= 12 * 2**30 and free > 1024 * 2**20
            fraction = min(.95, (free - 1024 * 2**20) / total)
            torch.cuda.set_per_process_memory_fraction(fraction)
            torch.cuda.reset_peak_memory_stats()
            report.update(memory_fraction=fraction, initial_free_mib=free / 2**20,
                          torch=torch.__version__, cuda_runtime=torch.version.cuda)
            assert torch.cuda.get_device_name(0) == contract['expected_GPU_name']
            torch.manual_seed(44);np.random.seed(44)
            upstream = load_upstream(args.source)
            upstream.set_random_seed(44)
            views = load_views(args.data)
            manifest = validate_scene_input(args.data, views, args.scene_manifest)
            assert manifest['scene'] == 'leaves'
            for view in views:
                view['pixels'] = view['pixels'].cuda()
            priors, _ = load_depth_priors(args.priors, views, sha(args.data / 'cameras.json'),
                                         sha(args.data / 'points.npz'), sha)
            with np.load(args.data / 'points.npz', allow_pickle=False) as values:
                points, colors = values['points'].copy(), values['rgb'].copy()
            centers = np.array([v['c2w'] for v in views])[:, :3, 3]
            scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
            parser = type('TrainOnlyPoints', (), {'points': points, 'points_rgb': colors})()
            params, optimizers = upstream.create_splats_with_optimizers(parser, scene_scale=scene_scale, device='cuda')
            initial = parameter_digest(params)
            assert initial == contract['expected_initial_parameters_sha256'], 'Initial parameters differ'
            assert all(v.dtype == torch.float32 and v.is_cuda for v in params.values())
            report.update(initial_parameters_sha256=initial, initial_gaussians=len(points),
                          optimizer_objects_created=len(optimizers),
                          expected_view_order_hashes=sampling_hashes(torch),
                          profile={'resolution': [504, 378], 'batch': 1, 'precision': 'float32',
                                   'render_mode': 'RGB+ED', 'seed': 44, 'scale': .1})
            report['synthetic_loss_checks'] = synthetic_loss_checks(torch, uniform_inverse_depth_loss,
                cauchy_inverse_depth_loss, depth_objective_diagnostics, contract['numeric_checks'])
            report['known_depth_check'] = known_depth_check(torch, rasterization, uniform_inverse_depth_loss,
                cauchy_inverse_depth_loss, contract['numeric_checks'])
            probes = []
            # Fixed independent gradient probes, never a training/optimization loop.
            for index, view in enumerate(views):
                prior = priors[index]
                for kind, coefficient in [('l1', .01), ('l1', .005), ('cauchy', .01)]:
                    for value in params.values():
                        value.grad = None
                    image, alpha, _ = render(params, view, degree=3, depth=True)
                    assert image.shape == (1, 378, 504, 4) and image.dtype == torch.float32
                    assert torch.isfinite(image).all() and torch.isfinite(alpha).all()
                    rgb = image[..., :3]
                    rgb_loss = .8 * (rgb - view['pixels'][None]).abs().mean() + .2 * (1 - fused_ssim(
                        rgb.permute(0, 3, 1, 2), view['pixels'][None].permute(0, 3, 1, 2), padding='valid'))
                    objective, fraction_valid = (uniform_inverse_depth_loss(image[..., 3], alpha,
                        prior['target'], prior['mask'], prior['normalizer']) if kind == 'l1'
                        else cauchy_inverse_depth_loss(image[..., 3], alpha, prior['target'],
                            prior['mask'], prior['normalizer'], .1))
                    loss = rgb_loss + coefficient * objective
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values())
                    diagnostics = depth_objective_diagnostics(image[..., 3], alpha, prior['target'],
                        prior['mask'], prior['normalizer'], kind, coefficient, .1)
                    probes.append({'image': view['name'], 'kind': kind, 'lambda': coefficient,
                        'SH_degree': 3, 'RGB_plus_depth_loss': loss.item(),
                        'depth_objective': objective.item(), 'valid_fraction': fraction_valid.item(),
                        'diagnostics': diagnostics, 'finite_gradients_all_parameters': True})
            report['real_train_gradient_probes'] = probes
            assert parameter_digest(params) == initial, 'Preflight mutated parameters'
            assert all(not optimizer.state for optimizer in optimizers.values()), 'Optimizer state mutated'
            report.update(final_parameters_sha256=initial, optimizer_state_entries=0,
                          parameter_mutation=False)
            assert not attempts
            assert boundary.reads == {str(Path(v['absolute_path']).resolve()) for v in contract['assets']}
            assert not boundary.rejections
            torch.cuda.synchronize()
    finally:
        report.update(torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                      **sampler.stop())
        temporary = getattr(sys.modules.get('torch.distributed.nn.jit.instantiator'), '_TEMP_DIR', None)
        if temporary is not None:
            temporary.cleanup()


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'protocol': 'leaves-robust-depth-zero-update-v1', 'status': 'failed',
              'optimizer_updates': 0, 'scheduler_updates': 0, 'densification_updates': 0,
              'checkpoint_loads': 0, 'checkpoint_saves': 0, 'formal_training': False,
              'evaluation': False, 'forbidden_operation_attempts': []}
    started = time.perf_counter();boundary = None
    try:
        contract = json.loads(args.contract.read_text())
        validate_contract(args, contract)
        boundary = ReadBoundary(contract['protected_asset_root'],
            [v['absolute_path'] for v in contract['assets']], contract['denied_roots'])
        sys.addaudithook(boundary)
        verify_files(contract)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        before = gpu_identity()
        assert before['name'] == contract['expected_GPU_name']
        report.update(boot_id=boot, GPU_before=before, contract_sha256=sha(args.contract))
        run(args, contract, boundary, report)
        verify_files(contract)
        after = gpu_identity()
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
        assert all(before[k] == after[k] for k in ['uuid', 'name', 'driver', 'total_mib'])
        assert report['sampler_error'] is None and report['samples'] > 0
        assert report['whole_board_sampled_peak_mib'] < 12 * 1024
        report.update(status='passed', GPU_after=after)
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['process_seconds'] = time.perf_counter() - started
        if boundary is not None:
            report['actual_asset_reads'] = sorted(boundary.reads)
            report['forbidden_asset_or_network_attempts'] = boundary.rejections
        (args.output / 'preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
