"""Synthetic gsplat preflight only; no dataset, pretrained weights or benchmark."""
import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--gaussians', type=int, default=4096)
    ap.add_argument('--width', type=int, default=320)
    ap.add_argument('--height', type=int, default=240)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if min(args.gaussians, args.width, args.height, args.steps) < 1:
        ap.error('sizes and steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        result = run(args)
        result['status'] = 'passed'
    except Exception as exc:
        result = {'status': 'failed', 'exception': repr(exc)}
        raise
    finally:
        result['process_wall_seconds'] = time.perf_counter() - started
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


def run(args):
    import torch
    import gsplat
    from PIL import Image
    from gsplat.rendering import rasterization

    assert torch.cuda.is_available(), 'CUDA is unavailable'
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.manual_seed(args.seed)
    device = 'cuda'
    def board_memory():
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], text=True).strip()
    baseline_board_mib = board_memory()
    view = torch.eye(4, device=device)[None]
    K = torch.tensor([[[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]], device=device)
    means = torch.tensor([[0., 0., 2.]], device=device, requires_grad=True)
    quat = torch.tensor([[1., 0., 0., 0.]], device=device)
    scale = torch.full((1, 3), .1, device=device)
    opa = torch.tensor([.8], device=device)
    color = torch.full((1, 3), .4, device=device)
    def analytic(z):
        return rasterization(z, quat, scale, opa, color, view, K, 64, 64,
                             packed=True, render_mode='RGB+ED')[:2]
    image, alpha = analytic(means)
    mask = alpha[..., 0].detach() > .01
    assert mask.any(), 'No visible Gaussian'
    expected = image[..., 3][mask]
    depth_error = (expected - 2.).abs().max().item()
    assert depth_error < 1e-4, depth_error
    expected.mean().backward()
    gradient = means.grad[0, 2].item()
    assert abs(gradient - 1.) < 1e-3, gradient
    with torch.no_grad():
        delta = torch.tensor([[0., 0., 1e-3]], device=device)
        plus = analytic(means + delta)[0][0, 32, 32, 3]
        minus = analytic(means - delta)[0][0, 32, 32, 3]
        finite_difference = ((plus - minus) / .002).item()
    assert abs(gradient - finite_difference) < 2e-3, finite_difference
    del image, alpha, expected
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    n = args.gaussians
    xyz = torch.rand(n, 3, device=device)
    xyz[:, :2] = (xyz[:, :2] - .5) * 3.
    xyz[:, 2] = 2. + xyz[:, 2] * 2.
    params = {
        'means': torch.nn.Parameter(xyz),
        'quats': torch.nn.Parameter(torch.tensor([1., 0., 0., 0.], device=device).repeat(n, 1)),
        'log_scales': torch.nn.Parameter(torch.full((n, 3), -4.5, device=device)),
        'opacity_logits': torch.nn.Parameter(torch.full((n,), -2., device=device)),
        'sh': torch.nn.Parameter(torch.randn(n, 16, 3, device=device) * .05),
    }
    optim = torch.optim.Adam(params.values(), lr=1e-3)
    w, h = args.width, args.height
    K = torch.tensor([[[w*.8, 0., w/2], [0., w*.8, h/2], [0., 0., 1.]]], device=device)
    durations, losses = [], []
    for step in range(args.steps + 3):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        rendered, alpha, _ = rasterization(
            params['means'], params['quats'], params['log_scales'].exp(),
            params['opacity_logits'].sigmoid(), params['sh'], view, K, w, h,
            packed=True, render_mode='RGB+ED', sh_degree=3)
        loss = (rendered[..., :3] - .25).square().mean()
        loss.backward()
        for name, param in params.items():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        optim.step()
        torch.cuda.synchronize()
        if step >= 3:
            durations.append(time.perf_counter() - tick)
            losses.append(loss.item())
    assert all(torch.isfinite(x).all() for x in (rendered, alpha))
    rgb = rendered[0, ..., :3].detach().clamp(0, 1)
    Image.fromarray((rgb.cpu().numpy() * 255).astype('uint8')).save(args.output/'synthetic-rgb.png')
    result = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'torch_cuda_runtime': torch.version.cuda, 'gsplat': gsplat.__version__,
        'device': torch.cuda.get_device_name(),
        'compute_capability': list(torch.cuda.get_device_capability()),
        'profile': {'gaussians': n, 'width': w, 'height': h, 'batch': 1,
                    'steps_measured': args.steps, 'warmup_steps': 3, 'sh_degree': 3,
                    'packed': True, 'precision': 'float32', 'seed': args.seed,
                    'loss': 'synthetic RGB MSE', 'densification': False},
        'checks': {'expected_depth_max_error': depth_error, 'depth_z_gradient': gradient,
                   'depth_z_finite_difference': finite_difference,
                   'finite_gradients_all_parameters': True},
        'mean_step_ms': statistics.mean(durations) * 1000,
        'median_step_ms': statistics.median(durations) * 1000,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'board_memory_before_mib': baseline_board_mib,
        'board_memory_after_mib': board_memory(),
        'first_measured_loss': losses[0], 'last_measured_loss': losses[-1],
        'scope': 'Synthetic CUDA execution only; no NVS quality or full training cost claim.'}
    return result


if __name__ == '__main__':
    main()
