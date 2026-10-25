"""Optional Cauchy objective and detached diagnostics; no geometry weights."""
import math

import torch

def _positive_scalar(value, label, dtype):
    value = float(value)
    limits = torch.finfo(dtype)
    if not math.isfinite(value) or not limits.tiny <= value <= limits.max:
        raise ValueError(label + ' must be positive, finite and representable in the tensor dtype')
    return value


def cauchy_penalty(residual, scale=.1):
    """c log(1+(r/c)^2); stable equivalent branches, no clipping/epsilon."""
    scale = _positive_scalar(scale, 'Cauchy scale', residual.dtype)
    magnitude = residual.abs()
    small = magnitude <= scale
    small_r = torch.where(small, residual, torch.zeros_like(residual))
    large_r = torch.where(small, torch.full_like(residual, scale), magnitude)
    low = scale * torch.log1p((small_r / scale).square())
    high = scale * (2 * (large_r.log() - math.log(scale))
                    + torch.log1p((scale / large_r).square()))
    return torch.where(small, low, high)


def _residuals(z, alpha, target, mask, normalizer):
    z, alpha = z.squeeze(), alpha.squeeze()
    normalizer = _positive_scalar(normalizer, 'Normalizer', z.dtype)
    valid = mask & (alpha.detach() >= .5) & torch.isfinite(z) & (z.detach() > 1e-6)
    safe_z = torch.where(valid, z, torch.ones_like(z))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    residual = (safe_z.reciprocal() - safe_target) / normalizer
    return residual, valid, safe_z, normalizer


def cauchy_inverse_depth_loss(z, alpha, target, mask, normalizer, scale=.1):
    residual, valid, _, _ = _residuals(z, alpha, target, mask, normalizer)
    count = valid.sum()
    loss = torch.where(valid, cauchy_penalty(residual, scale), torch.zeros_like(residual)).sum() / count.clamp_min(1)
    return loss, count.to(z.dtype) / mask.numel()


@torch.no_grad()
def depth_objective_diagnostics(z, alpha, target, mask, normalizer, kind,
                                coefficient, scale=.1):
    if kind not in ('l1', 'cauchy'):
        raise ValueError('Unknown diagnostic objective')
    residual, valid, safe_z, normalizer = _residuals(z, alpha, target, mask, normalizer)
    values = residual[valid]
    count = int(values.numel())
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite valid residual in objective diagnostics')
    if kind == 'cauchy':
        scale = _positive_scalar(scale, 'Cauchy scale', residual.dtype)
        # Stable 2t/(1+t^2): use reciprocal t on the large branch.
        magnitude = values.abs()
        small = magnitude <= scale
        small_t = torch.where(small, magnitude, torch.zeros_like(magnitude)) / scale
        large_t = scale / torch.where(small, torch.full_like(magnitude, scale), magnitude)
        influence = torch.where(small, 2 * small_t / (1 + small_t.square()),
                                2 * large_t / (1 + large_t.square()))
        objective = cauchy_penalty(values, scale)
    else:
        influence = (values != 0).to(values.dtype)
        objective = values.abs()
    # Derivative of the valid-pixel MEAN objective, after lambda, w.r.t. ED.
    gradient = coefficient * influence / max(count, 1) / normalizer / safe_z[valid].square()
    empty = count == 0
    quantiles = (torch.quantile(values.abs(), torch.tensor([.5, .9, .99], device=values.device, dtype=values.dtype)).tolist()
                 if count else [None, None, None])
    return {'depth_objective_kind': kind, 'depth_objective': float(objective.mean()) if count else 0.,
            'depth_normalized_l1': float(values.abs().mean()) if count else 0.,
            'depth_valid_count': count, 'depth_valid_fraction': count / mask.numel(),
            'depth_weight': float(coefficient), 'depth_cauchy_scale': scale if kind == 'cauchy' else None,
            'depth_abs_residual_p50': quantiles[0], 'depth_abs_residual_p90': quantiles[1],
            'depth_abs_residual_p99': quantiles[2],
            'depth_influence_mean': float(influence.mean()) if count else None,
            'depth_influence_min': float(influence.min()) if count else None,
            'depth_influence_max': float(influence.max()) if count else None,
            'depth_weighted_abs_dloss_dED_sum': float(gradient.sum()),
            'depth_weighted_abs_dloss_dED_mean': float(gradient.mean()) if not empty else None}
