"""Pure-Python option validation; reject invalid profiles before CUDA imports."""
import math


def validate_depth_options(kind, scale, coefficient, has_priors, diagnostics=False):
    if kind not in ('l1', 'cauchy'):
        raise ValueError('Unknown depth loss; expected l1 or cauchy')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Cauchy scale must be finite and positive')
    if not float.fromhex('0x1.0p-126') <= scale <= float.fromhex('0x1.fffffep127'):
        raise ValueError('Cauchy scale must be in the normalized float32 range')
    if not math.isfinite(coefficient) or not 0 <= coefficient < 1:
        raise ValueError('Depth coefficient must be finite and in [0,1)')
    if not has_priors and (coefficient > 0 or kind != 'l1' or diagnostics):
        raise ValueError('Depth objective/diagnostics require explicit depth priors')
