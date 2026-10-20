"""Baseline entry with explicit train-only assets and optional depth objectives."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.depth_options import validate_depth_options


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max-steps',type=int,default=30000)
    ap.add_argument('--stop-after',type=int)
    ap.add_argument('--depth-priors',type=Path,help='Explicit frozen train-only aligned priors; enables RGB+ED in both controls')
    ap.add_argument('--depth-lambda',type=float,default=0.)
    ap.add_argument('--depth-start',type=int,default=1000)
    ap.add_argument('--scene-manifest',type=Path,help='Explicit train/input-manifest.json for approved development scenes; omit for legacy fern')
    ap.add_argument('--depth-loss',choices=['l1','cauchy'],default='l1')
    ap.add_argument('--depth-cauchy-scale',type=float,default=.1)
    ap.add_argument('--depth-diagnostics',action='store_true',help='Detached objective diagnostics; automatically enabled for Cauchy')
    args=ap.parse_args(argv)
    if args.max_steps<1 or (args.stop_after is not None and not 0<args.stop_after<=args.max_steps):
        ap.error('Invalid step count')
    if args.depth_start<0:
        ap.error('Invalid depth weight or start')
    try:
        validate_depth_options(args.depth_loss,args.depth_cauchy_scale,args.depth_lambda,
                               args.depth_priors is not None,args.depth_diagnostics)
    except ValueError as exc:
        ap.error(str(exc))
    return args


if __name__=='__main__':
    args=parse_args()
    from geosplat_r.baseline import train
    train(args)
