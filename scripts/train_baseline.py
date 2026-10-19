"""Train the authorized fern 3-view vanilla baseline on explicit train-only assets."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.baseline import train

if __name__=='__main__':
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
    args=ap.parse_args()
    if args.max_steps<1 or (args.stop_after is not None and not 0<args.stop_after<=args.max_steps):
        ap.error('Invalid step count')
    if not 0<=args.depth_lambda<1 or args.depth_start<0:
        ap.error('Invalid depth weight or start')
    if args.depth_lambda>0 and args.depth_priors is None:
        ap.error('Positive depth weight requires explicit depth priors')
    train(args)
