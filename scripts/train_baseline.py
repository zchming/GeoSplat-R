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
    args=ap.parse_args()
    if args.max_steps<1 or (args.stop_after is not None and not 0<args.stop_after<=args.max_steps):
        ap.error('Invalid step count')
    train(args)
