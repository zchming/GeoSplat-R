"""Summarize only after four full training runs and independent evaluations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geosplat_r.robust_pilot_summary import ARMS, METRICS, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest['training_complete_before_any_evaluation'] is not True:
        raise ValueError('All four must complete before any evaluation')
    if [r['id'] for r in manifest['runs']] != list(ARMS):
        raise ValueError('Frozen arm order required')
    records = {}
    for run in manifest['runs']:
        if run['training_status'] != 'passed' or run['optimizer_updates'] != 30000:
            raise ValueError('Incomplete training run')
        raw = Path(run['metrics']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != run['metrics_sha256']:
            raise ValueError('Metric file changed')
        record = json.loads(raw)
        if record['checkpoint_sha256'] != run['checkpoint_sha256']:
            raise ValueError('Evaluation checkpoint identity differs')
        records[run['id']] = record
    result = summarize(records)
    args.output.mkdir(parents=True, exist_ok=False)
    result['manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'all-pairwise-per-image.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['contrast', 'split', 'image', *METRICS, 'regressed_metrics'])
        for contrast, splits in result['all_pairwise_contrasts'].items():
            for split, values in splits.items():
                for row in values['per_image']:
                    writer.writerow([contrast, split, row['image'],
                        *[row['delta'][m] for m in METRICS], ';'.join(row['regressed_metrics'])])
    print(json.dumps(result['prospective_test_direction_gate'], indent=2))


if __name__ == '__main__':
    main()
