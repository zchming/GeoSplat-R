"""Portable generated-data geometry regressions, without models or real assets."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULES = ['test_sensor_geometry', 'test_raw_track_diagnostics', 'test_raw_track_evidence',
           'test_relative_diagnostics', 'test_relative_io', 'test_delivery_relative_io',
           'test_delivery_execution_entry', 'test_ordinal_margin', 'test_margin_confirmation']
EXPECTED_SKIPS = {
    'test_raw_track_evidence.CompletedRawTrackEvidence.test_input_sources_and_single_frozen_execution',
    'test_raw_track_evidence.CompletedRawTrackEvidence.test_complete_records_grouping_nulls_and_declared_gates'}


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'gsplat', 'fused_ssim', 'depth_anything_v2'}:
            raise RuntimeError('Model/CUDA imports forbidden in geometry CPU CI: '+fullname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(k.startswith('GEOSPLAT_') for k in os.environ):
        raise RuntimeError('Unset real evidence/GPU settings for portable CI')
    sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'src')]
    sys.meta_path.insert(0, NoModels())
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    skips = [dict(test=t.id(), reason=reason) for t, reason in result.skipped]
    success = result.wasSuccessful() and {x['test'] for x in skips} == EXPECTED_SKIPS
    report = dict(successful=success, tests_run=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(skips),
        failures=len(result.failures), errors=len(result.errors), skipped=skips,
        elapsed_seconds=time.monotonic()-started, scope=MODULES,
        model_imported=False, real_data_mounted=False,
        excluded='Actual datasets, models, GPU preflight/inference and completed-run evidence')
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report))
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
