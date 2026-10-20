"""Run portable CPU regressions, with explicit exclusions for external evidence."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'test_robust_pilot_summary', 'test_depth_gpu_preflight', 'test_robust_depth',
    'test_scene_input', 'test_train_diagnostics', 'test_crossview_diagnostics',
    'test_seed_summary', 'test_rgb_paths',
)
EXPECTED_SKIPS = {
    'test_scene_input.RealDevelopmentPreparation.test_prepared_inputs_splits_and_scene_geometry_have_auditable_hashes',
    'test_scene_input.RealDevelopmentPreparation.test_train_priors_and_gpu_preflight_use_no_test_assets_or_updates',
    'test_train_diagnostics.RealD1Evidence.test_complete_proxy_evidence_and_cpu_resource_boundary',
    'test_crossview_diagnostics.RealCrossviewEvidence.test_complete_output_cpu_and_unknown_records',
    'test_seed_summary.RealSeedSummary.test_all_six_run_provenance_and_same_gt_are_validated',
    'test_rgb_paths.RealPathEvidence.test_all_nine_runs_and_matched_initialization_have_valid_provenance',
}


class NoTrainingExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gsplat', 'fused_ssim'}:
            raise RuntimeError('Training extension import forbidden in CPU CI: ' + fullname)


EXCLUDED_TEST = ('test_robust_depth.DepthEntryAndWiring.'
                 'test_existing_output_is_refused_before_cuda_or_input_reads')


def portable_suite(suite):
    result = unittest.TestSuite()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            result.addTests(portable_suite(test))
        elif test.id() != EXCLUDED_TEST:
            result.addTest(test)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON report path; existing files are refused.')
    parser.add_argument('--group', choices=['depth', 'protocol'], required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(key.startswith('GEOSPLAT_') for key in os.environ):
        raise RuntimeError('Unset GEOSPLAT_* evidence/GPU settings for portable CPU CI')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'src')]
    blocker = NoTrainingExtensions()
    sys.meta_path.insert(0, blocker)
    torch = None
    if args.group == 'depth':
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA was initialized before CPU checks')
        torch.set_num_threads(4)
    modules = MODULES[:4] if args.group == 'depth' else MODULES[4:]
    expected_skips = {test for test in EXPECTED_SKIPS
                      if test.split('.')[0] in modules}
    started = time.perf_counter()
    guard = (patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CUDA forbidden'))
             if torch else nullcontext(None))
    with guard as cuda:
        suite = portable_suite(unittest.defaultTestLoader.loadTestsFromNames(modules))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    cuda_attempts = cuda.call_count if cuda else 0
    cuda_initialized = torch.cuda.is_initialized() if torch else False
    skipped = [{'test': test.id(), 'reason': reason} for test, reason in result.skipped]
    actual_skips = {item['test'] for item in skipped}
    successful = (result.wasSuccessful() and actual_skips == expected_skips
                  and result.testsRun == (42 if args.group == 'depth' else 43)
                  and cuda_attempts == 0 and not cuda_initialized)
    report = {
        'successful': successful, 'device': 'CPU', 'torch': torch.__version__ if torch else None, 'group': args.group,
        'tests_run': result.testsRun,
        'passed': result.testsRun - len(result.failures) - len(result.errors) - len(skipped),
        'failures': len(result.failures), 'errors': len(result.errors), 'skipped': skipped,
        'skip_inventory_matches': actual_skips == expected_skips,
        'cuda_initialization_attempts': cuda_attempts,
        'cuda_initialized': cuda_initialized,
        'elapsed_seconds': time.perf_counter() - started,
        'scope': list(modules),
        'excluded': ['CUDA rasterizer parity and training',
                     'external data, priors, checkpoints and completed-run evidence',
                     'legacy suites importing the CUDA baseline at module scope',
                     EXCLUDED_TEST + ' (imports CUDA baseline/extensions)'],
    }
    # Torch 2.4.1 owns a JIT temporary directory even in CPU-only processes.
    module = sys.modules.get('torch.distributed.nn.jit.instantiator')
    if module is not None and hasattr(module, '_TEMP_DIR'):
        module._TEMP_DIR.cleanup()
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
    return 0 if successful else 1


if __name__ == '__main__':
    sys.exit(main())
