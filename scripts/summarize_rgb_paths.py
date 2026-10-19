"""Validate the three new RGB-only runs against all six preserved depth controls."""
import argparse
import json
import sys
from pathlib import Path
from summarize_depth_seeds import collect,sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.rgb_path_summary import summarize_paths


def collect_paths(manifest):
    depth=collect(json.loads(Path(manifest['depth_manifest']).read_text()))
    if sorted(r['seed'] for r in manifest['rgb_runs'])!=[42,43,44]:raise ValueError('Three authorized RGB-only seeds required')
    rows=[];base={p['seed']:p for p in depth['pairs']}
    expected={'cameras.json','points.npz','images/image001.png','images/image010.png','images/image019.png'}
    for item in sorted(manifest['rgb_runs'],key=lambda r:r['seed']):
        run=Path(item['run']);evaluation=Path(item['evaluation'])
        c=json.loads((run/'config.json').read_text());r=json.loads((run/'resources.json').read_text())
        m=json.loads((evaluation/'metrics.json').read_text());seed=item['seed'];paired=base[seed]
        if c['seed']!=seed or c['depth_loss'] or 'depth_protocol' in c:raise ValueError('Expected RGB-only configuration')
        if {k:c[k] for k in depth['frozen_core']}!=depth['frozen_core']:raise ValueError('RGB-only frozen configuration/input mismatch')
        if r['status']!='passed' or r['optimizer_updates']!=30000 or m['optimizer_updates']!=30000:
            raise ValueError('Incomplete RGB-only run')
        if r['initial_parameters_sha256']!=paired['initial_parameters_sha256']:raise ValueError('Matched initial parameter mismatch')
        if sha(run/'checkpoint.pt')!=m['checkpoint_sha256']:raise ValueError('RGB-only evaluation checkpoint mismatch')
        if sha(run/'source/baseline.py')!=c['training_implementation_sha256']:raise ValueError('RGB-only source mismatch')
        if set(json.loads((run/'training-access.json').read_text()))!=expected:raise ValueError('Unexpected RGB training asset access')
        if m['metric_protocol']!=depth['metric_protocol'] or m['versions']!=depth['versions']:raise ValueError('Metric protocol/version mismatch')
        gt={name:sha(evaluation/'test'/Path(name).stem/'gt.png') for name in depth['test_gt_png_sha256']}
        if gt!=depth['test_gt_png_sha256']:raise ValueError('RGB-only test GT differs')
        if r['whole_board_sampled_peak_mib']>=12282 or r['sampler_error'] is not None:raise ValueError('RGB-only resource checks failed')
        rows.append({'seed':seed,'initial_parameters_sha256':r['initial_parameters_sha256'],
            'rgb_only':{'train':m['train'],'test':m['test'],'resources':r,'checkpoint_sha256':m['checkpoint_sha256'],
                'metrics_sha256':sha(evaluation/'metrics.json')},'rgb_ed_zero':paired['zero'],'uniform':paired['uniform']})
    return {'scope':'Frozen fern RGB-only path diagnostic; 3 paths x 3 seeds x 30k',
        'rows':rows,**summarize_paths(rows),'frozen_core':depth['frozen_core'],
        'test_gt_png_sha256':depth['test_gt_png_sha256'],'metric_protocol':depth['metric_protocol'],
        'versions':depth['versions'],'depth_manifest_sha256':sha(Path(manifest['depth_manifest']))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=collect_paths(json.loads(args.manifest.read_text()));result['manifest_sha256']=sha(args.manifest)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['arm_aggregates']['test'],indent=2))
    for name,value in result['contrasts'].items():print(name,json.dumps(value['splits']['test']))


if __name__=='__main__':main()
