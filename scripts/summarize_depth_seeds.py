"""Validate six completed runs and aggregate the authorized three-seed depth pairs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geosplat_r.seed_summary import summarize_pairs


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(manifest):
    if sorted(p['seed'] for p in manifest['pairs'])!=[42,43,44]:raise ValueError('Require authorized seed42/43/44 pairs')
    reference=None;reference_depth=None;metric_protocol=None;versions=None;gt_hashes=None;pairs=[]
    fixed=['upstream','train_images','cameras_sha256','points_sha256','max_steps','executed_steps','resolution',
        'scene_scale','strategy','packed','precision','sh_degree','sh_degree_interval','rgb_loss','batch_size',
        'near_plane','far_plane','camera_optimization','appearance_optimization','image_cache','training_implementation_sha256']
    expected_access={'cameras.json','points.npz','images/image001.png','images/image010.png','images/image019.png'}
    scene_input=None;view_names=None
    if manifest.get('scene_input_manifest'):
        from geosplat_r.scene_input import validate_scene_input,SCENE_TESTS
        input_path=Path(manifest['scene_input_manifest'])
        views=json.loads((input_path.parent/'cameras.json').read_text())
        scene_input=validate_scene_input(input_path.parent,views,input_path)
        view_names={'train':[v['name'] for v in views],'test':SCENE_TESTS[scene_input['scene']]}
        expected_access={'cameras.json','points.npz','input-manifest.json',*[f'images/{n}' for n in view_names['train']]}
        fixed=fixed+['scene_input']
    for item in sorted(manifest['pairs'],key=lambda p:p['seed']):
        pair={'seed':item['seed']};initial=[]
        for arm,weight in [('zero',0.),('uniform',.01)]:
            run=Path(item[arm]['run']);evaluation=Path(item[arm]['evaluation'])
            config=json.loads((run/'config.json').read_text());resources=json.loads((run/'resources.json').read_text())
            metrics=json.loads((evaluation/'metrics.json').read_text())
            gt={name:sha(evaluation/'test'/Path(name).stem/'gt.png') for name in (view_names['test'] if view_names else ['image000.png','image008.png','image016.png'])}
            if gt_hashes is None:gt_hashes=gt
            elif gt!=gt_hashes:raise ValueError('Evaluation test GT differs across runs')
            if config['seed']!=item['seed'] or config['depth_protocol']['lambda']!=weight:
                raise ValueError('Seed or depth coefficient mismatch')
            if config['depth_protocol']['start_zero_based_step']!=1000 or config['depth_loss']!=(weight>0):
                raise ValueError('Frozen depth schedule mismatch')
            if resources['status']!='passed' or resources['optimizer_updates']!=30000 or metrics['optimizer_updates']!=30000:
                raise ValueError('Incomplete or failed run cannot be silently omitted')
            if config['max_steps']!=30000 or config['executed_steps']!=30000:raise ValueError('Training budget changed')
            if sha(run/'checkpoint.pt')!=metrics['checkpoint_sha256']:raise ValueError('Evaluation checkpoint mismatch')
            if sha(run/'source/baseline.py')!=config['training_implementation_sha256']:raise ValueError('Training source hash mismatch')
            if sha(run/'source/depth.py')!=config['depth_protocol']['depth_implementation_sha256']:raise ValueError('Depth source hash mismatch')
            if scene_input is not None:
                if config['scene_input']['input']!=scene_input or config['scene_input']['manifest_sha256']!=sha(input_path):
                    raise ValueError('Run scene input differs from the frozen manifest')
                if sha(run/'source/scene_input.py')!=config['scene_input']['adapter_sha256']:
                    raise ValueError('Scene adapter source mismatch')
            if set(json.loads((run/'training-access.json').read_text()))!=expected_access:raise ValueError('Unexpected training asset access')
            if resources['whole_board_sampled_peak_mib']>=12282 or resources['sampler_error'] is not None:
                raise ValueError('Resource checks failed')
            core={k:config[k] for k in fixed};depth={k:v for k,v in config['depth_protocol'].items() if k!='lambda'}
            if reference is None:reference=core;reference_depth=depth;metric_protocol=metrics['metric_protocol'];versions=metrics['versions']
            elif core!=reference or depth!=reference_depth:raise ValueError('Frozen inputs, code or protocol differ across runs')
            if metrics['metric_protocol']!=metric_protocol or metrics['versions']!=versions:raise ValueError('Evaluation metric protocol/version mismatch')
            initial.append(resources['initial_parameters_sha256'])
            pair[arm]={'train':metrics['train'],'test':metrics['test'],'resources':resources,
                'checkpoint_sha256':metrics['checkpoint_sha256'],'metrics_sha256':sha(evaluation/'metrics.json')}
        if initial[0]!=initial[1]:raise ValueError('Pair initial parameters differ')
        pair['initial_parameters_sha256']=initial[0];pairs.append(pair)
    summary=summarize_pairs(pairs,view_names)
    return {'scope':f"Phase 2 frozen {scene_input['scene'] if scene_input else 'fern'} uniform-depth seed stability, 42/43/44, 30k",
        'pairs':pairs,**summary,'frozen_core':reference,'depth_protocol_except_lambda':reference_depth,
        'metric_protocol':metric_protocol,'versions':versions,'test_gt_png_sha256':gt_hashes,
        **({'view_names':view_names} if view_names else {})}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True,help='Explicit run/evaluation locations for each arm and seed')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=collect(json.loads(args.manifest.read_text()));result['input_manifest_sha256']=sha(args.manifest)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['aggregates'],indent=2))


if __name__=='__main__':main()
