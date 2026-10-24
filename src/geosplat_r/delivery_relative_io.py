"""Independent delivery_area identity/seal adapter; frozen ordinal math unchanged."""
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('legacy_relative_io', Path(__file__).with_name('relative_io.py'))
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
rd = legacy.rd
PLAN_SHA = '8d8ad3a455aa7f454dc4ccc88204ecbb4edc2d1e0c69a4432eed2cc96f220ed1'
PROTOCOL = 'delivery-area-relative-adapter-v1'
NAMES = ['dslr_images/DSC_0675.JPG', 'dslr_images/DSC_0696.JPG', 'dslr_images/DSC_0718.JPG']


def load_contexts(plan_path, manifest_path, base_path, review_path, sources):
    if rd.digest(plan_path) != PLAN_SHA:
        raise ValueError('Delivery adapter plan drift')
    plan = json.loads(Path(plan_path).read_text())
    paths = [('input_manifest_sha256', manifest_path), ('base_plan_sha256', base_path),
             ('review_plan_sha256', review_path)]
    for key, path in paths:
        if rd.digest(path) != plan[key]:
            raise ValueError('Frozen identity drift: '+key)
    manifest, base, review = [json.loads(Path(p).read_text()) for p in [manifest_path, base_path, review_path]]
    if (manifest['dataset']['selected_names'] != NAMES or manifest['dataset']['selected_ranks'] != [0,21,43]
            or manifest['dataset']['native_wh'] != [6048,4032]
            or manifest['numeric_core_sha256'] != plan['numerical_core_sha256']
            or review['unchanged_protocol']['semantic_sections_sha256'] != plan['inherited_semantic_sections_sha256']
            or sources.get('relative_diagnostics.py') != plan['numerical_core_sha256']
            or sources.get('relative_io.py') != plan['legacy_IO_sha256']):
        raise ValueError('Frozen scene/core identity mismatch')
    # Source digest of the complete base plan binds every numerical rule and model file.
    predictor_inputs = {}
    labels = []
    for row in manifest['inputs']:
        if row['role'] == 'predictor_rgb':
            predictor_inputs['rgb/'+row['image']] = {k:row[k] for k in ['bytes','sha256']}
        elif row['role'] == 'predictor_cameras':
            predictor_inputs['cameras.json'] = {k:row[k] for k in ['bytes','sha256']}
        elif row['role'] == 'evaluator_GT':
            labels.append({k:row[k] for k in ['image','bytes','sha256']})
        else:
            raise ValueError('Unknown asset role')
    if (len(manifest['inputs']) != 7 or len(predictor_inputs) != 4
            or [r['image'] for r in labels] != NAMES):
        raise ValueError('Expected four predictor assets and three labels')
    provenance = dict(sources)
    provenance.update({'model/'+r['path']:r['sha256'] for r in base['prediction']['model']['source_files']})
    shared = dict(protocol=PROTOCOL, plan_sha256=PLAN_SHA,
                  input_manifest_sha256=plan['input_manifest_sha256'],
                  base_plan_sha256=plan['base_plan_sha256'], names=NAMES.copy(),
                  native_wh=[6048,4032], work_wh=[1008,672], source_sha256=provenance,
                  input_sha256={k:v['sha256'] for k,v in predictor_inputs.items()})
    predictor = dict(**shared, inputs=predictor_inputs, model=base['prediction']['model'],
                     camera_records=manifest['dataset']['selected_camera_records'])
    evaluator = dict(**shared, labels=labels)
    return predictor, evaluator


def admit_predictor(data, config):
    if (config['protocol'] != PROTOCOL or config['plan_sha256'] != PLAN_SHA
            or config['names'] != NAMES or config['native_wh'] != [6048,4032]
            or config['work_wh'] != [1008,672]):
        raise ValueError('Wrong delivery predictor context')
    views = legacy.admit_predictor(data, config)
    if views != config['camera_records']:
        raise ValueError('Selected camera records mismatch')
    return views


def seal_predictions(root, config):
    """Seal an existing fresh three-output directory; never produces predictions."""
    if config['protocol'] != PROTOCOL or config['plan_sha256'] != PLAN_SHA or config['names'] != NAMES:
        raise ValueError('Wrong delivery seal context')
    root = Path(root)
    if (root.is_symlink() or not root.is_dir()
            or {p.name for p in root.iterdir()} != {'0.npy','1.npy','2.npy'}):
        raise ValueError('Exactly three unsealed raw outputs required')
    rows = []
    for i, name in enumerate(config['names']):
        path = root/('%d.npy'%i)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Unsafe prediction path')
        q = np.load(path, allow_pickle=False)
        if q.dtype != np.float32 or q.shape != tuple(reversed(config['work_wh'])):
            raise ValueError('Raw prediction schema mismatch')
        rows.append(dict(name=name, file=path.name, sha256=rd.digest(path)))
    payload = {k:config[k] for k in ['protocol','plan_sha256','input_manifest_sha256',
                                   'base_plan_sha256','source_sha256','input_sha256']}
    rd.write_json(root/'seal.json', dict(status='sealed_all_predictions', predictions=rows, **payload))


def sealed_predictions(root, config):
    root = Path(root)
    if (root.is_symlink() or not root.is_dir()
            or {p.name for p in root.iterdir()} != {'0.npy','1.npy','2.npy','seal.json'}
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError('Unsafe prediction root')
    seal = json.loads((root/'seal.json').read_text())
    keys = ['protocol','plan_sha256','input_manifest_sha256','base_plan_sha256','source_sha256','input_sha256']
    if (config['protocol'] != PROTOCOL or config['plan_sha256'] != PLAN_SHA or config['names'] != NAMES
            or seal.get('status') != 'sealed_all_predictions'
            or any(seal.get(k) != config[k] for k in keys)
            or [r['name'] for r in seal.get('predictions',[])] != config['names']):
        raise ValueError('Wrong scene or seal provenance')
    names = [r['file'] for r in seal['predictions']]
    if (names != ['0.npy','1.npy','2.npy']
            or {p.name for p in root.iterdir()} != set(names)|{'seal.json'}
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError('Prediction inventory mismatch')
    arrays = []
    for row in seal['predictions']:
        path = root/row['file']
        if rd.digest(path) != row['sha256']:
            raise ValueError('Prediction digest mismatch')
        q = np.load(path, allow_pickle=False)
        if q.dtype != np.float32 or q.shape != tuple(reversed(config['work_wh'])):
            raise ValueError('Raw prediction schema mismatch')
        arrays.append(q)
    return arrays


def evaluate_sealed(predictions, labels, config, output, access):
    """Callable CPU bridge, tested on synthetic raw labels only in this stage."""
    arrays = sealed_predictions(predictions, config)
    if len(config['labels']) != 3 or [r['image'] for r in config['labels']] != config['names']:
        raise ValueError('Three ordered mapped labels required')
    # Check all names before even the first reference metadata/payload access.
    for name in config['names']:
        if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name:
            raise ValueError('Unsafe reference mapping')
    output = Path(output).resolve()
    for input_root in [Path(predictions).resolve(), Path(labels).resolve()]:
        if output == input_root or input_root in output.parents or output in input_root.parents:
            raise ValueError('Disjoint evaluation output required')
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for i, (name, row, q) in enumerate(zip(config['names'], config['labels'], arrays)):
        access.append(name)
        gt = legacy.native_reference(Path(labels)/name, row, config['native_wh'])
        report, nodes, groups = rd.evaluate_arrays(gt, q, i)
        rd.write_json(output/('image-%d.json'%i), report)
        payload = dict(native_indices=nodes['indices'], reference_Z=nodes['z'], cell=nodes['cell'])
        for kind in ['local','cross']:
            payload[kind+'_candidate_pairs'] = np.concatenate([c['candidates'] for c in groups[kind]])
            payload[kind+'_eligible'] = np.concatenate([c['eligible'] for c in groups[kind]])
            payload[kind+'_source_cell'] = np.concatenate([np.full(len(c['candidates']),c['cell']) for c in groups[kind]])
        np.savez(output/('image-%d-evaluator-only-pairs.npz'%i), **payload)
        reports.append(report)
        del gt
    assessment = rd.assess(reports)
    assessment['interpretation'] = 'development ordinal structure only; no metric accuracy or weighting gain'
    rd.write_json(output/'assessment.json', assessment)
    return assessment
