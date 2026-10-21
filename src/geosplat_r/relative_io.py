"""Sealed real-input bridge for the frozen ordinal protocol; CPU IO only."""
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('frozen_ordinal_core', Path(__file__).with_name('relative_diagnostics.py'))
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def configs(plan, adapter_sources):
    names = plan['dataset']['selected_names']
    inputs, labels = {}, []
    for row in plan['dataset']['inputs']:
        if row['role'] == 'predictor_rgb':
            inputs['rgb/'+row['image']] = {k: row[k] for k in ['bytes','sha256']}
        elif row['role'] == 'predictor_cameras':
            inputs['cameras.json'] = {k: row[k] for k in ['bytes','sha256']}
        elif row['role'] == 'evaluator_GT':
            labels.append({k: row[k] for k in ['image','bytes','sha256']})
    provenance = dict(adapter_sources)
    provenance.update({'model/'+r['path']: r['sha256'] for r in plan['prediction']['model']['source_files']})
    shared = dict(protocol=plan['protocol'], plan_sha256=rd.PLAN_SHA, names=names,
                  native_wh=plan['dataset']['native_wh'], work_wh=plan['prediction']['working_wh'],
                  source_sha256=provenance, input_sha256={k:v['sha256'] for k,v in inputs.items()})
    predictor = dict(**shared, inputs=inputs, model=plan['prediction']['model'])
    evaluator = dict(**shared, labels=labels)
    return predictor, evaluator


def validate_context(config):
    if (config['plan_sha256'] != rd.PLAN_SHA or config['protocol'] != 'terrace-anchor-free-relative-plan-v1'
            or config['names'] != ['dslr_images/DSC_0259.JPG','dslr_images/DSC_0271.JPG','dslr_images/DSC_0285.JPG']
            or config['native_wh'] != [6048,4032] or config['work_wh'] != [1008,672]):
        raise ValueError('Frozen real-input protocol mismatch')


def admit_predictor(data, config):
    data = Path(data)
    expected = {'cameras.json'} | {'rgb/'+n for n in config['names']}
    if (data.is_symlink() or any(p.is_symlink() for p in data.rglob('*'))
            or {str(p.relative_to(data)) for p in data.rglob('*') if p.is_file()} != expected
            or set(config['inputs']) != expected):
        raise ValueError('Exactly three selected RGBs and camera export required')
    for rel in sorted(expected):
        p, spec = data/rel, config['inputs'][rel]
        if p.stat().st_size != spec['bytes'] or rd.digest(p) != spec['sha256']:
            raise ValueError('Input digest/length drift: '+rel)
    views = json.loads((data/'cameras.json').read_text())
    if ([v['name'] for v in views] != config['names'] or len({v['image_id'] for v in views}) != 3
            or any([v['camera']['width'],v['camera']['height']] != config['native_wh'] for v in views)):
        raise ValueError('Camera/name/native-grid mismatch')
    return views


def sealed_predictions(root, config):
    """Every seal/digest and every shape/type must pass before any GT loader."""
    seal = rd.verify_prediction_seal(root, config['names'], config['source_sha256'], config['input_sha256'])
    arrays = []
    for row in seal['predictions']:
        q = np.load(Path(root)/row['file'], allow_pickle=False)
        if q.dtype != np.float32 or q.shape != tuple(reversed(config['work_wh'])):
            raise ValueError('Raw relative prediction shape/type mismatch')
        arrays.append(q)
    return arrays


def native_reference(path, record, native_wh):
    path = Path(path)
    expected = 4*int(native_wh[0])*int(native_wh[1])
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != expected
            or record['bytes'] != expected or rd.digest(path) != record['sha256']):
        raise ValueError('Native evaluator-only reference digest/length mismatch')
    return np.memmap(path, dtype='<f4', mode='r', shape=tuple(reversed(native_wh)))


def evaluate_sealed(predictions, labels, config, output, access):
    """No reference metadata/digest/payload access until all predictions accepted."""
    arrays = sealed_predictions(predictions, config)
    if len(config['labels']) != 3 or [r['image'] for r in config['labels']] != config['names']:
        raise ValueError('Exactly three mapped evaluator labels required')
    reports = []
    for i, (name, row, q) in enumerate(zip(config['names'], config['labels'], arrays)):
        if row['image'] != name or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Reference name mapping mismatch')
        access.append(name)
        gt = native_reference(Path(labels)/name, row, config['native_wh'])
        report, nodes, groups = rd.evaluate_arrays(gt, q, i)
        rd.write_json(Path(output)/('image-%d.json'%i), report)
        payload = dict(native_indices=nodes['indices'], reference_Z=nodes['z'], cell=nodes['cell'])
        for kind in ['local','cross']:
            payload[kind+'_candidate_pairs'] = np.concatenate([c['candidates'] for c in groups[kind]])
            payload[kind+'_eligible'] = np.concatenate([c['eligible'] for c in groups[kind]])
            payload[kind+'_source_cell'] = np.concatenate([np.full(len(c['candidates']),c['cell']) for c in groups[kind]])
        np.savez(Path(output)/('image-%d-evaluator-only-pairs.npz'%i), **payload)
        reports.append(report)
        del gt
    assessment = rd.assess(reports)
    assessment['interpretation'] = 'development-scene ordinal structure only; not metric accuracy, confidence weighting or training gain'
    rd.write_json(Path(output)/'assessment.json', assessment)
    return assessment
