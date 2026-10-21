"""Frozen prediction-only margin and independent ordinal risk evaluation; CPU only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('frozen_ordinal', Path(__file__).with_name('relative_diagnostics.py'))
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)
PLAN_SHA = '3fb84489f24f97ec137ba85158c935771981a527ae7bf12bdf823ecf16e82ebb'
COVERAGES = (.25, .5, .75, 1.)
SCENES = ('terrace', 'delivery_area')
NAMES = {'terrace': ['dslr_images/DSC_0259.JPG','dslr_images/DSC_0271.JPG','dslr_images/DSC_0285.JPG'],
         'delivery_area': ['dslr_images/DSC_0675.JPG','dslr_images/DSC_0696.JPG','dslr_images/DSC_0718.JPG']}


def digest(path):
    return rd.digest(path)


def checked_plan(path):
    if digest(path) != PLAN_SHA:
        raise ValueError('Frozen margin protocol drift')
    plan = json.loads(Path(path).read_text())
    if digest(Path(__file__).with_name('relative_diagnostics.py')) != plan['inherited_protocol']['numerical_core_sha256']:
        raise ValueError('Frozen numerical core drift')
    return plan


def asset_key(row):
    key = row['scene']+'/'+row['relative_path']
    if '\\' in key or str(Path(key)) != key or Path(key).is_absolute() or '..' in Path(key).parts:
        raise ValueError('Unsafe asset key')
    return key


def prediction_header(path):
    """Only NPY header/length; never load real prediction values in admission."""
    with Path(path).open('rb') as f:
        if np.lib.format.read_magic(f) != (1,0):
            raise ValueError('Expected frozen NPY v1 header')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        payload_offset = f.tell()
    if shape != (672,1008) or fortran or dtype != np.dtype('<f4'):
        raise ValueError('Prediction header mismatch')
    if Path(path).stat().st_size != payload_offset+672*1008*4:
        raise ValueError('Prediction payload length mismatch')
    return dict(shape=list(shape),dtype=str(dtype),header_bytes=payload_offset)


def inspect_assets(rows):
    """Opaque hashes for evaluator assets; parse only scorer NPY headers/seals."""
    headers = []
    for row in rows:
        path = Path(row['path'])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Asset identity mismatch: '+row['key'])
        if row['role']=='scorer' and row['key'].endswith('.npy'):
            headers.append(dict(key=row['key'],**prediction_header(path)))
    by_key = {row['key']:row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError('Duplicate asset key')
    for row in rows:
        if row['role']=='scorer' and row['key'].endswith('/seal.json'):
            seal = json.loads(Path(row['path']).read_text())
            scene = row['key'].split('/')[0]
            if (seal['status']!='sealed_all_predictions' or [x['name'] for x in seal['predictions']]!=NAMES[scene]
                    or [x['file'] for x in seal['predictions']]!=['0.npy','1.npy','2.npy']):
                raise ValueError('Wrong frozen scene seal')
            prefix = row['key'].rsplit('/',1)[0]
            for record in seal['predictions']:
                if by_key[prefix+'/'+record['file']]['sha256'] != record['sha256']:
                    raise ValueError('Prediction seal digest mismatch')
    return dict(status='passed_metadata_only',files=len(rows),prediction_headers=headers,
                descriptor_computed=False,evaluator_payload_deserialized=False)


def sorted_reference(values):
    a = np.asarray(values,dtype=np.float64).ravel()
    valid = np.isfinite(a)&(a>=0)
    return np.sort(a[valid]),int((~valid).sum())


def empirical_cdf(reference, values):
    reference = np.asarray(reference,dtype=np.float64)
    if reference.ndim!=1 or not len(reference) or not np.isfinite(reference).all() or (reference<0).any() or (np.diff(reference)<0).any():
        raise ValueError('Nonempty sorted finite nonnegative reference required')
    values = np.asarray(values,dtype=np.float64)
    out = np.full(values.shape,np.nan)
    valid = np.isfinite(values)&(values>=0)
    lo = np.searchsorted(reference,values[valid],side='left')
    hi = np.searchsorted(reference,values[valid],side='right')
    out[valid] = (lo+hi)/(2.*len(reference))
    return out


def pair_risk(features,pairs,valid):
    pairs = np.asarray(pairs,dtype=np.int64).reshape(-1,2)
    a,b = pairs.T
    good = np.asarray(valid)[a]&np.asarray(valid)[b]&np.isfinite(features[a])&np.isfinite(features[b])
    risk = np.ones(len(pairs),dtype=np.float64)
    risk[good] = 1-np.abs(features[a[good]]-features[b[good]])
    return risk


def fractional_risk(loss,risk,coverage):
    loss,risk = np.asarray(loss,dtype=np.float64),np.asarray(risk,dtype=np.float64)
    if (loss.ndim!=1 or not len(loss) or loss.shape!=risk.shape or not 0<coverage<=1
            or not np.isfinite(risk).all() or ((risk<0)|(risk>1)).any()
            or not np.isin(loss,[0.,.5,1.]).all()):
        raise ValueError('Invalid frozen risk inputs')
    mass = coverage*len(loss)
    boundary = np.partition(risk,int(np.ceil(mass))-1)[int(np.ceil(mass))-1]
    below,equal = risk<boundary,risk==boundary
    weight = (mass-int(below.sum()))/int(equal.sum())
    return float((loss[below].sum()+weight*loss[equal].sum())/mass)


def shuffled_features(features,valid,seed,image_index):
    out = np.asarray(features,dtype=np.float64).copy()
    good = np.asarray(valid)&np.isfinite(out)
    rng = np.random.Generator(np.random.PCG64(seed+1000*image_index))
    out[good] = out[good][rng.permutation(int(good.sum()))]
    return out


def descriptors(rows,output,source_hashes):
    if any(row['role']!='scorer' for row in rows):
        raise ValueError('Evaluator asset supplied to descriptor stage')
    inspect_assets(rows)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    records = []
    for scene in SCENES:
        maps = sorted([r for r in rows if r['key'].startswith(scene+'/') and r['key'].endswith('.npy')],key=lambda r:r['key'])
        if len(maps)!=3:
            raise ValueError('Three frozen maps per scene required')
        for i,row in enumerate(maps):
            raw = np.load(row['path'],allow_pickle=False)
            if raw.shape!=(672,1008) or raw.dtype!=np.float32:
                raise ValueError('Raw schema mismatch')
            reference,invalid = sorted_reference(raw[2:670,2:1006])
            dest = output/(scene+'-%d.npy'%i)
            with dest.open('xb') as f:np.save(f,reference,allow_pickle=False)
            records.append(dict(scene=scene,image_index=i,name=NAMES[scene][i],file=dest.name,
                sha256=digest(dest),raw_key=row['key'],raw_sha256=row['sha256'],domain_pixels=670672,
                valid_pixels=len(reference),invalid_pixels=invalid,
                status='passed' if len(reference)>=.99*670672 else 'blocked_prediction_descriptor_support'))
    if len(records)!=6:
        raise ValueError('Six descriptors required before seal')
    rd.write_json(output/'seal.json',dict(status='sealed_six_descriptors',plan_sha256=PLAN_SHA,
        source_sha256=source_hashes,input_sha256={r['key']:r['sha256'] for r in rows},records=records))
    return records


def verified_descriptors(root,rows,source_hashes,load_values=True):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
        raise ValueError('Unsafe descriptor directory')
    seal = json.loads((root/'seal.json').read_text())
    if (seal.get('status')!='sealed_six_descriptors' or seal.get('plan_sha256')!=PLAN_SHA
            or seal.get('source_sha256')!=source_hashes
            or seal.get('input_sha256')!={r['key']:r['sha256'] for r in rows if r['role']=='scorer'}):
        raise ValueError('Descriptor seal provenance mismatch')
    expected = [(s,i) for s in SCENES for i in range(3)]
    if [(x['scene'],x['image_index']) for x in seal['records']]!=expected:
        raise ValueError('All six ordered descriptors required')
    names = {s+'-%d.npy'%i for s,i in expected}
    if {p.name for p in root.iterdir()} != names|{'seal.json'}:
        raise ValueError('Unexpected descriptor files')
    references = []
    by_key = {r['key']:r for r in rows if r['role']=='scorer'}
    for record,(scene,i) in zip(seal['records'],expected):
        name = scene+'-%d.npy'%i
        if (record['file']!=name or record['name']!=NAMES[scene][i]
                or record['raw_key']!=scene+'/prediction/stage/result/predictions/%d.npy'%i
                or record['raw_sha256']!=by_key[record['raw_key']]['sha256']
                or record['valid_pixels']+record['invalid_pixels']!=670672
                or record['domain_pixels']!=670672 or digest(root/name)!=record['sha256']):
            raise ValueError('Descriptor schema/content drift')
        expected_status='passed' if record['valid_pixels']>=.99*670672 else 'blocked_prediction_descriptor_support'
        if record['status']!=expected_status:
            raise ValueError('Descriptor validity status mismatch')
        if load_values:
            a = np.load(root/name,allow_pickle=False)
            if (a.dtype!=np.float64 or a.shape!=(record['valid_pixels'],)
                    or not np.isfinite(a).all() or (a<0).any() or (np.diff(a)<0).any()):
                raise ValueError('Invalid sorted descriptor')
            references.append(a)
    return seal,references


def prepare_image(raw,cache_path,baseline_path):
    baseline = json.loads(Path(baseline_path).read_text())
    with np.load(cache_path,allow_pickle=False) as cache:
        nodes = dict(indices=cache['native_indices'],cell=cache['cell'],z=cache['reference_Z'])
        n = len(nodes['indices'])
        if (nodes['indices'].dtype!=np.int64 or nodes['cell'].dtype!=np.int64
                or nodes['indices'].shape!=(n,) or nodes['cell'].shape!=(n,) or nodes['z'].shape!=(n,)
                or (np.diff(nodes['indices'])<=0).any() or (nodes['indices']<0).any()
                or (nodes['indices']>=6048*4032).any() or not np.isfinite(nodes['z']).all() or (nodes['z']<=0).any()):
            raise ValueError('Frozen node schema mismatch')
        y,x = np.divmod(nodes['indices'],6048)
        if not np.array_equal(nodes['cell'],(y//448)*12+x//504):
            raise ValueError('Native cell identity mismatch')
        xy = np.column_stack([(x+.5)/6-.5,(y+.5)/6-.5])
        if ((xy[:,0]<2)|(xy[:,0]>1005)|(xy[:,1]<2)|(xy[:,1]>669)).any():
            raise ValueError('Node outside fixed domain')
        groups = {}
        for kind in ['local','cross']:
            candidates,eligible,source = [cache[kind+suffix] for suffix in ['_candidate_pairs','_eligible','_source_cell']]
            if (candidates.dtype!=np.int64 or candidates.ndim!=2 or candidates.shape[1]!=2
                    or eligible.dtype!=np.bool_ or eligible.shape!=(len(candidates),)
                    or source.shape!=(len(candidates),) or (source<0).any() or (source>=108).any()
                    or (candidates<0).any() or (candidates>=n).any()
                    or (candidates[:,0]>=candidates[:,1]).any()
                    or not np.array_equal(eligible,rd.contrast_mask(nodes['z'],candidates))):
                raise ValueError('Frozen pair schema/contrast mismatch')
            cells = baseline['model']['strata'][kind]['cells']
            if [c['cell'] for c in cells] != list(range(108)):
                raise ValueError('Frozen cell mapping mismatch')
            groups[kind] = []
            for c,metadata in enumerate(cells):
                selected = source==c
                cp,ok = candidates[selected],eligible[selected]
                if (len(cp)!=metadata['unique_candidates'] or int(ok.sum())!=metadata['eligible_pairs']
                        or metadata['included']!=(int(ok.sum())>=16)):
                    raise ValueError('Frozen membership count mismatch')
                groups[kind].append(dict(metadata,candidates=cp,eligible=ok,pairs=cp[ok]))
    if rd.pair_fingerprint(nodes,groups)!=baseline['pair_fingerprint']:
        raise ValueError('Pair fingerprint mismatch')
    q,valid = rd.sample_raw(raw,xy)
    actual = rd.score(nodes,groups,q,valid)
    if actual['loss'] is None or abs(actual['loss']-baseline['model']['loss'])>1e-12:
        raise ValueError('Frozen image target changed')
    wrong_edges,wrong_cells = set(),set()
    for kind,cells in groups.items():
        for cell,reported in zip(cells,baseline['model']['strata'][kind]['cells']):
            loss,invalid,ties = rd.relation_loss(q,nodes['z'],cell['pairs'],valid)
            if cell['included']:
                if not len(loss) or abs(float(loss.mean())-reported['loss'])>1e-12:
                    raise ValueError('Frozen cell target changed')
                edges = cell['pairs'][loss==1]
                wrong_edges.update(map(tuple,edges.tolist()))
                if len(edges):wrong_cells.add(cell['cell'])
            cell['target_loss'],cell['invalid_pairs'],cell['target_ties'] = loss,invalid,ties
    if wrong_edges:
        endpoint,count = np.unique(np.asarray(sorted(wrong_edges)).ravel(),return_counts=True)
        node_count,max_incidence = len(endpoint),float(count.max()/len(wrong_edges))
    else:node_count,max_incidence = 0,None
    support = dict(unique_wrong_pairs=len(wrong_edges),error_source_cells=len(wrong_cells),
        error_incident_nodes=node_count,max_node_error_incidence_fraction=max_incidence,
        passed=len(wrong_edges)>=50 and len(wrong_cells)>=3 and actual['loss']>0)
    return dict(nodes=nodes,groups=groups,q=q,valid=valid,baseline=baseline,support=support)


def risk_curves(image,features,mode='margin'):
    strata = {}
    for kind,cells in image['groups'].items():
        rows,means = [],{str(c):[] for c in COVERAGES}
        for cell in cells:
            if not cell['included']:
                rows.append(dict(cell=cell['cell'],included=False,risk=None))
                continue
            loss = cell['target_loss']
            if mode=='oracle':risk=loss.copy()
            elif mode=='constant':risk=np.where(cell['invalid_pairs'],1.,.5)
            else:risk=pair_risk(features,cell['pairs'],image['valid'])
            values = {str(c):fractional_risk(loss,risk,c) for c in COVERAGES}
            if abs(values['1.0']-cell['loss'])>1e-12:
                raise ValueError('Full coverage cell identity failure')
            for c,v in values.items():means[c].append(v)
            rows.append(dict(cell=cell['cell'],included=True,pairs=len(loss),risk=values,
                wrong_memberships=int((loss==1).sum()),tie_memberships=int((loss==.5).sum()),invalid_memberships=int(cell['invalid_pairs'].sum())))
        risks = {c:float(np.mean(v)) for c,v in means.items()}
        strata[kind] = dict(risk=risks,cells=rows,
            discrimination='unknown_no_error' if risks['1.0']==0 else 'observed_error')
    return dict(strata=strata,risk={str(c):float(np.mean([v['risk'][str(c)] for v in strata.values()])) for c in COVERAGES})


def assess(images):
    if [(x['scene'],x['image_index']) for x in images]!=[(s,i) for s in SCENES for i in range(3)]:
        raise ValueError('All six ordered development images required')
    if not all(x['support']['passed'] for x in images):
        return dict(status='blocked_error_support',scenes={})
    passed = True
    scenes = {}
    for image in images:
        r50,r100 = image['margin']['risk']['0.5'],image['margin']['risk']['1.0']
        passed &= r50<=.8*r100 and r50<=.9*image['shuffle_mean']['0.5']
        passed &= all(s['risk']['0.5']<=s['risk']['1.0']+1e-12 for s in image['margin']['strata'].values())
    for scene in SCENES:
        selected = [x for x in images if x['scene']==scene]
        risks = {str(c):float(np.mean([x['margin']['risk'][str(c)] for x in selected])) for c in COVERAGES}
        scene_pass = risks['0.5']<=.75*risks['1.0']
        scenes[scene] = dict(risk=risks,scene_gate_passed=scene_pass)
        passed &= scene_pass
    return dict(status='passed_development_margin_screen' if passed else 'failed_margin_direction_gate',scenes=scenes)


def evaluate(rows,descriptor_root,output,source_hashes,access):
    # All six descriptor identities and arrays precede any evaluator asset interpretation.
    seal,references = verified_descriptors(descriptor_root,rows,source_hashes)
    inspect_assets(rows)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if any(r['status']!='passed' for r in seal['records']):
        result = dict(status='blocked_prediction_descriptor_support',scenes={})
        rd.write_json(output/'assessment.json',result)
        return result
    prepared,images = [],[]
    by_key = {r['key']:r for r in rows}
    for record in seal['records']:
        scene,i = record['scene'],record['image_index']
        raw = np.load(by_key[record['raw_key']]['path'],allow_pickle=False)
        pair_rows = [r for r in rows if r['key'].startswith(scene+'/') and r['key'].endswith('image-%d-evaluator-only-pairs.npz'%i)]
        baseline_rows = [r for r in rows if r['key'].startswith(scene+'/') and r['key'].endswith('/image-%d.json'%i)]
        if len(pair_rows)!=1 or len(baseline_rows)!=1:raise ValueError('Exactly one pair/baseline mapping required')
        access.extend([pair_rows[0]['key'],baseline_rows[0]['key']])
        image = prepare_image(raw,pair_rows[0]['path'],baseline_rows[0]['path'])
        prepared.append(image)
        images.append(dict(scene=scene,image_index=i,name=record['name'],support=image['support']))
    if all(x['support']['passed'] for x in images):
        for image,row,reference in zip(prepared,images,references):
            features = empirical_cdf(reference,image['q'])
            row.update(margin=risk_curves(image,features),constant=risk_curves(image,features,'constant'),oracle=risk_curves(image,features,'oracle'),shuffled=[])
            for seed in range(42,62):
                row['shuffled'].append(dict(seed=seed,**risk_curves(image,shuffled_features(features,image['valid'],seed,row['image_index']))))
            row['shuffle_mean'] = {str(c):float(np.mean([x['risk'][str(c)] for x in row['shuffled']])) for c in COVERAGES}
            row['shuffle_min'] = {str(c):min(x['risk'][str(c)] for x in row['shuffled']) for c in COVERAGES}
            row['shuffle_max'] = {str(c):max(x['risk'][str(c)] for x in row['shuffled']) for c in COVERAGES}
    for row in images:rd.write_json(output/(row['scene']+'-%d.json'%row['image_index']),row)
    result = assess(images)
    result['interpretation'] = 'observed development ambiguity control only;not geometric confidence or training gain'
    rd.write_json(output/'assessment.json',result)
    return result
