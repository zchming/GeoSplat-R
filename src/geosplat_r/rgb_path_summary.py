"""Descriptive matched path contrasts; not a formal equivalence test."""
import math
import statistics
from geosplat_r.seed_summary import describe,METRICS

ARMS=('rgb_only','rgb_ed_zero','uniform')
CONTRASTS={'rgb_ed_zero_minus_rgb_only':('rgb_only','rgb_ed_zero'),
    'uniform_minus_rgb_only':('rgb_only','uniform'),
    'uniform_minus_rgb_ed_zero':('rgb_ed_zero','uniform')}


def summarize_paths(rows):
    if sorted(p['seed'] for p in rows)!=[42,43,44]:raise ValueError('Exactly seed42/43/44 required once each')
    names={'train':['image001.png','image010.png','image019.png'],
        'test':['image000.png','image008.png','image016.png']}
    for p in rows:
        for arm in ARMS:
            for split in ['train','test']:
                result=p[arm][split]
                if [r['image'] for r in result['per_image']]!=names[split]:raise ValueError('Frozen view names/order mismatch')
                for metric in METRICS:
                    values=[r[metric] for r in result['per_image']]
                    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite quality metric')
                    if not math.isclose(statistics.mean(values),result['mean'][metric],abs_tol=1e-6):
                        raise ValueError('Reported metric mean mismatch')
    arms={split:{arm:{m:describe(p[arm][split]['mean'][m] for p in rows) for m in METRICS} for arm in ARMS}
        for split in ['train','test']}
    contrasts={}
    for name,(reference,candidate) in CONTRASTS.items():
        item={'reference':reference,'candidate':candidate,'splits':{},'paired_test_images':[],'degradations':[]}
        for split in ['train','test']:
            item['splits'][split]={}
            for m in METRICS:
                delta=[p[candidate][split]['mean'][m]-p[reference][split]['mean'][m] for p in rows]
                oriented=[-v if m=='lpips_alex' else v for v in delta]
                item['splits'][split][m]={'delta':describe(delta),'improved_seeds':sum(v>0 for v in oriented),
                    'degraded_seeds':sum(v<0 for v in oriented),'tied_seeds':sum(v==0 for v in oriented)}
        for p in rows:
            for a,b in zip(p[reference]['test']['per_image'],p[candidate]['test']['per_image']):
                record={'seed':p['seed'],'image':a['image'],'reference':{m:a[m] for m in METRICS},
                    'candidate':{m:b[m] for m in METRICS},'delta':{m:b[m]-a[m] for m in METRICS}}
                record['degraded_metrics']=[m for m in METRICS if (b[m]>a[m] if m=='lpips_alex' else b[m]<a[m])]
                item['paired_test_images'].append(record)
                if record['degraded_metrics']:item['degradations'].append(record)
        contrasts[name]=item
    return {'arm_aggregates':arms,'contrasts':contrasts,
        'statistical_unit':'Three paired seeds on one development scene; mean of per-seed scene means',
        'limits':'Descriptive sample SD only. One run per seed/path, coupled prior-cache/diagnostics differences and CUDA nondeterminism prevent causal or formal equivalence claims.'}
