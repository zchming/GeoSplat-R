"""Descriptive paired seed summaries; seeds and views are not independent scenes."""
import math
import statistics

METRICS=('psnr','ssim','lpips_alex')


def describe(values):
    values=[float(v) for v in values]
    if len(values)<2 or not all(math.isfinite(v) for v in values):
        raise ValueError('At least two finite values required for sample variation')
    return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),
        'median':statistics.median(values),'min':min(values),'max':max(values),'n_seeds':len(values)}


def summarize_pairs(pairs):
    seeds=[p['seed'] for p in pairs]
    if sorted(seeds)!=[42,43,44]:raise ValueError('Exactly one pair for each authorized seed42/43/44 required')
    for p in pairs:
        a=p['zero']['test'];b=p['uniform']['test']
        expected=['image000.png','image008.png','image016.png']
        if [r['image'] for r in a['per_image']]!=expected or [r['image'] for r in b['per_image']]!=expected:
            raise ValueError('Frozen test-image order required in both arms')
        for arm in ['zero','uniform']:
            for split in ['train','test']:
                result=p[arm][split]
                for metric in METRICS:
                    mean=statistics.mean(r[metric] for r in result['per_image'])
                    if not math.isfinite(mean) or not math.isclose(mean,result['mean'][metric],abs_tol=1e-6):
                        raise ValueError('Per-image and reported mean metrics disagree')
    aggregates={}
    for split in ['train','test']:
        aggregates[split]={}
        for metric in METRICS:
            zero=[p['zero'][split]['mean'][metric] for p in pairs]
            uniform=[p['uniform'][split]['mean'][metric] for p in pairs]
            delta=[b-a for a,b in zip(zero,uniform)]
            good=[-d if metric=='lpips_alex' else d for d in delta]
            aggregates[split][metric]={'zero':describe(zero),'uniform':describe(uniform),
                'delta_uniform_minus_zero':describe(delta),
                'improved_seeds':sum(v>0 for v in good),'degraded_seeds':sum(v<0 for v in good),
                'tied_seeds':sum(v==0 for v in good)}
    per_image=[];degradations=[]
    for p in pairs:
        for a,b in zip(p['zero']['test']['per_image'],p['uniform']['test']['per_image']):
            row={'seed':p['seed'],'image':a['image'],'zero':{m:a[m] for m in METRICS},
                'uniform':{m:b[m] for m in METRICS},'delta_uniform_minus_zero':{m:b[m]-a[m] for m in METRICS}}
            row['degraded_metrics']=[m for m in METRICS if (b[m]>a[m] if m=='lpips_alex' else b[m]<a[m])]
            per_image.append(row)
            if row['degraded_metrics']:degradations.append(row)
    return {'aggregates':aggregates,'paired_test_images':per_image,'degradations':degradations,
        'statistical_unit':'Three seeds on the same fern development scene; summarize per-seed scene means, not nine independent images',
        'inferential_limits':'Descriptive sample SD only; no p-value, cross-scene generalization or isolated CUDA-determinism conclusion'}
