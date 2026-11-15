"""Paired summaries; no oracle metric is reported as generated SVS quality."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .experiment import ARMS, SEEDS, atomic_json


METRICS=('cer','mrstft_sc','mrstft_logmag','f0_rmse_cents','voicing_error','joint_voiced_fraction','latent_mse')


def summarize_rows(rows):
    groups=defaultdict(list)
    for r in rows: groups[r['language']].append(r['metrics'])
    means={language:{key:float(np.mean([r[key] for r in items if r.get(key) is not None]))
                     if any(r.get(key) is not None for r in items) else None for key in METRICS}
           for language,items in groups.items()}
    macro={key:float(np.mean([r[key] for r in means.values() if r[key] is not None]))
           if any(r[key] is not None for r in means.values()) else None for key in METRICS}
    return dict(n=len(rows),language_macro=macro,per_language=means)


def paired_song_ci(left,right,key,seed=4567):
    if set(left)!=set(right): raise ValueError('paired sample IDs differ')
    groups=defaultdict(list)
    for item in sorted(left):
        a,b=left[item],right[item]
        if a['song_group']!=b['song_group']: raise ValueError('paired song groups differ')
        av,bv=a['metrics'].get(key),b['metrics'].get(key)
        if av is not None and bv is not None: groups[a['song_group']].append(av-bv)
    differences=np.array([np.mean(v) for v in groups.values()])
    if not len(differences): return None
    rng=np.random.default_rng(seed); boot=rng.choice(differences,size=(2000,len(differences)),replace=True).mean(1)
    return dict(song_macro_difference=float(differences.mean()),ci95=list(map(float,np.quantile(boot,[.025,.975]))),songs=len(differences))


def summarize(root):
    root=Path(root); test={}; validation={}
    for split,target in [('test',test),('valid',validation)]:
        for path in (root/'evaluation'/split).glob('*.json'):
            target[path.stem]=json.loads(path.read_text())['samples']
    expected=['reference','codec','common']+[f'{arm}-{seed}' for seed in SEEDS for arm in ARMS]
    if any(k not in test for k in expected): raise ValueError('missing test arm/control')
    ids={r['id'] for r in test['reference']}
    if any({r['id'] for r in test[k]}!=ids for k in expected): raise ValueError('incomplete test evaluation')
    table={label:summarize_rows(rows) for label,rows in test.items()}
    arms={}
    for arm in ARMS:
        items=[table[f'{arm}-{seed}']['language_macro'] for seed in SEEDS]
        arms[arm]={key:dict(mean=float(np.mean([r[key] for r in items if r[key] is not None])),
                           seed_std=float(np.std([r[key] for r in items if r[key] is not None],ddof=1)))
                   for key in METRICS if all(r[key] is not None for r in items)}
    paired={}
    # Average a phrase over seeds first; bootstrap song clusters, not seed/phrase pseudo-replicates.
    averaged={}
    for arm in ARMS:
        indexed=[{r['id']:r for r in test[f'{arm}-{seed}']} for seed in SEEDS]
        averaged[arm]={}
        for item in ids:
            first=indexed[0][item]
            metrics={key:float(np.mean([d[item]['metrics'][key] for d in indexed]))
                     if all(d[item]['metrics'].get(key) is not None for d in indexed) else None for key in METRICS}
            averaged[arm][item]=dict(first,metrics=metrics)
    for baseline in ('baseline','exposure'):
        paired['rrq_minus_'+baseline]={key:paired_song_ci(averaged['rrq'],averaged[baseline],key) for key in METRICS}
    # Confirm matching optimization budgets and data order for every seed/epoch.
    schedule_audit=[]
    for seed in SEEDS:
        for epoch in range(1,7):
            logs=[json.loads((root/f'{arm}-{seed}'/f'validation-{epoch:02d}.json').read_text()) for arm in ARMS]
            if len({r['schedule_sha256'] for r in logs})!=1 or len({r['mixture_draws'] for r in logs})!=1:
                raise ValueError('training schedule/mixture not matched')
            schedule_audit.append(dict(seed=seed,epoch=epoch,schedule_sha256=logs[0]['schedule_sha256'],mixture_draws=logs[0]['mixture_draws']))
    result=dict(test_per_model=table,test_three_seed_aggregate=arms,paired_song_bootstrap=paired,
        multilingual_validation={k:summarize_rows(v) for k,v in validation.items()},schedule_audit=schedule_audit,
        uncertainty='Seed SD conditions on one common warmup; CI resamples song clusters after seed averaging. Small pilot, not population guarantee.',
        interpretation='Reference/codec ASR controls calibrate recognizer limitations. CE improvement alone does not establish intelligibility.')
    atomic_json(root/'comparison.json',result)
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','n',*METRICS]); writer.writeheader()
        for label,row in table.items(): writer.writerow(dict(model=label,n=row['n'],**row['language_macro']))
    print(json.dumps(arms,indent=2),flush=True)
    return result
