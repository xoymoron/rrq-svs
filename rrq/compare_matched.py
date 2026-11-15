"""Paired primary outcome with song-family cluster bootstrap, not phrase-independent CI."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .io import write_json


def compare(reports,repetitions=2000,seed=17):
    if set(reports)!={'baseline','exposure','rrq'}:raise ValueError('all three arms required')
    if len({r['protocol_sha256'] for r in reports.values()})!=1 or len({r['split'] for r in reports.values()})!=1:
        raise ValueError('experiment protocol/split mismatch')
    maps={arm:{r['id']:r for r in report['samples']} for arm,report in reports.items()}
    ids=sorted(maps['baseline'])
    if any(set(m)!=set(ids) or len(m)!=len(reports[arm]['samples']) for arm,m in maps.items()):raise ValueError('inexact paired coverage')
    rows=[maps['baseline'][i] for i in ids]
    for arm,m in maps.items():
        if any((m[i]['language'],m[i]['song_group'])!=(r['language'],r['song_group']) for i,r in zip(ids,rows)):
            raise ValueError('paired metadata mismatch')
    langs=sorted({r['language'] for r in rows});groups=sorted({r['song_group'] for r in rows})
    metric='mrstft_spectral_convergence'
    outcomes={arm:np.array([m[i]['arms']['generated'][metric] for i in ids]) for arm,m in maps.items()}
    if any(not np.isfinite(v).all() for v in outcomes.values()):raise ValueError('nonfinite primary metric')
    indices={l:np.array([i for i,r in enumerate(rows) if r['language']==l]) for l in langs}
    def macro(values,weights):
        return float(np.mean([np.average(values[ix],weights=weights[ix]) for ix in indices.values()]))
    result=dict(protocol_sha256=reports['baseline']['protocol_sha256'],split=reports['baseline']['split'],
                metric=metric,n_phrases=len(ids),n_song_families=len(groups),
                language_macro={a:macro(v,np.ones(len(ids))) for a,v in outcomes.items()},
                inference='self-generated all nine codebooks; no fixed/oracle prefix',
                scope='Within one training seed; bootstrap does not measure seed-to-seed uncertainty.')
    rng=np.random.default_rng(seed);weights=[];attempts=0
    while len(weights)<repetitions and attempts<20*repetitions:
        attempts+=1;counts=Counter(rng.choice(groups,size=len(groups),replace=True))
        w=np.array([counts[r['song_group']] for r in rows])
        if all(w[ix].sum()>0 for ix in indices.values()):weights.append(w)
    if len(weights)!=repetitions:raise ValueError('too few song families for requested language-macro bootstrap')
    for other in ('baseline','exposure'):
        delta=outcomes['rrq']-outcomes[other]
        boot=[macro(delta,w) for w in weights]
        result['rrq_minus_'+other]=dict(delta=macro(delta,np.ones(len(ids))),
            ci95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_repetitions=repetitions,
            per_language={l:float(delta[ix].mean()) for l,ix in indices.items()})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--split',choices=['valid','test'],default='valid');a=p.parse_args();root=Path(a.root)
    reports={arm:json.loads((root/arm/f'evaluation-{a.split}.json').read_text()) for arm in ('baseline','exposure','rrq')}
    result=compare(reports);write_json(root/f'comparison-{a.split}.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
