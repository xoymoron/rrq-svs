"""Read-only full-cache readiness audit; never starts or approves training."""
import argparse
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from .uniaudio_data import load_recipe, PreparedData, _identity
from .uniaudio_train import protocol
from .io import write_json


def validate_preflight(preflight,expected_protocol,languages):
    if preflight.get('world_size')!=2:raise ValueError('two-GPU preflight required')
    if (preflight.get('protocol_sha256')!=expected_protocol or preflight.get('optimizer_steps')!=0
            or preflight.get('status')!='passed' or not preflight.get('weights_unchanged')):
        raise ValueError('missing or stale zero-step preflight')
    expected=Counter({(language,arm):2 for language in languages for arm in ('baseline','exposure','rrq')})
    if Counter((r['language'],r['arm']) for r in preflight['results'])!=expected:
        raise ValueError('preflight does not cover every arm and language on both GPUs')


def audit(cfg,preflight_path):
    data=PreparedData(cfg['prepared_path'],cfg['cache_path'])
    identity=_identity(cfg)
    if any(data.recipe.get(k)!=v for k,v in identity.items() if k!='config'):
        raise ValueError('runtime recipe differs')
    preflight=json.loads(Path(preflight_path).read_text())
    languages={r['language'] for r in data.rows.values()}
    validate_preflight(preflight,protocol(data),languages)
    samples=0
    for row in tqdm(data.rows.values(),desc='Verify complete native cache'):
        record=data._cached(row);samples+=record['samples']
    return dict(status='ready_for_explicit_common_finetuning',optimizer_steps=0,
                protocol_sha256=protocol(data),rows=len(data.rows),hours=samples/16000/3600,
                languages=sorted(languages),segments=dict(Counter(s['split'] for s in data.segments)),
                heldout_unknowns=data.vocab['heldout_unknowns'],test_used_for_model_selection=False,
                synthesis_quality_passed=False,rrq_training_ready=False,
                blocker='Common checkpoint must pass the matching validation-panel gate before RRQ branches.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--workspace',required=True)
    p.add_argument('--preflight',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(load_recipe(a.config,a.workspace),a.preflight)
    write_json(a.output,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
