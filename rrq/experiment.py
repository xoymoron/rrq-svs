"""Locked protocol and deterministic shared schedules for the three-arm experiment."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .io import sha256, write_json
from .toksing_data import read_rows, inventory, annotation_hash
from .toksing_run import atomic_save


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def epoch_schedule(rows,seed,epoch,world,accumulation,chinese_weight):
    groups=defaultdict(lambda:defaultdict(list))
    for row in sorted(rows,key=lambda r:r['id']):groups[row['language']][row['singer']].append(row['id'])
    languages=sorted(groups);weights=[chinese_weight if l=='Chinese' else 1. for l in languages]
    if not 0<chinese_weight<=1:raise ValueError('Chinese sampling weight must be in (0,1]')
    rng=random.Random(seed+1000003*epoch)
    count=math.ceil(len(rows)/(world*accumulation))*world*accumulation
    result=[]
    for _ in range(count):
        lang=rng.choices(languages,weights=weights,k=1)[0]
        singer=rng.choice(sorted(groups[lang]));result.append(rng.choice(groups[lang][singer]))
    return result


def mixture_choice(seed,epoch,index,probability):
    if not 0<=probability<=1:raise ValueError('invalid mixture probability')
    value=int(hashlib.sha256(f'mixture:{seed}:{epoch}:{index}'.encode()).hexdigest()[:16],16)/2**64
    return value<probability


def model_config(vocab,config):
    return dict(phones=len(vocab['phones']),singers=len(vocab['singers']),languages=len(vocab['languages']),
                codebooks=9,entries=1024,**config['model'])


class Experiment:
    def __init__(self,prepared,output,codec,config):
        self.prepared=Path(prepared).resolve();self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.codec=Path(codec).resolve();self.config=json.loads(Path(config).read_text(encoding='utf-8'))
        self.manifest=self.prepared/'manifest.jsonl';self.rows=read_rows(self.manifest)
        self.by_id={r['id']:r for r in self.rows}
        self.train=[r for r in self.rows if r['split']=='train'];self.valid=[r for r in self.rows if r['split']=='valid']
        if not self.train or not self.valid:raise ValueError('empty train or validation split')
        if any(r['corpus']!='gtsinger' for r in self.rows):raise ValueError('this protocol is GTSinger singing only')
        self.vocab=inventory(self.train)
        self.panels=json.loads((self.prepared/'panels.json').read_text())
        for split,ids in self.panels.items():
            if len(set(ids))!=len(ids) or any(self.by_id[i]['split']!=split for i in ids):
                raise ValueError('invalid or leaked validation/train panel')
        for lang in {r['language'] for r in self.valid}:
            expected=min(self.config['validation_panel_per_language'],sum(r['language']==lang for r in self.valid))
            if sum(self.by_id[i]['language']==lang for i in self.panels['valid'])!=expected:
                raise ValueError('validation panel count does not match protocol')
        audit=json.loads((self.prepared/'audit.json').read_text())
        if audit['manifest_sha256']!=sha256(self.manifest):raise ValueError('preprocessing audit/manifest mismatch')
        for split in ('valid','test'):
            held=[r for r in self.rows if r['split']==split]
            if not {r['singer'] for r in held}<=set(self.vocab['singers']):raise ValueError('unseen singer in seen-singer protocol')
        self.protocol=dict(version=1,manifest_sha256=sha256(self.manifest),codec_sha256=sha256(self.codec),
            config=self.config,vocabulary_sha256=fingerprint(self.vocab),panels_sha256=fingerprint(self.panels),
            preprocess_config=audit['config'],conditioning='provided phone duration, MIDI pitch, singer, language, six technique flags',
            prefix='offline frozen common-model greedy first codebook; all arms self-generate at primary evaluation',
            selection='fixed validation panel language-macro free-running MRSTFT spectral convergence')
        self.protocol['implementation']={name:sha256(Path(__file__).parent/name) for name in
            ('experiment.py','matched_run.py','toksing_model.py','toksing_data.py','targets.py','core.py','dac_codec.py','metrics.py','distributed.py')}
        self.hash=fingerprint(self.protocol)
        self.protocol['sha256']=self.hash
        lock=self.root/'protocol.json'
        if lock.exists() and json.loads(lock.read_text())!=self.protocol:raise ValueError('immutable experiment protocol mismatch')
        # Only rank zero creates shared records; other ranks check the same content.
        import os
        if int(os.environ.get('RANK','0'))==0:
            write_json(lock,self.protocol);write_json(self.root/'vocabulary.json',self.vocab)
        self.cache=self.root/'cache';self.cache.mkdir(exist_ok=True)
        self.bank=self.root/'target-bank';self.bank.mkdir(exist_ok=True)

    def read_cache(self,row,device='cpu'):
        c=torch.load(self.cache/(row['id']+'.pt'),map_location=device,weights_only=True)
        if (c['id']!=row['id'] or c['annotation_sha256']!=annotation_hash(row) or
            c['source_sha256']!=row['source_sha256'] or c['codec_sha256']!=self.protocol['codec_sha256']):
            raise ValueError('cache provenance mismatch: '+row['id'])
        c['codes']=c['codes'].long();c['frame_index']=c['frame_index'].long()
        return c

    def read_bank(self,row,common_hash,device='cpu'):
        b=torch.load(self.bank/(row['id']+'.pt'),map_location=device,weights_only=True)
        if (b['id']!=row['id'] or b['common_sha256']!=common_hash or b['protocol_sha256']!=self.hash or
            b['annotation_sha256']!=annotation_hash(row) or b['source_sha256']!=row['source_sha256']):
            raise ValueError('target bank provenance mismatch')
        b['predicted']=b['predicted'].long();b['corrected']=b['corrected'].long()
        return b

    def checkpoint(self,path,device='cpu'):
        state=torch.load(path,map_location=device,weights_only=True)
        if state['protocol_sha256']!=self.hash:raise ValueError('checkpoint protocol mismatch')
        return state

    def require_cache(self):
        done=self.root/'cache-complete.json'
        if not done.exists():raise ValueError('cache stage has not completed')
        value=json.loads(done.read_text())
        if value['protocol_sha256']!=self.hash or value['n']!=len(self.rows):raise ValueError('cache completion mismatch')

    def require_bank(self):
        path=self.root/'common/best.pt'
        if not path.exists():raise ValueError('common warmup checkpoint is required before target generation/branches')
        checksum=sha256(path);done=self.root/'target-bank-complete.json'
        if not done.exists():raise ValueError('target-bank stage has not completed')
        value=json.loads(done.read_text())
        if value['protocol_sha256']!=self.hash or value['common_sha256']!=checksum:raise ValueError('stale frozen target bank')
        if not value['oracle_room_detected']:raise ValueError('No matched oracle improvement; review before spending branch training budget')
        return checksum


def common_quality_gate(initial,best,config):
    metric=config['selection_metric']
    before=initial['language_macro']['generated'][metric];after=best['language_macro']['generated'][metric]
    positive=[lang for lang,m in best['diagnostics'].items() if
              m['phone_reverse_ce_gap']>config['gate_min_condition_gap'] and
              m['note_shift_ce_gap']>config['gate_min_condition_gap']]
    passed=after<before*config['gate_relative_sc'] and len(positive)>=config['gate_min_languages']
    return dict(passed=passed,initial_sc=before,best_sc=after,ratio=after/max(before,1e-12),
                positive_condition_languages=positive,
                note='Predeclared operational feasibility checks, not proof of intelligibility or statistical significance.')
