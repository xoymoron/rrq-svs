"""Immutable, audited GTSinger singing manifests; whole song families stay together."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .corpora import digest
from .io import sha256, write_json
from .toksing_data import frame_condition

TECHNIQUES = ('mix_tech','falsetto_tech','breathy_tech','pharyngeal_tech','vibrato_tech','glissando_tech')


def title_key(title):
    text = unicodedata.normalize('NFKD', title).casefold()
    return ''.join(c for c in text if c.isalnum() and not unicodedata.combining(c))


def build_families(rows, aliases):
    keys = {r['id']: title_key(r['song_title']) for r in rows}
    if not all(keys.values()):
        raise ValueError('empty song title')
    parent = {v:v for v in keys.values()}
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v
    def join(a,b):
        a,b = sorted((find(a),find(b))); parent[b] = a
    for group in aliases:
        known = [title_key(t) for t in group if title_key(t) in parent]
        for key in known[1:]:join(known[0],key)
    duplicates = {}
    for row in rows:
        checksum = row.get('pcm_sha256')
        if checksum:
            key = keys[row['id']]
            if checksum in duplicates:join(key,duplicates[checksum])
            duplicates[checksum] = key
    return {r['id']:'song-'+digest(find(keys[r['id']]))[:20] for r in rows}


def assign_splits(rows, families, seed=17):
    groups = defaultdict(list)
    for row in rows:groups[families[row['id']]].append(row)
    def order(g):return digest(f'gtsinger-family-v1:{seed}:{g}')
    assignment = {g:('test' if int(order(g)[:8],16)%100<10 else
                      'valid' if int(order(g)[:8],16)%100<20 else 'train') for g in groups}
    coverage = defaultdict(set)
    for group,items in groups.items():
        for row in items:
            coverage[('singer',row['singer'])].add(group)
            coverage[('language',row['language'])].add(group)
    minimum={key:(3 if key[0]=='language' else 1) for key in coverage}
    if any(len(gs)<3*minimum[key] for key,gs in coverage.items()):
        raise ValueError('insufficient song families for singer and language split coverage')
    affected={g:[key for key,gs in coverage.items() if g in gs] for g in groups}
    # Preserve all donor constraints while repairing language and singer coverage.
    for key,gs in sorted(coverage.items()):
        for split in ('train','valid','test'):
            while sum(assignment[g]==split for g in gs)<minimum[key]:
                candidates=[]
                for g in gs:
                    old=assignment[g]
                    if old==split:continue
                    if all(sum(assignment[x]==old for x in coverage[k])>minimum[k] for k in affected[g]):
                        candidates.append(g)
                if not candidates:raise ValueError('cannot preserve split coverage without song leakage')
                assignment[min(candidates,key=order)]=split
    return {r['id']:assignment[families[r['id']]] for r in rows}


def prepare_row(row, info):
    row=dict(row)
    if info['channels']!=1:raise ValueError('non-mono waveform')
    seconds=info['frames']/info['samplerate']
    if seconds<=0:raise ValueError('empty waveform')
    frames=int(np.ceil(round(seconds*44100)/512))
    frame_condition(row,frames,round(seconds*44100),44100,512)
    for key in TECHNIQUES:
        values=np.asarray(row.get('techniques',{}).get(key,[]))
        if values.shape!=(len(row['phones']),) or not np.isin(values,[0,1]).all():
            raise ValueError('invalid technique annotation: '+key)
    row.update(audio_seconds=seconds,source_sample_rate=info['samplerate'],
               source_samples=info['frames'],conditioning='score_pitch_gt_phone_duration_and_technique',
               lyric_unit='provided_phoneme_alignment')
    return row


def select_panel(rows, split='valid', per_language=8):
    selected=[]
    for language in sorted({r['language'] for r in rows}):
        singers=defaultdict(list)
        for r in rows:
            if r['split']==split and r['language']==language:singers[r['singer']].append(r)
        for s in singers:singers[s].sort(key=lambda r:digest('panel-v1:'+r['id']))
        count=0; used=set()
        while count<per_language and any(singers.values()):
            for s in sorted(singers):
                if not singers[s] or count==per_language:continue
                candidates=[r for r in singers[s] if r['song_group'] not in used]
                r=(candidates or singers[s])[0];singers[s].remove(r)
                selected.append(r['id']);used.add(r['song_group']);count+=1
    return sorted(selected)


def deduplicate(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['pcm_sha256']].append(row)
    kept=[];duplicates=[];rejected=[]
    for group in groups.values():
        group=sorted(group,key=lambda r:r['id'])
        conditions=set()
        for row in group:
            used={k:row.get(k) for k in ('phones','phone_durations','techniques','singer','language')}
            # ep_notedurs/ep_types/ph2words are preserved metadata, not model inputs.
            used['midi']=row['score'].get('ep_pitches',row['score'].get('notes'))
            conditions.add(json.dumps(used,sort_keys=True))
        if len(conditions)>1:
            rejected.extend(dict(id=r['id'],reason='identical PCM has conflicting conditioning annotations',
                                 duplicate_ids=[x['id'] for x in group]) for r in group)
            continue
        kept.append(group[0])
        duplicates.extend(dict(id=r['id'],kept=group[0]['id']) for r in group[1:])
    return sorted(kept,key=lambda r:r['id']),duplicates,rejected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--aliases',required=True)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    complete=json.loads((root/'download-status.json').read_text())
    if not complete.get('complete'):raise ValueError('full source acquisition is not verified complete')
    rows=[json.loads(x) for x in Path(a.source).read_text(encoding='utf-8').splitlines() if x]
    aliases=json.loads(Path(a.aliases).read_text(encoding='utf-8'))['groups']
    config=dict(source_sha256=sha256(a.source),aliases_sha256=sha256(a.aliases),seed=a.seed,
                policy='song-family-v2',min_families_per_language_split=3,
                audio_tolerance_seconds=.04,techniques=list(TECHNIQUES),
                duplicate_policy='reject_conflicting_model_conditioning_v1',implementation_sha256=sha256(__file__))
    lock=out/'preprocess-config.json'
    if lock.exists() and json.loads(lock.read_text())!=config:raise ValueError('refuse overwrite with different preprocessing configuration')
    write_json(lock,config)
    valid=[];rejected=[]
    def inspect(row):
        try:
            path=Path(row['audio']);info=sf.info(path)
            result=prepare_row(row,dict(samplerate=info.samplerate,frames=info.frames,channels=info.channels))
            checksum=hashlib.sha256(f'{info.samplerate}:{info.channels}:'.encode());energy=0.;peak=0.;clipped=0
            for block in sf.blocks(path,blocksize=65536,dtype='float32',always_2d=True):
                if not np.isfinite(block).all():raise ValueError('nonfinite audio')
                checksum.update(block.astype('<f4',copy=False).tobytes())
                energy+=float(np.square(block.astype(np.float64)).sum())
                peak=max(peak,float(np.abs(block).max()));clipped+=int((np.abs(block)>=.9999).sum())
            if energy==0:raise ValueError('all-zero waveform')
            result.update(pcm_sha256=checksum.hexdigest(),source_sha256=sha256(path),
                          rms=float(np.sqrt(energy/info.frames)),peak=peak,clipped_fraction=clipped/info.frames)
            return result,None
        except (ValueError,RuntimeError,OSError) as exc:return None,dict(id=row['id'],reason=str(exc))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result,error in tqdm(pool.map(inspect,rows),total=len(rows),desc='Audio/annotation audit'):
            if error:rejected.append(error)
            else:valid.append(result)
    # Keep one canonical record for exact duplicate PCM, after family union.
    families=build_families(valid,aliases)
    unique,duplicates,conflicting=deduplicate(valid);rejected.extend(conflicting)
    splits=assign_splits(unique,families,a.seed)
    for row in unique:
        row.update(song_group=families[row['id']],split=splits[row['id']],split_policy='song-family-v2-seen-singer')
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in unique)
    path=out/'manifest.jsonl'
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('immutable prepared manifest changed')
    path.write_text(text,encoding='utf-8')
    stats={}
    for split in ('train','valid','test'):
        subset=[r for r in unique if r['split']==split]
        stats[split]=dict(n=len(subset),hours=sum(r['audio_seconds'] for r in subset)/3600,
                         songs=len({r['song_group'] for r in subset}),singers=len({r['singer'] for r in subset}),
                         per_language={l:dict(n=sum(r['language']==l for r in subset),hours=sum(r['audio_seconds'] for r in subset if r['language']==l)/3600) for l in sorted({r['language'] for r in unique})})
    write_json(out/'panels.json',dict(valid=select_panel(unique),train=select_panel(unique,'train',2)))
    write_json(out/'audit.json',dict(config=config,manifest_sha256=sha256(path),input_rows=len(rows),
        included=len(unique),rejected=rejected,exact_duplicate_pcm=duplicates,splits=stats,
        min_seconds=min(r['audio_seconds'] for r in unique),max_seconds=max(r['audio_seconds'] for r in unique),
        annotation_limit='Provided timing/technique and retrospective score; not autonomous timing or zero-shot.',
        song_limit='Conservative title/explicit-alias/PCM grouping. Unidentified translated compositions remain a documented limitation.'))
    print(json.dumps(stats,ensure_ascii=False),flush=True)
    if rejected:print(f'Explicitly excluded {len(rejected)} invalid records; raw files retained.',flush=True)


if __name__=='__main__':main()
