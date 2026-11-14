"""Official-source pilot preparation. This does not run the SVS generator."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import subprocess
import urllib.parse
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .corpora import gtsinger_row, m4singer_row, select_rows, frame_phones, model_phone, validate_annotations
from .data_download import download, RangeReader, extract_member, safe_relative, free_space
from .io import sha256, write_json

GT_REV='4426c862beed558b7e1cb8a4dce7e8c0c83bb208'
GT_BASE=f'https://huggingface.co/datasets/GTSinger/GTSinger/resolve/{GT_REV}/'
M4_URL='https://drive.usercontent.google.com/download?id=1xC37E59EWRRFFLdG3aJkVqwtLDgtFNqW&export=download&confirm=t'
LANGUAGES=['Chinese','English','French','German','Italian','Japanese','Korean','Russian','Spanish']


def jsonl(path,rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def fetch_gtsinger(root,per_singer,languages):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    download(GT_BASE+'dataset_license.md',root/'dataset_license.md',max_bytes=100000)
    rows=[]; counts={}
    for language in languages:
        path=download(GT_BASE+f'processed/{language}/metadata.json',root/'metadata'/language/'metadata.json')
        items=json.loads(path.read_text(encoding='utf-8')); counts[language]=len(items)
        for item in items:
            row=gtsinger_row(item)
            # Older metadata may use absolute author paths: rebuild from the
            # documented six-component item ID, never trust an external path.
            parts=item['item_name'].split('#')
            row['source_audio']='/'.join(parts[:-1]+[parts[-1]+'.wav'])
            safe_relative(row['source_audio']); rows.append(row)
    chosen=select_rows(rows,per_singer)
    files=[]
    for row in chosen:
        base=Path(row['source_audio']).with_suffix('').as_posix()
        files.extend(base+ext for ext in ('.wav','.json','.TextGrid','.musicxml'))
    def get(rel):
        try:
            download(GT_BASE+urllib.parse.quote(rel,safe='/'),root/'raw'/rel,max_bytes=16*1024**2)
            return None
        except Exception as exc: return dict(path=rel,error=str(exc))
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(get,files)):
            if result: failures.append(result)
            if i%100==0: print(f'gtsinger files {i+1}/{len(files)}',flush=True)
    jsonl(root/'selected.jsonl',chosen)
    write_json(root/'download-report.json',dict(corpus='gtsinger',revision=GT_REV,
        per_singer=per_singer,source_items=counts,selected=len(chosen),failures=failures,
        subset='deterministic song-diverse pilot, not full corpus',license='CC-BY-NC-SA-4.0',
        citation='https://arxiv.org/abs/2409.13832'))
    if failures: raise RuntimeError(f'{len(failures)} downloads failed; rerun to resume')


def fetch_m4singer(root,per_singer):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    download('https://raw.githubusercontent.com/M4Singer/M4Singer/master/dataset_license.md',root/'dataset_license.md',max_bytes=100000)
    # Aggregate cap: metadata 128 MiB + 4 workers * 448 MiB = 1920 MiB.
    with RangeReader(M4_URL,max_bytes=128*1024**2) as source, zipfile.ZipFile(source) as archive:
        names=archive.namelist()
        metadata=[p for p in names if p.endswith('/meta.json') or p=='meta.json']
        if len(metadata)!=1: raise ValueError('expected one M4Singer meta.json')
        meta=extract_member(archive,metadata[0],root/'raw')
        prefix=str(Path(metadata[0]).parent).replace('\\','/')
        prefix='' if prefix=='.' else prefix+'/'
        rows=[m4singer_row(r) for r in json.loads(meta.read_text(encoding='utf-8'))]
        chosen=select_rows(rows,per_singer); name_set=set(names)
        members=[]
        for row in chosen:
            relative=prefix+row['source_audio']; safe_relative(relative)
            members.append(relative)
            row['source_audio']=relative
            base=str(Path(relative).with_suffix('')).replace('\\','/')
            for ext in ('.TextGrid','.json','.mid','.txt'):
                if base+ext in name_set: members.append(base+ext)
        subset_size=sum(archive.getinfo(n).file_size for n in members)+meta.stat().st_size
        if subset_size>1024**3:
            raise ValueError('selected uncompressed M4 subset exceeds 1 GiB pilot cap')
        free_space(root,sum(archive.getinfo(n).file_size for n in members))
        expected_size=source.size
        def get_chunk(chunk):
            with RangeReader(M4_URL,max_bytes=448*1024**2) as reader, zipfile.ZipFile(reader) as subset:
                if reader.size!=expected_size: raise ValueError('archive changed during download')
                for i,name in enumerate(chunk):
                    extract_member(subset,name,root/'raw',max_bytes=16*1024**2)
                    if i%100==0: print(f'm4singer worker files {i+1}/{len(chunk)}',flush=True)
                return reader.transferred
        with ThreadPoolExecutor(max_workers=4) as pool:
            transferred=sum(pool.map(get_chunk,[members[i::4] for i in range(4)]))+source.transferred
        jsonl(root/'selected.jsonl',chosen)
        write_json(root/'download-report.json',dict(corpus='m4singer',url=M4_URL,
            archive_bytes=source.size,transferred_bytes=transferred,metadata_sha256=sha256(meta),
            source_items=len(rows),selected=len(chosen),per_singer=per_singer,
            subset='deterministic song-diverse pilot, not full corpus',license='CC-BY-NC-SA-4.0',
            citation='https://openreview.net/forum?id=qiDmAaG6mP'))


def harvest(audio,sr):
    import pyworld
    return pyworld.harvest(audio.astype(np.float64),sr,frame_period=20)[0]


def process_row(row,raw,out,vocabulary,f0_function=harvest):
    validate_annotations(row)
    source=Path(raw)/safe_relative(row['source_audio'])
    audio,sr=sf.read(source,dtype='float64',always_2d=True)
    if audio.shape[1]!=1: raise ValueError('expected mono; no implicit downmix')
    audio=audio[:,0]
    if not np.isfinite(audio).all(): raise ValueError('nonfinite audio')
    if not .2<=len(audio)/sr<=13: raise ValueError('audio duration outside .2..13 seconds')
    if np.max(np.abs(audio))<1e-5: raise ValueError('silent audio')
    resample=lambda rate: resample_poly(audio,rate//math.gcd(sr,rate),sr//math.gcd(sr,rate))
    a16=resample(16000); a24=resample(24000)
    frames=math.ceil(len(a16)/320)
    phones=frame_phones(row['phones'],row['phone_durations'],frames,len(audio)/sr)
    phones=[model_phone(p,row['language']) for p in phones]
    unknown=sorted(set(phones)-vocabulary)
    f0=np.asarray(f0_function(a16,16000))
    extra=len(f0)-frames
    if extra not in (0,1): raise ValueError('F0 clock mismatch beyond one endpoint frame')
    f0=f0[:frames]
    if not np.isfinite(f0).all() or (f0<0).any(): raise ValueError('invalid F0')
    # Match upstream rounding, [45,850] clipping and mean-222 rescaling.
    quantized=np.round(f0).astype(int)
    quantized=np.where(quantized>0,np.clip(quantized,45,850),0)
    voiced=quantized[quantized>0]
    average=int(np.round(voiced.mean())) if len(voiced) else 0
    relative=np.round(quantized*222/average).astype(int) if average else np.zeros(frames,dtype=int)
    pitch_oov=bool((relative>850).any())
    directory=Path(out)/'items'/row['id']; directory.mkdir(parents=True,exist_ok=True)
    free_space(directory,2*1024**2)
    waveform=directory/'audio24k.wav'; features=directory/'features.npz'
    sf.write(waveform,a24,24000,subtype='FLOAT')
    np.savez_compressed(features,f0_hz=f0.astype(np.float32),f0_rescale=relative,
                        phone_frames=np.asarray(phones))
    result=dict(row,audio=str(waveform.resolve()),features=str(features.resolve()),
        source_audio_path=str(source.resolve()),source_audio_sha256=sha256(source),
        audio_sha256=sha256(waveform),source_sample_rate=sr,source_channels=1,
        audio_seconds=len(audio)/sr,n_frames=frames,gain=1.0,
        frame_rate=50,phone_duration_error_seconds=sum(row['phone_durations'])-len(audio)/sr,
        frame_grid='codec ceil(N16k/320), phones frame centers, Harvest t=0 at 20ms hop',
        f0_extra_endpoint_frames=extra,f0_avg=average,unknown_phones=unknown,
        f0_rescale_out_of_vocabulary=pitch_oov,
        prompt_singer_eligible=not unknown and average>0 and not pitch_oov,
        # Do not infer gender or high/low from a name or voice. Fixed prompt
        # bypasses these attributes in the RRQ-adapted upstream loader.
        fixed_prompt='A singer sings.',gender=None,pitch=None,
        normalization='none; no trim, stretch or loudness normalization')
    write_json(directory/'item.json',result)
    return result


def _worker(args):
    row,raw,out,vocab=args
    try: return process_row(row,raw,out,vocab),None
    except Exception as exc: return None,dict(id=row['id'],error=f'{type(exc).__name__}: {exc}')


def preprocess(root,out,dictionary,workers):
    root=Path(root); out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in (root/'selected.jsonl').read_text(encoding='utf-8').splitlines()]
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('duplicate sample IDs')
    vocab={line.rsplit(' ',1)[0] for line in Path(dictionary).read_text().splitlines()}
    accepted=[]; rejected=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(item,error) in enumerate(pool.map(_worker,[(r,root/'raw',out,vocab) for r in rows])):
            if error: rejected.append(error)
            else: accepted.append(item)
            if i%50==0: print(f'processed {i+1}/{len(rows)}; rejected={len(rejected)}',flush=True)
    jsonl(out/'manifest.jsonl',accepted); jsonl(out/'rejected.jsonl',rejected)
    counts=Counter(x['singer'] for x in accepted)
    fields=['item_name','task','audio_path','gender','phone','txt','f0_rescale','f0_avg','n_frames','pitch','prompt','volume']
    for split in ('train','valid','test'):
        with (out/(split+'.tsv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fields,delimiter='\t'); writer.writeheader()
            for r in accepted:
                if r['split']!=split or not r['prompt_singer_eligible']: continue
                with np.load(r['features']) as feat:
                    writer.writerow(dict(item_name=r['id'],task='text_to_acoustic_sing',audio_path=r['audio'],
                        gender='unspecified',phone=' '.join(feat['phone_frames']),txt='',
                        f0_rescale=' '.join(map(str,feat['f0_rescale'])),f0_avg=r['f0_avg'],
                        n_frames=r['n_frames'],pitch='unspecified',prompt=r['fixed_prompt'],volume='None'))
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True)
    write_json(out/'report.json',dict(accepted=len(accepted),rejected=len(rejected),singers=dict(counts),
        hours=sum(r['audio_seconds'] for r in accepted)/3600,
        prompt_singer_eligible=sum(r['prompt_singer_eligible'] for r in accepted),
        splits=dict(Counter(r['split'] for r in accepted)),
        unknown_phones=sorted({p for r in accepted for p in r['unknown_phones']}),
        rejection_reasons=dict(Counter(r['error'] for r in rejected)),
        implementation_commit=revision.stdout.strip(),dictionary_sha256=sha256(dictionary),
        input_manifest_sha256=sha256(root/'selected.jsonl'),
        caveats=['GT duration and extracted target F0: annotation-conditioned, not score-only',
                 'Provisional song-title split; not held-out singer; cross-language aliases unreviewed',
                 'Prompt-Singer pretraining overlap unverified; M4Singer is an upstream training corpus',
                 'Unknown phonemes kept in universal manifest, excluded from model TSV',
                 'Scores preserved with original units, not reinterpreted as per-phone durations'] ))
    print(json.dumps({'accepted':len(accepted),'rejected':len(rejected),'report':str(out/'report.json')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    fetch=sub.add_parser('fetch'); fetch.add_argument('--corpus',choices=['gtsinger','m4singer'],required=True)
    fetch.add_argument('--root',type=Path,required=True); fetch.add_argument('--per-singer',type=int,default=32)
    fetch.add_argument('--languages',nargs='+',choices=LANGUAGES,default=LANGUAGES)
    prep=sub.add_parser('preprocess'); prep.add_argument('--root',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True); prep.add_argument('--workers',type=int,default=4)
    prep.add_argument('--dictionary',type=Path,default=Path('data/dict_uni.txt'))
    args=parser.parse_args()
    if args.command=='fetch':
        if args.corpus=='gtsinger': fetch_gtsinger(args.root,args.per_singer,args.languages)
        else: fetch_m4singer(args.root,args.per_singer)
    else:
        if not 1<=args.workers<=8: parser.error('workers must be 1..8')
        preprocess(args.root,args.output,args.dictionary,args.workers)


if __name__=='__main__': main()
