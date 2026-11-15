"""Frozen, unprompted Chinese ASR diagnostics, calibrated with reference audio."""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import unicodedata

from .io import sha256,write_json


def normalize(value,converter=None):
    if converter is None:
        from opencc import OpenCC
        converter=OpenCC('t2s')
    text=''.join(value) if isinstance(value,list) else value
    text=converter.convert(unicodedata.normalize('NFKC',re.sub(r'<[^>]*>','',text))).lower()
    return ''.join(c for c in text if c.isalnum())


def edit_errors(reference,hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(hypothesis,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def cer(reference,hypothesis):
    if not reference:raise ValueError('empty reference transcript')
    return edit_errors(reference,hypothesis)/len(reference)


def gate(metrics,thresholds):
    for key,limit in (('reference_cer','reference_cer_max'),('codec_cer','codec_cer_max'),
                      ('generated_cer','generated_cer_max'),('generated_excess_cer','excess_cer_max')):
        value=metrics.get(key)
        if value is None or not math.isfinite(value) or value>thresholds[limit]:return False
    return True


@lru_cache(maxsize=2)
def recognizer(folder,device_index):
    from faster_whisper import WhisperModel
    root=Path(folder);provenance=json.loads((root/'provenance.json').read_text())
    if sha256(root/'model.bin')!=provenance['model_sha256']:raise ValueError('ASR model checksum mismatch')
    model=WhisperModel(str(root),device='cuda',device_index=device_index,compute_type='float16',cpu_threads=2)
    contract=dict(provenance,normalization='NFKC; t2s; lower; alnum; remove annotation tags',
                  faster_whisper=importlib.metadata.version('faster-whisper'),
                  ctranslate2=importlib.metadata.version('ctranslate2'),
                  language='zh',beam_size=5,temperature=0.,vad_filter=False,
                  condition_on_previous_text=False,initial_prompt=None)
    return model,contract


def transcribe(path,settings,device_index):
    model,contract=recognizer(settings['model_path'],device_index)
    key=hashlib.sha256(json.dumps([sha256(path),contract],sort_keys=True).encode()).hexdigest()
    cache=Path(settings['cache_path']);cache.mkdir(parents=True,exist_ok=True)
    output=cache/(key+'.json')
    if output.exists():return json.loads(output.read_text()),contract
    segments,_=model.transcribe(str(path),language='zh',beam_size=5,temperature=0.,
        vad_filter=False,condition_on_previous_text=False,initial_prompt=None)
    text=''.join(segment.text for segment in segments)
    result=dict(raw=text,normalized=normalize(text),key=key)
    write_json(output,result)
    return result,contract


def evaluate(lyrics,audio_paths,settings,device_index):
    reference=normalize(lyrics)
    if not reference:raise ValueError('empty normalized dataset lyrics')
    transcripts={};metrics={};contract=None
    for arm in ('reference','codec','generated'):
        transcripts[arm],contract=transcribe(audio_paths[arm],settings,device_index)
        errors=edit_errors(reference,transcripts[arm]['normalized'])
        metrics[arm+'_cer']=errors/len(reference);metrics[arm+'_errors']=errors
    metrics['characters']=len(reference)
    metrics['generated_excess_cer']=metrics['generated_cer']-metrics['reference_cer']
    return dict(lyrics=reference,transcripts=transcripts,metrics=metrics,contract=contract)


def aggregate(rows):
    if not rows:return {}
    characters=sum(row['metrics']['characters'] for row in rows)
    result={arm+'_cer':sum(row['metrics'][arm+'_errors'] for row in rows)/characters
            for arm in ('reference','codec','generated')}
    result['generated_excess_cer']=result['generated_cer']-result['reference_cer']
    result['characters']=characters
    return result
