"""Append-only multilingual inputs. Scores use the provided phoneme alignment.

No target audio/F0 is read by conditioning or pack. This is NOT a duration model.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from .corpora import frame_phones, model_phone
from .io import sha256, write_json


class Vocabulary:
    def __init__(self,tokens):
        self.tokens=list(tokens)
        if len(set(tokens))!=len(tokens): raise ValueError('duplicate vocabulary entries')
        self.index={t:i for i,t in enumerate(tokens)}

    @classmethod
    def original(cls,path):
        # Same special-token order as fairseq.data.Dictionary.__init__.
        return cls(['<s>','<pad>','</s>','<unk>']+
                   [s.rsplit(' ',1)[0] for s in Path(path).read_text(encoding='utf-8').splitlines()])

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8'))['tokens'])

    def extended(self,phones,pitches):
        new=set(phones)|{'f0_'+str(int(p)) for p in pitches}
        return Vocabulary(self.tokens+sorted(new-set(self.tokens)))

    def encode(self,tokens):
        missing=set(tokens)-self.index.keys()
        if missing: raise ValueError('unknown tokens: '+str(sorted(missing)))
        return [self.index[t] for t in tokens]


def conditioning(row):
    """Published note pitches on GT phone clock; no extraction from target F0."""
    notes=row.get('score',{}).get('ep_pitches',row.get('score',{}).get('notes'))
    if notes is None or len(notes)!=len(row['phones']):
        raise ValueError('missing or misaligned score pitches: '+row['id'])
    notes=np.asarray(notes,dtype=float)
    if not np.isfinite(notes).all() or (notes<0).any() or (notes>127).any() or (notes!=np.round(notes)).any():
        raise ValueError('score pitches must be MIDI integers 0..127 (0=rest)')
    if not row.get('lyrics') and not row.get('phones'):
        raise ValueError('missing lyrics and phonetic transcription')
    phones=frame_phones(row['phones'],row['phone_durations'],row['n_frames'],row['audio_seconds'])
    phones=[model_phone(p,row['language']) for p in phones]
    midi=np.asarray(frame_phones(notes,row['phone_durations'],row['n_frames'],row['audio_seconds']))
    hz=np.where(midi>0,440*2**((midi-69)/12),0)
    quantized=np.round(hz).astype(int)
    voiced=quantized[quantized>0]
    average=int(np.round(voiced.mean())) if len(voiced) else 0
    relative=np.round(quantized*222/average).astype(int) if average else np.zeros(len(midi),dtype=int)
    return phones,relative,average


def pack(row,vocab,codes=None):
    phones,pitch,bias=conditioning(row)
    # Native Prompt-Singer layout: every condition occupies a triplet/patch.
    items=['<text_to_acoustic_sing>','<prompt_start>']+['<continuous_token>']*77+[
        '<prompt_end>','<f0_rescale_start>']+['f0_'+str(p) for p in pitch]+[
        '<f0_rescale_end>','<text_start>']+phones+['<text_end>','<f0_bias_start>',
        'f0_'+str(bias),'<f0_bias_end>','<acoustic_start>']
    prefix=np.repeat(vocab.encode(items),3).tolist()
    start=len(prefix)
    mask=[False]*start
    if codes is not None:
        codes=np.asarray(codes)
        if codes.shape!=(3,row['n_frames']) or codes.dtype.kind not in 'iu' or (codes<0).any() or (codes>=1024).any():
            raise ValueError('codec target must be integer [3,frames] in [0,1024)')
        acoustic=['aco_'+str(int(codes[q,t])+q*1024) for t in range(codes.shape[1]) for q in range(3)]
        prefix+=vocab.encode(acoustic); mask += [True]*len(acoustic)
        prefix+=np.repeat(vocab.encode(['<acoustic_end>','</s>']),3).tolist(); mask += [False]*6
    return torch.tensor(prefix,dtype=torch.long),torch.tensor(mask,dtype=torch.bool),start


def read_rows(paths):
    return [json.loads(line) for p in paths for line in Path(p).read_text(encoding='utf-8').splitlines() if line.strip()]


def export_bundle(manifests,dictionary,output):
    rows=read_rows(manifests)
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('duplicate IDs')
    vocab=Vocabulary.original(dictionary); base_size=len(vocab.tokens)
    phones=set(); pitches=set(); processed=[]
    for r in rows:
        if not Path(r['audio']).is_file(): raise ValueError('missing waveform: '+r['id'])
        p,f,b=conditioning(r); phones.update(p); pitches.update(map(int,f)); pitches.add(b)
        role='auxiliary' if r['language']=='Chinese' else 'primary'
        processed.append(dict(r,dataset_role=role,conditioning='score_pitch_gt_phone_duration',
            lyric_representation='orthographic' if r.get('lyrics') else 'provided_phonetic_transcription',
            multilingual_eligible=True))
    vocab=vocab.extended(phones,pitches)
    for r in processed: pack(r,vocab)  # strict coverage; no silent UNK fallback
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'vocabulary.json',dict(tokens=vocab.tokens,base_size=base_size,
        base_dictionary_sha256=sha256(dictionary),inventory='all supplied annotations; no audio-derived tokens'))
    for name,selected in [('all',processed),('primary',[r for r in processed if r['dataset_role']=='primary']),
                          ('auxiliary',[r for r in processed if r['dataset_role']=='auxiliary'])]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    report=dict(total=len(rows),primary=sum(r['dataset_role']=='primary' for r in processed),
        auxiliary=sum(r['dataset_role']=='auxiliary' for r in processed),rejected=0,
        languages=dict(Counter(r['language'] for r in rows)),vocabulary_size=len(vocab.tokens),base_size=base_size,
        conditioning='score_pitch_gt_phone_duration',
        caveats=['Provided phoneme durations, not autonomous score-to-duration prediction',
                 'Original title-group split retained; not unseen-singer evaluation',
                 'New embeddings require adaptation; token coverage is not language quality',
                 'Phoneme inventory includes validation/test symbols; no target audio is used',
                 'M4Singer without orthographic text uses original phonetic lyrics, never invented words'])
    write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',nargs='+',type=Path,required=True)
    p.add_argument('--dictionary',type=Path,default=Path('data/dict_uni.txt'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(export_bundle(a.manifests,a.dictionary,a.output)),flush=True)


if __name__=='__main__': main()
