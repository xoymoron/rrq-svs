"""Codec-clock score conditioning, independent of Prompt-Singer's 50 Hz grid."""
import json
import hashlib
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from .io import safe_id


def annotation_hash(row):
    payload={k:row[k] for k in ('phones','phone_durations','score','language','singer','split')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('empty manifest or duplicate IDs')
    for row in rows:
        safe_id(row['id'])
    groups = {}
    for row in rows:
        previous = groups.setdefault(row['song_group'],row['split'])
        if previous != row['split']:
            raise ValueError('song leakage across splits')
    return rows


def load_audio(path, sample_rate):
    x, sr = sf.read(path,dtype='float32',always_2d=True)
    if x.shape[1] != 1 or not np.isfinite(x).all():
        raise ValueError('require finite mono audio; no silent downmix')
    factor = math.gcd(sr,sample_rate)
    return resample_poly(x[:,0],sample_rate//factor,sr//factor).astype(np.float32)


def frame_condition(row, frames, samples, sr, hop):
    duration = np.asarray(row['phone_durations'],dtype=np.float64)
    notes = np.asarray(row['score'].get('ep_pitches',row['score'].get('notes')),dtype=float)
    if (not len(duration) or len(duration)!=len(row['phones']) or
        notes.shape != duration.shape or not np.isfinite(duration).all() or (duration<0).any()):
        raise ValueError('invalid duration/phone/note correspondence')
    if abs(duration.sum()-samples/sr) > .04:
        raise ValueError('annotation duration mismatch; no time stretching')
    if not np.isfinite(notes).all() or (notes<0).any() or (notes>127).any() or (notes!=notes.round()).any():
        raise ValueError('MIDI notes must be integers in 0..127')
    if frames != math.ceil(samples/hop):
        raise ValueError('codec frame count mismatch')
    # Cumulative boundaries, preserving real time. Extend only endpoint padding.
    idx = np.searchsorted(np.cumsum(duration),(np.arange(frames)+.5)*hop/sr,side='right')
    idx = np.minimum(idx,len(duration)-1)
    return dict(phone_index=idx.astype(np.int64),midi=notes[idx].astype(np.int64))


def inventory(rows):
    # Transductive annotation inventory is disclosed; no acoustic statistics.
    return {key: sorted(set(values)) for key,values in {
        'phones':[r['language']+'::'+p for r in rows for p in r['phones']],
        'singers':[r['singer'] for r in rows],
        'languages':[r['language'] for r in rows]}.items()}


def score_batch(row, cache, vocab, device):
    maps = {k:{v:i for i,v in enumerate(values)} for k,values in vocab.items()}
    notes = row['score'].get('ep_pitches',row['score'].get('notes'))
    values = dict(phones=[maps['phones'][row['language']+'::'+p] for p in row['phones']],
        midi=notes,durations=row['phone_durations'],frame_index=cache['frame_index'].tolist())
    batch = {k:torch.tensor([v],device=device,dtype=torch.float32 if k=='durations' else torch.long)
             for k,v in values.items()}
    batch.update(singer=torch.tensor([maps['singers'][row['singer']]],device=device),
                 language=torch.tensor([maps['languages'][row['language']]],device=device))
    return batch
