"""Corpus adapters preserve annotation provenance; no inferred technique labels."""
import hashlib
import math
import unicodedata
from collections import defaultdict

import numpy as np


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def song_key(title):
    # Conservative title grouping, across singers/takes AND corpora. Translation
    # aliases and unrelated same-name songs still need manual benchmark review.
    title = unicodedata.normalize('NFKC', title).casefold()
    return ''.join(x for x in title if x.isalnum())


def base_row(corpus, name, singer, song, language, phones, durations):
    group = song_key(song)
    if not group: raise ValueError('empty song title')
    bucket = int(digest('rrq-song-v1:' + group)[:8], 16) % 100
    return dict(id=corpus+'-'+digest(name)[:24], corpus=corpus, source_id=name,
                singer=corpus+':'+singer, song_title=song, song_group=group,
                language=language, phones=list(phones), phone_durations=list(durations),
                split='test' if bucket < 10 else 'valid' if bucket < 20 else 'train',
                split_policy='title-group-v1-provisional-not-zero-shot',
                conditioning='annotation_conditioned', techniques=None)


def gtsinger_row(row):
    language,singer,tech,song,take,segment = row['item_name'].split('#')
    if 'Speech' in take: raise ValueError('speech is not singing')
    out=base_row('gtsinger',row['item_name'],singer,song,language,row['ph'],row['ph_durs'])
    out.update(source_audio=row['wav_fn'], lyrics=row.get('txt'), take=take,
        technique_group=tech, score={k:row[k] for k in ('ep_pitches','ep_notedurs','ep_types','ph2words') if k in row},
        score_provenance='GTSinger published retrospective realistic score',
        styles={k:row.get(k) for k in ('emotion','singing_method','pace','range')})
    labels={k:row[k] for k in ('mix_tech','falsetto_tech','breathy_tech','pharyngeal_tech','vibrato_tech','glissando_tech') if k in row}
    if labels: out['techniques']=labels
    return out


def m4singer_row(row):
    singer,song,segment=row['item_name'].split('#')
    out=base_row('m4singer',row['item_name'],singer,song,'Chinese',row['phs'],row['ph_dur'])
    out.update(source_audio=f'{singer}#{song}/{segment}.wav', lyrics=row.get('txt'),
        score={k:row[k] for k in ('notes','notes_dur','is_slur') if k in row},
        score_provenance='M4Singer performance-aligned annotated score', styles=None)
    return out


def select_rows(rows, per_singer):
    if per_singer <= 0: raise ValueError('per_singer must be positive')
    groups=defaultdict(list)
    for row in rows:
        d=np.asarray(row['phone_durations'],dtype=float)
        if (len(d)!=len(row['phones']) or not len(d) or not np.isfinite(d).all()
                or (d<0).any() or not .2 <= d.sum() <= 13):
            continue
        groups[row['singer']].append(row)
    out=[]
    for singer,items in sorted(groups.items()):
        # Round-robin songs prevents the pilot being dominated by one long song.
        songs=defaultdict(list)
        for r in sorted(items,key=lambda r:digest('rrq-pilot-v1:'+r['id'])):
            songs[r['song_group']].append(r)
        queues=[songs[k] for k in sorted(songs,key=digest)]
        chosen=[]
        while queues and len(chosen)<per_singer:
            for q in list(queues):
                chosen.append(q.pop(0))
                if not q: queues.remove(q)
                if len(chosen)==per_singer: break
        out.extend(chosen)
    return out


def model_phone(phone, language):
    if phone in ('<SP>','SP','sil','sp','SIL',''): return 'SP'
    if phone in ('<AP>','AP'): return 'AP'
    return phone[:-3] if language=='Chinese' and phone.endswith('_zh') else phone


def frame_phones(phones, durations, frames, audio_seconds, tolerance=.04):
    d=np.asarray(durations,dtype=float)
    if (len(d)!=len(phones) or not len(d) or not np.isfinite(d).all()
            or (d<0).any() or frames<=0 or not math.isfinite(audio_seconds)):
        raise ValueError('invalid phone duration annotation')
    if abs(float(d.sum())-audio_seconds)>tolerance:
        raise ValueError('phone duration differs from audio; no time stretching allowed')
    # A single cumulative clock avoids independent rounding drift. Only the
    # explicitly reported <=40ms endpoint tolerance is extended with last phone.
    times=(np.arange(frames)+.5)/50
    indices=np.minimum(np.searchsorted(np.cumsum(d),times,side='right'),len(phones)-1)
    return [phones[i] for i in indices]
